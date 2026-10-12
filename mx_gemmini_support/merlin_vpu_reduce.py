"""Source-checked last-axis MX VPU reductions outside fused softmax."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from math import inf, prod
from pathlib import Path

from .target_profile import profile_sha256


_BODY = {"arith.addf": "rsum", "arith.maximumf": "rmax"}


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(prod(shape[axis + 1:]) for axis in range(len(shape)))


def selected_row_reduce_sites(mlir_text: str, profile: dict) -> list[dict]:
    """Select pure zero/-inf initialized reductions with no softmax consumer."""
    from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir

    if (not profile["resources"].get("vpu") or
            profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_columns") != 16):
        return []
    inventory = parse_linalg_mlir(mlir_text)
    consumers = defaultdict(list)
    for op in inventory["ops"]:
        for item in op["ins"]:
            source = item.get("source")
            if source and source[0] == "op":
                consumers[source[1]].append(op)
    ordinals = defaultdict(int)
    sites = []
    for op in inventory["ops"]:
        if (op["kind"] != "linalg.reduce" or len(op["body_ops"]) != 1 or
                op["body_ops"][0] not in _BODY or len(op["ins"]) != 1 or
                len(op["outs"]) != 1 or len(op["results"]) != 1 or
                any(consumer["prov"].get("op") == "softmax"
                    for consumer in consumers[op["id"]])):
            continue
        operation = _BODY[op["body_ops"][0]]
        body = op["scalar_body"] or {}
        operations = body.get("operations") or ()
        if (len(operations) != 1 or
                operations[0].get("operands") != ["arg:0", "arg:1"] or
                body.get("yields") != ["op:0:0"] or body.get("captures") or
                body.get("effects") or operations[0].get("effects") or
                operations[0].get("regions")):
            continue
        input_item, output_item, result = op["ins"][0], op["outs"][0], op["results"][0]
        shape = tuple(input_item["shape"])
        output_shape = tuple(result["shape"])
        identity = 0.0 if operation == "rsum" else -inf
        if (not 2 <= len(shape) <= 4 or any(type(dim) is not int or dim <= 0 for dim in shape)
                or shape[-1] > 8184 or output_shape != shape[:-1] or
                tuple(output_item["shape"]) != output_shape or
                any(item["dtype"] != "f32" for item in (input_item, output_item, result)) or
                op["reduction_dims"] != [len(shape) - 1] or
                output_item.get("source") != ("init", "splat") or
                output_item.get("const_value") != identity):
            continue
        ordinal = ordinals[operation, shape]
        ordinals[operation, shape] += 1
        name = (f"mx_vpu_{operation}_" + "_".join(map(str, shape)))
        sites.append({
            "region": f"{operation}:{'x'.join(map(str, shape))}:{ordinal}",
            "operation": operation, "shape": shape,
            "output_shape": output_shape, "logical_rows": prod(output_shape),
            "row_width": shape[-1], "symbol": name,
            "kernel_symbol": name + "_kernel",
        })
    return sites


def emit_memref_reduce_shim(site: dict) -> str:
    """Emit ranked input/output C ABI including the return descriptor."""
    shape, output_shape = tuple(site["shape"]), tuple(site["output_shape"])
    rank_in, rank_out = len(shape), len(output_shape)
    symbol, kernel = site["symbol"], site["kernel_symbol"]
    typedefs = "\n".join(
        f"typedef struct {{ void *allocated, *aligned; intptr_t offset; "
        f"intptr_t sizes[{rank}], strides[{rank}]; }} mx_memref_{rank}d;"
        for rank in sorted({rank_in, rank_out}))
    groups = (("a", shape), ("b", output_shape))

    def params(name: str, dims: tuple[int, ...]) -> list[str]:
        return [f"void *{name}_alloc", f"void *{name}_aligned", f"intptr_t {name}_off",
                *(f"intptr_t {name}_s{i}" for i in range(len(dims))),
                *(f"intptr_t {name}_st{i}" for i in range(len(dims)))]

    declarations = ",\n    ".join(item for name, dims in groups
                                     for item in params(name, dims))
    checks = " ||\n      ".join(
        [f"{name}_s{i} != {dim}" for name, dims in groups
         for i, dim in enumerate(dims)] +
        [f"{name}_st{i} != {step}" for name, dims in groups
         for i, step in enumerate(_strides(dims))])
    dims = ", ".join(map(str, output_shape))
    steps = ", ".join(map(str, _strides(output_shape)))
    call = ",\n      ".join(", ".join(
        [f"{name}->allocated", f"{name}->aligned", f"{name}->offset",
         *(f"{name}->sizes[{i}]" for i in range(len(dims))),
         *(f"{name}->strides[{i}]" for i in range(len(dims)))])
        for name, dims in groups)
    stores = "\n".join(
        f"  destination->{field} = returned.{field};"
        for field in ("allocated", "aligned", "offset",
                      *(f"sizes[{i}]" for i in range(rank_out)),
                      *(f"strides[{i}]" for i in range(rank_out))))
    return f'''#include <stdint.h>
{typedefs}
extern void {kernel}(const float *, float *);
mx_memref_{rank_out}d {symbol}(\n    {declarations}) {{
  if (!a_aligned || !b_aligned || a_off < 0 || b_off < 0 ||
      {checks}) __builtin_trap();
  {kernel}((const float *)a_aligned + a_off, (float *)b_aligned + b_off);
  return (mx_memref_{rank_out}d){{b_alloc, b_aligned, b_off,
                                  {{{dims}}}, {{{steps}}}}};
}}
void _mlir_ciface_{symbol}(mx_memref_{rank_out}d *result,
                          const mx_memref_{rank_in}d *a,
                          const mx_memref_{rank_out}d *b) {{
  if (!result || !a || !b) __builtin_trap();
  mx_memref_{rank_out}d returned = {symbol}(\n      {call});
  volatile mx_memref_{rank_out}d *destination = (volatile mx_memref_{rank_out}d *)result;
{stores}
}}
'''


class MXVPUReducePostOffload:
    """Replace verified source reductions with destination-passing VPU calls."""

    def __init__(self, profile: dict):
        self.profile = profile

    def __call__(self, prepared: Path, sidecar: Path, workdir: Path) -> Path:
        from xdsl.dialects import func, tensor
        from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
        from xdsl.ir import Region

        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text

        prepared, workdir = Path(prepared), Path(workdir)
        source = prepared.read_text()
        sites = selected_row_reduce_sites(source, self.profile)
        if not sites:
            return prepared
        module = parse_mlir_file(prepared)
        forward = [op for op in module.walk()
                   if isinstance(op, func.FuncOp) and op.body.blocks and
                   op.sym_name.data == "forward"]
        if len(forward) != 1:
            raise ValueError("MX VPU reduce rewrite needs one forward function")
        block = forward[0].body.block
        selected = {site["region"]: site for site in sites}
        candidates = {}
        ordinals = defaultdict(int)
        for op in block.ops:
            if op.name != "linalg.reduce" or len(op.operands) != 2:
                continue
            body_ops = [step.name for step in op.regions[0].blocks[0].ops
                        if step.name != "linalg.yield"]
            if len(body_ops) != 1 or body_ops[0] not in _BODY:
                continue
            operation = _BODY[body_ops[0]]
            init_owner = getattr(op.operands[1], "owner", None)
            const_owner = (getattr(init_owner.operands[0], "owner", None)
                           if getattr(init_owner, "name", None) == "tensor.splat"
                           and len(init_owner.operands) == 1 else None)
            identity = ("0.000000e+00 : f32" if operation == "rsum"
                        else "0xff800000 : f32")
            if (getattr(const_owner, "name", None) != "arith.constant" or
                    str(getattr(const_owner, "value", None)) != identity):
                continue
            shape_type = str(op.operands[0].type)
            if not shape_type.startswith("tensor<") or not shape_type.endswith("xf32>"):
                continue
            shape = tuple(int(dim) for dim in shape_type[7:-5].split("x"))
            region = f"{operation}:{'x'.join(map(str, shape))}:{ordinals[operation, shape]}"
            if region in selected:
                candidates.setdefault(region, []).append(op)
                ordinals[operation, shape] += 1
        if set(candidates) != set(selected) or any(
                len(group) != 1 for group in candidates.values()):
            raise ValueError("MX VPU reduction source sites changed after routing")
        declarations = {}
        records = []
        for site in sites:
            op = candidates[site["region"]][0]
            input_type = f"tensor<{'x'.join(map(str, site['shape']))}xf32>"
            output_type = f"tensor<{'x'.join(map(str, site['output_shape']))}xf32>"
            if (str(op.operands[0].type) != input_type or
                    str(op.results[0].type) != output_type):
                raise ValueError("MX VPU reduction tensor types changed")
            result_type = op.results[0].type
            destination = tensor.EmptyOp((), result_type)
            call = func.CallOp(site["symbol"],
                               [op.operands[0], destination.results[0]],
                               [result_type])
            block.insert_op_before(destination, op)
            block.insert_op_before(call, op)
            op.results[0].replace_all_uses_with(call.results[0])
            op.detach()
            op.erase()
            signature = tuple(value.type for value in call.arguments)
            previous = declarations.setdefault(site["symbol"], signature)
            if previous != signature:
                raise ValueError("MX VPU reduction symbol has conflicting signatures")
            records.append({"region": site["region"], "symbol": site["symbol"],
                            "operation": site["operation"]})
        read = DictionaryAttr({"bufferization.access": StringAttr("read")})
        write = DictionaryAttr({"bufferization.access": StringAttr("write")})
        for symbol, signature in declarations.items():
            module.body.block.add_op(func.FuncOp(
                symbol, (signature, (signature[-1],)), Region(),
                visibility="private", arg_attrs=ArrayAttr([read, write])))
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / "mx_vpu_reduce.mlir"
        out.write_text(text(module))
        receipt = {
            "schema": "mx_gemmini.merlin_vpu_reduce_rewrite.v1",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "sidecar_sha256": hashlib.sha256(Path(sidecar).read_bytes()).hexdigest(),
            "profile_sha256": profile_sha256(self.profile),
            "numeric_policy": "f32_to_bf16_rne_vpu_reduce_to_f32",
            "regions": records,
        }
        (workdir / "mx_vpu_reduce_rewrite.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return out
