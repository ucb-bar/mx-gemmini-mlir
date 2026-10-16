"""Source-checked MX VPU unary arithmetic for captured graphs."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from math import prod
from pathlib import Path

from .target_profile import profile_sha256

_SIGMOID_PATTERN = ["arith.constant", "arith.negf", "math.exp",
                    "arith.addf", "arith.divf"]
_SQUARE_PATTERN = ["arith.constant", "math.powf"]
_GELU_PATTERN = (["arith.constant"] * 4 + ["arith.mulf"] * 3 +
                 ["arith.addf", "arith.mulf", "math.tanh", "arith.addf"] +
                 ["arith.mulf"] * 2)


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(prod(shape[axis + 1:]) for axis in range(len(shape)))


def selected_unary_sites(mlir_text: str, profile: dict) -> list[dict]:
    """Select exact pure unary generics with identity maps."""
    from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir

    if (not profile["resources"].get("vpu") or
            profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_columns") != 16):
        return []
    seen = set()
    ordinals = defaultdict(int)
    sites = []
    for op in parse_linalg_mlir(mlir_text)["ops"]:
        prov = op["prov"]
        source_region = prov.get("region_id")
        body_ops = op["body_ops"] if op["kind"] == "linalg.generic" else []
        operation = ("rsqrt" if body_ops == ["math.rsqrt"] else
                     "tanh" if body_ops == ["math.tanh"] else
                     "sigmoid" if body_ops == _SIGMOID_PATTERN else
                     "pow10000" if (body_ops == _SQUARE_PATTERN and
                                    prov.get("aten") == "aten.pow.Scalar") else
                     "square" if body_ops in (_SQUARE_PATTERN, ["arith.mulf"]) else
                     "gelu" if body_ops == _GELU_PATTERN else
                     "neg" if body_ops == ["arith.negf"] else None)
        if source_region and operation:
            ordinal = ordinals[source_region, operation]
            ordinals[source_region, operation] += 1
        else:
            ordinal = None
        if (op["kind"] != "linalg.generic" or operation is None or
                len(op["ins"]) != 1 or len(op["results"]) != 1 or
                (prov.get("op"), prov.get("aten")) not in (
                    {("rsqrt", "aten.rsqrt.default"),
                     ("layer_norm", "aten.native_layer_norm.default")}
                    if operation == "rsqrt" else
                    {("sigmoid", "aten.sigmoid.default")}
                    if operation == "sigmoid" else
                    {("tanh", "aten.tanh.default")}
                    if operation == "tanh" else
                    {("pow", "aten.pow.Tensor_Scalar")}
                    if operation == "square" else
                    {("neg", "aten.neg.default")}
                    if operation == "neg" else
                    {("gelu", "aten.gelu.default")}
                    if operation == "gelu" else
                    {("pow", "aten.pow.Scalar")})):
            continue
        body = op["scalar_body"] or {}
        operations = body.get("operations") or ()
        if (body.get("captures") or body.get("effects") or
                any(item.get("effects") or item.get("regions") for item in operations)):
            continue
        if operation in {"rsqrt", "neg", "tanh"}:
            if (len(operations) != 1 or operations[0].get("operands") != ["arg:0"] or
                    body.get("yields") != ["op:0:0"]):
                continue
        elif operation == "sigmoid" and (len(operations) != 5 or
              operations[0].get("attributes", {}).get("value") != "1.000000e+00 : f32" or
              [item.get("operands") for item in operations] !=
              [[], ["arg:0"], ["op:1:0"], ["op:0:0", "op:2:0"],
               ["op:0:0", "op:3:0"]] or
              body.get("yields") != ["op:4:0"]):
            continue
        elif operation == "square":
            power_two = (len(operations) == 2 and
                         operations[0].get("attributes", {}).get("value") ==
                         "2.000000e+00 : f32" and
                         [item.get("operands") for item in operations] ==
                         [[], ["arg:0", "op:0:0"]] and
                         body.get("yields") == ["op:1:0"])
            self_multiply = (len(operations) == 1 and
                             operations[0].get("operands") == ["arg:0", "arg:0"] and
                             body.get("yields") == ["op:0:0"])
            if not (power_two or self_multiply):
                continue
        elif operation == "pow10000" and (len(operations) != 2 or
              operations[0].get("attributes", {}).get("value") != "1.000000e+04 : f32" or
              [step.get("operands") for step in operations] !=
              [[], ["op:0:0", "arg:0"]] or
              body.get("yields") != ["op:1:0"]):
            continue
        elif operation == "gelu":
            expected_values = ("5.000000e-01 : f32", "1.000000e+00 : f32",
                               "4.471500e-02 : f32", "0.797884583 : f32")
            expected_operands = ([[]] * 4 +
                                 [["arg:0", "arg:0"], ["op:4:0", "arg:0"],
                                  ["op:2:0", "op:5:0"], ["arg:0", "op:6:0"],
                                  ["op:3:0", "op:7:0"], ["op:8:0"],
                                  ["op:1:0", "op:9:0"],
                                  ["op:0:0", "arg:0"],
                                  ["op:11:0", "op:10:0"]])
            if (len(operations) != 13 or
                    tuple(step.get("attributes", {}).get("value")
                          for step in operations[:4]) != expected_values or
                    [step.get("operands") for step in operations] != expected_operands or
                    body.get("yields") != ["op:12:0"]):
                continue
        result = op["results"][0]
        shape = tuple(result["shape"])
        dtype = result["dtype"]
        if (not 1 <= len(shape) <= 4 or any(type(dim) is not int or dim <= 0 for dim in shape)
                or dtype not in {"bf16", "f32"} or
                tuple(op["ins"][0]["shape"]) != shape or
                op["ins"][0]["dtype"] != dtype or
                op["iterator_types"] != ["parallel"] * len(shape)):
            continue
        axes = ", ".join(f"d{i}" for i in range(len(shape)))
        identity = f"affine_map<({axes}) -> ({axes})>"
        if op["indexing_maps"] != [identity, identity]:
            continue
        region = f"{source_region}#{operation}{ordinal}"
        if not source_region or region in seen:
            raise ValueError("MX VPU unary has no unique source operation")
        seen.add(region)
        symbol = f"mx_vpu_{operation}_" + dtype + "_" + "_".join(map(str, shape))
        sites.append({
            "region": region, "source_region": source_region,
            "shape": shape, "dtype": dtype, "operation": operation, "symbol": symbol,
            "kernel_symbol": symbol + "_kernel",
        })
    return sites


def emit_memref_unary_shim(site: dict) -> str:
    """Emit the ranked C interface and return descriptor for one unary shape."""
    shape = tuple(site["shape"])
    rank = len(shape)
    symbol, kernel = site["symbol"], site["kernel_symbol"]
    scalar = "float" if site["dtype"] == "f32" else "uint16_t"
    strides = _strides(shape)
    dims = ", ".join(map(str, shape))
    steps = ", ".join(map(str, strides))

    def params(name: str) -> list[str]:
        return [f"void *{name}_alloc", f"void *{name}_aligned", f"intptr_t {name}_off",
                *(f"intptr_t {name}_s{i}" for i in range(rank)),
                *(f"intptr_t {name}_st{i}" for i in range(rank))]

    declarations = ",\n    ".join((*params("a"), *params("b")))
    checks = " ||\n      ".join(
        [f"{name}_s{i} != {dim}" for name in ("a", "b")
         for i, dim in enumerate(shape)] +
        [f"{name}_st{i} != {step}" for name in ("a", "b")
         for i, step in enumerate(strides)])
    call = ",\n      ".join(", ".join(
        [f"{name}->allocated", f"{name}->aligned", f"{name}->offset",
         *(f"{name}->sizes[{i}]" for i in range(rank)),
         *(f"{name}->strides[{i}]" for i in range(rank))])
        for name in ("a", "b"))
    stores = "\n".join(
        f"  destination->{field} = returned.{field};"
        for field in ("allocated", "aligned", "offset",
                      *(f"sizes[{i}]" for i in range(rank)),
                      *(f"strides[{i}]" for i in range(rank))))
    return f'''#include <stdint.h>
typedef struct {{ void *allocated, *aligned; intptr_t offset;
                 intptr_t sizes[{rank}], strides[{rank}]; }} mx_memref_{rank}d;
extern void {kernel}(const {scalar} *, {scalar} *);
mx_memref_{rank}d {symbol}(\n    {declarations}) {{
  if (!a_aligned || !b_aligned || a_off < 0 || b_off < 0 ||
      {checks}) __builtin_trap();
  {kernel}((const {scalar} *)a_aligned + a_off,
           ({scalar} *)b_aligned + b_off);
  return (mx_memref_{rank}d){{b_alloc, b_aligned, b_off,
                              {{{dims}}}, {{{steps}}}}};
}}
void _mlir_ciface_{symbol}(mx_memref_{rank}d *result,
                          const mx_memref_{rank}d *a,
                          const mx_memref_{rank}d *b) {{
  if (!result || !a || !b) __builtin_trap();
  mx_memref_{rank}d returned = {symbol}(\n      {call});
  volatile mx_memref_{rank}d *destination = (volatile mx_memref_{rank}d *)result;
{stores}
}}
'''


class MXVPUUnaryPostOffload:
    """Replace exact source rsqrt generics with destination-passing VPU calls."""

    def __init__(self, profile: dict, *, enabled_regions: frozenset[str] | None = None):
        self.profile = profile
        self.enabled_regions = enabled_regions

    def __call__(self, prepared: Path, sidecar: Path, workdir: Path) -> Path:
        from xdsl.dialects import func, tensor
        from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
        from xdsl.ir import Region

        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text

        prepared, workdir = Path(prepared), Path(workdir)
        source = prepared.read_text()
        all_sites = selected_unary_sites(source, self.profile)
        if self.enabled_regions is not None and not self.enabled_regions <= {
                site["region"] for site in all_sites}:
            raise ValueError("MX VPU unary policy names unknown source regions")
        sites = [site for site in all_sites if self.enabled_regions is None or
                 site["region"] in self.enabled_regions]
        if not sites:
            return prepared
        module = parse_mlir_file(prepared)
        forward = [op for op in module.walk()
                   if isinstance(op, func.FuncOp) and op.body.blocks and
                   op.sym_name.data == "forward"]
        if len(forward) != 1:
            raise ValueError("MX VPU unary rewrite needs one forward function")
        block = forward[0].body.block
        selected = {site["region"]: site for site in sites}
        candidates = {}
        ordinals = defaultdict(int)
        for op in block.ops:
            if op.name != "linalg.generic":
                continue
            source_region = getattr(op.attributes.get("prov.region_id"), "data", None)
            if not source_region:
                continue
            body_ops = [step.name for step in op.body.blocks[0].ops
                        if step.name != "linalg.yield"]
            aten = getattr(op.attributes.get("prov.aten"), "data", None)
            operation = ("rsqrt" if body_ops == ["math.rsqrt"] else
                         "tanh" if body_ops == ["math.tanh"] else
                         "sigmoid" if body_ops == _SIGMOID_PATTERN else
                         "pow10000" if (body_ops == _SQUARE_PATTERN and
                                        aten == "aten.pow.Scalar") else
                         "square" if body_ops in (_SQUARE_PATTERN, ["arith.mulf"]) else
                         "neg" if body_ops == ["arith.negf"] else
                         "gelu" if body_ops == _GELU_PATTERN else None)
            if operation is None:
                continue
            region = f"{source_region}#{operation}{ordinals[source_region, operation]}"
            ordinals[source_region, operation] += 1
            if region in selected:
                candidates.setdefault(region, []).append(op)
        if set(candidates) != set(selected) or any(
                len(group) != 1 for group in candidates.values()):
            raise ValueError("MX VPU unary source operations changed after routing")
        declarations = {}
        records = []
        for site in sites:
            op = candidates[site["region"]][0]
            shape_type = f"tensor<{'x'.join(map(str, site['shape']))}x{site['dtype']}>"
            if (len(op.inputs) != 1 or len(op.results) != 1 or
                    str(op.inputs[0].type) != shape_type or
                    str(op.results[0].type) != shape_type):
                raise ValueError("MX VPU unary tensor type changed")
            result_type = op.results[0].type
            destination = tensor.EmptyOp((), result_type)
            call = func.CallOp(site["symbol"], [op.inputs[0], destination.results[0]],
                               [result_type])
            block.insert_op_before(destination, op)
            block.insert_op_before(call, op)
            op.results[0].replace_all_uses_with(call.results[0])
            op.detach()
            op.erase()
            signature = tuple(value.type for value in call.arguments)
            previous = declarations.setdefault(site["symbol"], signature)
            if previous != signature:
                raise ValueError("MX VPU rsqrt symbol has conflicting signatures")
            records.append({"region": site["region"], "symbol": site["symbol"],
                            "operation": site["operation"]})
        read = DictionaryAttr({"bufferization.access": StringAttr("read")})
        write = DictionaryAttr({"bufferization.access": StringAttr("write")})
        for symbol, signature in declarations.items():
            module.body.block.add_op(func.FuncOp(
                symbol, (signature, (signature[-1],)), Region(),
                visibility="private", arg_attrs=ArrayAttr([read, write])))
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / "mx_vpu_unary.mlir"
        out.write_text(text(module))
        receipt = {
            "schema": "mx_gemmini.merlin_vpu_unary_rewrite.v1",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "sidecar_sha256": hashlib.sha256(Path(sidecar).read_bytes()).hexdigest(),
            "profile_sha256": profile_sha256(self.profile),
            "numeric_policy": "f32_to_bf16_rne_vpu_unary_to_f32_or_direct_bf16",
            "regions": records,
        }
        (workdir / "mx_vpu_unary_rewrite.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return out
