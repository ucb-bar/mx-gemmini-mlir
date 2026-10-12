"""Bind a captured softmax decomposition to a profile-specific VPU kernel.

Merlin owns the generic post-offload callback. This target package recognizes
one exact model2MLIR f32 row-softmax decomposition and replaces each complete
region with a destination-passing external call. The existing scalar lowering
remains available if the target does not select this callback.
"""

from __future__ import annotations

import hashlib
import json
from math import prod
from pathlib import Path

from .merlin_vpu_inventory import audit_softmax_boundaries
from .target_profile import profile_sha256


_PATTERN = (
    "arith.constant", "tensor.splat", "linalg.reduce",
    "tensor.collapse_shape", "tensor.expand_shape", "tensor.empty",
    "linalg.generic", "tensor.empty", "linalg.generic",
    "arith.constant", "tensor.splat", "linalg.reduce",
    "tensor.collapse_shape", "tensor.expand_shape", "tensor.empty",
    "linalg.generic",
)
_HELPERS = {"linalg.reduce", "tensor.empty"}


def symbol_for_shape(shape: tuple[int, ...]) -> str:
    if len(shape) != 4 or any(type(dim) is not int or dim <= 0 for dim in shape):
        raise ValueError("MX VPU softmax needs a static rank-four source shape")
    return "mx_vpu_softmax_" + "_".join(str(dim) for dim in shape)


def kernel_symbol_for_shape(shape: tuple[int, ...]) -> str:
    return symbol_for_shape(shape) + "_kernel"


def selected_softmax_sites(mlir_text: str, profile: dict) -> list[dict]:
    """Select f32 rank-four row softmax boundaries under this VPU profile."""
    if (not profile["resources"].get("vpu") or
            profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_columns") != 16):
        return []
    inventory = audit_softmax_boundaries(mlir_text, profile)
    sites = []
    for row in inventory["regions"]:
        shape = tuple(row["input_shape"])
        if (len(shape) != 4 or shape != tuple(row["output_shape"]) or
                row["input_dtype"] != "f32" or row["output_dtype"] != "f32"):
            continue
        if row["row_width"] > 8184:
            continue
        sites.append({"region": row["region"], "shape": shape,
                      "logical_rows": prod(shape[:-1]),
                      "row_width": shape[-1],
                      "symbol": symbol_for_shape(shape),
                      "kernel_symbol": kernel_symbol_for_shape(shape)})
    return sites


def emit_memref_softmax_shim(shape: tuple[int, ...]) -> str:
    """Emit the rank-four destination-passing ABI shim used by func.call."""
    symbol, kernel = symbol_for_shape(shape), kernel_symbol_for_shape(shape)
    strides = (shape[1] * shape[2] * shape[3],
               shape[2] * shape[3], shape[3], 1)
    def params(prefix: str) -> str:
        return ",\n    ".join((f"void *{prefix}_alloc", f"void *{prefix}_aligned",
                             f"intptr_t {prefix}_off",
                             *(f"intptr_t {prefix}_s{i}" for i in range(4)),
                             *(f"intptr_t {prefix}_st{i}" for i in range(4))))
    checks = " ||\n      ".join(
        [f"a_s{i} != {dim}" for i, dim in enumerate(shape)] +
        [f"b_s{i} != {dim}" for i, dim in enumerate(shape)] +
        [f"a_st{i} != {stride}" for i, stride in enumerate(strides)] +
        [f"b_st{i} != {stride}" for i, stride in enumerate(strides)])
    dims = ", ".join(str(dim) for dim in shape)
    step = ", ".join(str(stride) for stride in strides)
    groups = []
    for name in ("a", "b"):
        fields = [f"{name}->{field}" for field in ("allocated", "aligned", "offset")]
        fields.extend(f"{name}->sizes[{axis}]" for axis in range(4))
        fields.extend(f"{name}->strides[{axis}]" for axis in range(4))
        groups.append(", ".join(fields))
    call = ",\n      ".join(groups)
    stores = "\n".join(
        f"  destination->{field} = returned.{field};"
        for field in ("allocated", "aligned", "offset",
                      *(f"sizes[{axis}]" for axis in range(4)),
                      *(f"strides[{axis}]" for axis in range(4))))
    return f'''#include <stdint.h>
typedef struct {{
  void *allocated;
  void *aligned;
  intptr_t offset;
  intptr_t sizes[4];
  intptr_t strides[4];
}} mx_memref_4d;
extern void {kernel}(const float *, float *);
mx_memref_4d {symbol}(\n    {params("a")},\n    {params("b")}) {{
  if (!a_aligned || !b_aligned || a_off < 0 || b_off < 0 ||
      {checks}) __builtin_trap();
  {kernel}((const float *)a_aligned + a_off, (float *)b_aligned + b_off);
  return (mx_memref_4d){{b_alloc, b_aligned, b_off,
                         {{{dims}}}, {{{step}}}}};
}}
void _mlir_ciface_{symbol}(mx_memref_4d *result,
                          const mx_memref_4d *a, const mx_memref_4d *b) {{
  if (!result || !a || !b) __builtin_trap();
  mx_memref_4d returned = {symbol}(\n      {call});
  volatile mx_memref_4d *destination = (volatile mx_memref_4d *)result;
{stores}
}}
'''


class MXVPUSoftmaxPostOffload:
    """Replace exactly matched softmax regions after matrix catalog routing."""

    def __init__(self, profile: dict):
        self.profile = profile

    def __call__(self, prepared: Path, sidecar: Path, workdir: Path) -> Path:
        from xdsl.dialects import func, tensor
        from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr, TensorType, f32
        from xdsl.ir import Region

        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text

        prepared, workdir = Path(prepared), Path(workdir)
        source = prepared.read_text()
        selected = {row["region"]: row for row in
                    selected_softmax_sites(source, self.profile)}
        if not selected:
            return prepared
        module = parse_mlir_file(prepared)
        definitions = [op for op in module.walk()
                       if isinstance(op, func.FuncOp) and op.body.blocks]
        forward = [op for op in definitions if op.sym_name.data == "forward"]
        if len(forward) != 1:
            raise ValueError("MX VPU rewrite needs one model forward function")
        block = forward[0].body.block
        ops = list(block.ops)
        by_region: dict[str, list[int]] = {}
        for index, op in enumerate(ops):
            region = getattr(op.attributes.get("prov.region_id"), "data", None)
            if region in selected:
                by_region.setdefault(region, []).append(index)
        if set(by_region) != set(selected):
            raise ValueError("MX VPU selected softmax source regions changed")
        declarations: dict[str, tuple] = {}
        records = []
        for region, row in selected.items():
            indices = by_region[region]
            members = ops[indices[0]:indices[-1] + 1]
            if tuple(op.name for op in members) != _PATTERN:
                raise ValueError(f"MX VPU softmax {region} changed its source decomposition")
            for op in members:
                marker = getattr(op.attributes.get("prov.region_id"), "data", None)
                if marker != region and (marker is not None or op.name not in _HELPERS):
                    raise ValueError(f"MX VPU softmax {region} contains foreign source semantics")
            first, last = members[6], members[-1]
            if (not isinstance(first, type(last)) or len(first.inputs) != 2 or
                    len(last.results) != 1 or not first.inputs or
                    first.inputs[0].type != last.results[0].type or
                    not isinstance(first.inputs[0].type, TensorType) or
                    first.inputs[0].type.element_type != f32 or
                    tuple(int(dim) for dim in first.inputs[0].type.get_shape()) != row["shape"]):
                raise ValueError(f"MX VPU softmax {region} lost its f32 tensor boundary")
            member_set = set(members)
            if any(use.operation not in member_set
                   for op in members[:-1] for value in op.results for use in value.uses):
                raise ValueError(f"MX VPU softmax {region} has an escaping intermediate")
            score = first.inputs[0]
            if getattr(score, "owner", None) in member_set:
                raise ValueError(f"MX VPU softmax {region} consumes its own intermediate")
            result_type = last.results[0].type
            destination = tensor.EmptyOp((), result_type)
            call = func.CallOp(row["symbol"], [score, destination.results[0]],
                               [result_type])
            block.insert_op_before(destination, members[0])
            block.insert_op_before(call, members[0])
            last.results[0].replace_all_uses_with(call.results[0])
            for op in reversed(members):
                op.detach()
                op.erase()
            signature = (score.type, result_type)
            previous = declarations.setdefault(row["symbol"], signature)
            if previous != signature:
                raise ValueError("MX VPU shared symbol has conflicting tensor signatures")
            records.append({"region": region, "shape": list(row["shape"]),
                            "symbol": row["symbol"]})
        read = DictionaryAttr({"bufferization.access": StringAttr("read")})
        write = DictionaryAttr({"bufferization.access": StringAttr("write")})
        for symbol, (input_type, result_type) in declarations.items():
            module.body.block.add_op(func.FuncOp(
                symbol, ((input_type, result_type), (result_type,)), Region(),
                visibility="private", arg_attrs=ArrayAttr([read, write])))
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / "mx_vpu_softmax.mlir"
        out.write_text(text(module))
        receipt = {
            "schema": "mx_gemmini.merlin_vpu_softmax_rewrite.v1",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "sidecar_sha256": hashlib.sha256(Path(sidecar).read_bytes()).hexdigest(),
            "profile_sha256": profile_sha256(self.profile),
            "numeric_policy": "f32_scores_to_bf16_rne_vpu_round_each_step_to_f32_probabilities",
            "regions": records,
        }
        (workdir / "mx_vpu_softmax_rewrite.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return out
