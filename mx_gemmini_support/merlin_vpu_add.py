"""Source-bound MX VPU addition for BF16/f32 tensor and bias boundaries."""

from __future__ import annotations

import hashlib
import json
from math import prod
from pathlib import Path

from .target_profile import profile_sha256


def _symbol(shape: tuple[int, ...], dtype: str, broadcast: str) -> str:
    if (not 2 <= len(shape) <= 4 or any(type(dim) is not int or dim <= 0 for dim in shape)
            or dtype not in {"bf16", "f32"}
            or broadcast not in {"none", "last_dimension"}):
        raise ValueError("MX VPU add has an unsupported source signature")
    return "mx_vpu_add_" + dtype + "_" + (
        "bias_" if broadcast == "last_dimension" else "same_") + "_".join(str(dim) for dim in shape)


def selected_add_sites(mlir_text: str, profile: dict) -> list[dict]:
    """Select pure source addf generics with exact identity or bias maps."""
    from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir

    if (not profile["resources"].get("vpu") or
            profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_columns") != 16):
        return []
    inventory = parse_linalg_mlir(mlir_text)
    sites = []
    seen = set()
    for op in inventory["ops"]:
        prov = op["prov"]
        if (op["kind"] != "linalg.generic" or prov.get("op") not in {"add", "addmm"}
                or prov.get("aten") not in {"aten.add.Tensor", "aten.addmm.default"}
                or op["body_ops"] != ["arith.addf"]
                or len(op["ins"]) != 2 or len(op["results"]) != 1):
            continue
        body = op["scalar_body"] or {}
        operations = body.get("operations") or ()
        if (len(operations) != 1 or operations[0].get("operands") != ["arg:0", "arg:1"]
                or body.get("yields") != ["op:0:0"]
                or body.get("captures") or body.get("effects") or
                operations[0].get("effects") or operations[0].get("regions")):
            continue
        a, b, result = op["ins"][0], op["ins"][1], op["results"][0]
        shape = tuple(result["shape"])
        dtype = result["dtype"]
        if (not 2 <= len(shape) <= 4 or any(type(d) is not int or d <= 0 for d in shape)
                or dtype not in {"bf16", "f32"}
                or a["dtype"] != dtype or b["dtype"] != dtype
                or tuple(a["shape"]) != shape
                or op["iterator_types"] != ["parallel"] * len(shape)):
            continue
        dims = ", ".join(f"d{i}" for i in range(len(shape)))
        identity = f"affine_map<({dims}) -> ({dims})>"
        bias = f"affine_map<({dims}) -> (d{len(shape) - 1})>"
        maps = op["indexing_maps"]
        if tuple(b["shape"]) == shape and maps == [identity, identity, identity]:
            broadcast = "none"
        elif (tuple(b["shape"]) == (shape[-1],) and
              maps == [identity, bias, identity]):
            broadcast = "last_dimension"
        else:
            continue
        region = prov.get("region_id")
        if not region or region in seen:
            raise ValueError("MX VPU add has no unique source region")
        seen.add(region)
        symbol = _symbol(shape, dtype, broadcast)
        sites.append({"region": region, "shape": shape, "dtype": dtype,
                      "broadcast": broadcast, "row_width": shape[-1],
                      "elements": prod(shape), "symbol": symbol,
                      "kernel_symbol": symbol + "_kernel",
                      "rhs_shape": tuple(b["shape"])})
    return sites


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(prod(shape[axis + 1:]) for axis in range(len(shape)))


def emit_memref_add_shim(site: dict) -> str:
    """Emit the exact ranked C interface for one add source signature."""
    shape, rhs_shape = tuple(site["shape"]), tuple(site["rhs_shape"])
    lhs_shape = tuple(site.get("lhs_shape", shape))
    symbol, kernel = site["symbol"], site["kernel_symbol"]
    scalar = "float" if site["dtype"] == "f32" else "uint16_t"
    lhs_scalar = "float" if site.get("lhs_dtype", site["dtype"]) == "f32" else "uint16_t"
    rhs_scalar = "float" if site.get("rhs_dtype", site["dtype"]) == "f32" else "uint16_t"
    ranks = {len(shape), len(lhs_shape), len(rhs_shape)}
    typedefs = "\n".join(
        f"typedef struct {{ void *allocated, *aligned; intptr_t offset; "
        f"{f'intptr_t sizes[{rank}], strides[{rank}]; ' if rank else ''}"
        f"}} mx_memref_{rank}d;"
        for rank in sorted(ranks))
    groups = (("a", lhs_shape), ("b", rhs_shape), ("c", shape))

    def params(name: str, dims: tuple[int, ...]) -> list[str]:
        return [f"void *{name}_alloc", f"void *{name}_aligned", f"intptr_t {name}_off",
                *(f"intptr_t {name}_s{i}" for i in range(len(dims))),
                *(f"intptr_t {name}_st{i}" for i in range(len(dims)))]

    declarations = ",\n    ".join(item for name, dims in groups for item in params(name, dims))
    checks = " ||\n      ".join(
        [f"{name}_s{i} != {dim}" for name, dims in groups
         for i, dim in enumerate(dims)] +
        [f"{name}_st{i} != {step}" for name, dims in groups
         for i, step in enumerate(_strides(dims))])
    dims = ", ".join(map(str, shape))
    steps = ", ".join(map(str, _strides(shape)))
    call = ",\n      ".join(", ".join(
        [f"{name}->allocated", f"{name}->aligned", f"{name}->offset",
         *(f"{name}->sizes[{i}]" for i in range(len(dims))),
         *(f"{name}->strides[{i}]" for i in range(len(dims)))])
        for name, dims in groups)
    stores = "\n".join(
        f"  destination->{field} = returned.{field};"
        for field in ("allocated", "aligned", "offset",
                      *(f"sizes[{i}]" for i in range(len(shape))),
                      *(f"strides[{i}]" for i in range(len(shape)))))
    return f'''#include <stdint.h>
{typedefs}
extern void {kernel}(const {lhs_scalar} *, const {rhs_scalar} *, {scalar} *);
mx_memref_{len(shape)}d {symbol}(\n    {declarations}) {{
  if (!a_aligned || !b_aligned || !c_aligned ||
      a_off < 0 || b_off < 0 || c_off < 0 ||
      {checks}) __builtin_trap();
  {kernel}((const {lhs_scalar} *)a_aligned + a_off,
           (const {rhs_scalar} *)b_aligned + b_off,
           ({scalar} *)c_aligned + c_off);
  return (mx_memref_{len(shape)}d){{c_alloc, c_aligned, c_off,
                                    {{{dims}}}, {{{steps}}}}};
}}
void _mlir_ciface_{symbol}(mx_memref_{len(shape)}d *result,
                          const mx_memref_{len(lhs_shape)}d *a,
                          const mx_memref_{len(rhs_shape)}d *b,
                          const mx_memref_{len(shape)}d *c) {{
  if (!result || !a || !b || !c) __builtin_trap();
  mx_memref_{len(shape)}d returned = {symbol}(\n      {call});
  volatile mx_memref_{len(shape)}d *destination = (volatile mx_memref_{len(shape)}d *)result;
{stores}
}}
'''


class MXVPUAddPostOffload:
    """Replace exact source addf generics with destination-passing VPU calls."""

    def __init__(self, profile: dict, *, enabled_regions: frozenset[str] | None = None):
        self.profile = profile
        self.enabled_regions = enabled_regions

    def __call__(self, prepared: Path, sidecar: Path, workdir: Path) -> Path:
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text
        from xdsl.dialects import func, tensor
        from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
        from xdsl.ir import Region

        prepared, workdir = Path(prepared), Path(workdir)
        source = prepared.read_text()
        all_sites = selected_add_sites(source, self.profile)
        if self.enabled_regions is not None and not self.enabled_regions <= {
                site["region"] for site in all_sites}:
            raise ValueError("MX VPU add policy names unknown source regions")
        sites = [site for site in all_sites if self.enabled_regions is None or
                 site["region"] in self.enabled_regions]
        if not sites:
            return prepared
        module = parse_mlir_file(prepared)
        forward = [op for op in module.walk()
                   if isinstance(op, func.FuncOp) and op.body.blocks
                   and op.sym_name.data == "forward"]
        if len(forward) != 1:
            raise ValueError("MX VPU add rewrite needs one model forward function")
        block = forward[0].body.block
        candidates = {}
        for op in block.ops:
            if op.name != "linalg.generic":
                continue
            region = getattr(op.attributes.get("prov.region_id"), "data", None)
            if region in {site["region"] for site in sites}:
                candidates.setdefault(region, []).append(op)
        if set(candidates) != {site["region"] for site in sites} or any(
                len(group) != 1 for group in candidates.values()):
            raise ValueError("MX VPU add source regions changed after matrix routing")
        declarations = {}
        records = []
        for site in sites:
            op = candidates[site["region"]][0]
            if (len(op.inputs) != 2 or len(op.results) != 1 or
                    tuple(str(v.type) for v in op.inputs) !=
                    tuple(f"tensor<{ 'x'.join(map(str, dims)) }x{site['dtype']}>"
                          for dims in (site["shape"], site["rhs_shape"]))):
                raise ValueError("MX VPU add source tensor types changed")
            result_type = op.results[0].type
            destination = tensor.EmptyOp((), result_type)
            call = func.CallOp(site["symbol"],
                               [op.inputs[0], op.inputs[1], destination.results[0]],
                               [result_type])
            block.insert_op_before(destination, op)
            block.insert_op_before(call, op)
            op.results[0].replace_all_uses_with(call.results[0])
            op.detach()
            op.erase()
            signature = tuple(value.type for value in call.arguments)
            previous = declarations.setdefault(site["symbol"], signature)
            if previous != signature:
                raise ValueError("MX VPU add symbol has conflicting signatures")
            records.append({"region": site["region"], "symbol": site["symbol"],
                            "dtype": site["dtype"], "broadcast": site["broadcast"],
                            "shape": list(site["shape"])})
        read = DictionaryAttr({"bufferization.access": StringAttr("read")})
        write = DictionaryAttr({"bufferization.access": StringAttr("write")})
        for symbol, signature in declarations.items():
            module.body.block.add_op(func.FuncOp(
                symbol, (signature, (signature[-1],)), Region(),
                visibility="private", arg_attrs=ArrayAttr([read, read, write])))
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / "mx_vpu_add.mlir"
        out.write_text(text(module))
        receipt = {
            "schema": "mx_gemmini.merlin_vpu_add_rewrite.v1",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "sidecar_sha256": hashlib.sha256(Path(sidecar).read_bytes()).hexdigest(),
            "profile_sha256": profile_sha256(self.profile),
            "numeric_policy": "bf16_vpu_add_or_f32_to_bf16_rne_vpu_add_to_f32",
            "regions": records,
        }
        (workdir / "mx_vpu_add_rewrite.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return out


class MXVPUPostOffload:
    """Apply source-bound VPU rewrites and optional fixed-RHS pruning."""

    def __init__(self, profile: dict, *, fixed_weight_catalog: Path | None = None,
                 enabled_classes: frozenset[str] | None = None,
                 enabled_elementwise_operations: frozenset[str] | None = None,
                 enabled_elementwise_regions: frozenset[str] | None = None,
                 enabled_site_regions: dict[str, frozenset[str]] | None = None,
                 connect_multiplies: bool = False,
                 connect_broadcast_multiplies: bool = False,
                 connect_sigmoid_multiplies: bool = False,
                 share_activation_packing: bool = False,
                 activation_packing_budget_bytes: int = 1048576,
                 matrix_vpu_transport: bool = False,
                 vpu_matrix_transport: bool = False,
                 fuse_rms_packing: bool = False,
                 fuse_f32_rms_packing: bool = False,
                 group_projections: bool = False,
                 fuse_softmax_packing: bool = False,
                 fuse_gated_projections: bool = False,
                 batch_rhs_catalog: Path | None = None):
        from .merlin_vpu_elementwise import MXVPUElementwisePostOffload
        from .merlin_vpu_reduce import MXVPUReducePostOffload
        from .merlin_vpu_softmax import MXVPUSoftmaxPostOffload
        from .merlin_vpu_unary import MXVPUUnaryPostOffload

        self.enabled_site_regions = enabled_site_regions or {}
        self.softmax = MXVPUSoftmaxPostOffload(
            profile, enabled_regions=self.enabled_site_regions.get("softmax"))
        self.add = MXVPUAddPostOffload(
            profile, enabled_regions=self.enabled_site_regions.get("add"))
        self.elementwise = MXVPUElementwisePostOffload(
            profile, enabled_operations=enabled_elementwise_operations,
            connect_multiplies=connect_multiplies,
            connect_broadcast_multiplies=connect_broadcast_multiplies,
            enabled_regions=(self.enabled_site_regions.get("elementwise") or None
                             if "elementwise" in self.enabled_site_regions else
                             enabled_elementwise_regions))
        self.unary = MXVPUUnaryPostOffload(
            profile, enabled_regions=self.enabled_site_regions.get("unary"))
        self.reduce = MXVPUReducePostOffload(
            profile, enabled_regions=self.enabled_site_regions.get("reduce"))
        valid = frozenset(("softmax", "add", "elementwise", "unary", "reduce"))
        if not self.enabled_site_regions.keys() <= valid:
            raise ValueError("unknown MX VPU per-site policy class")
        self.enabled_classes = valid if enabled_classes is None else frozenset(enabled_classes)
        if not self.enabled_classes <= valid:
            raise ValueError(f"unknown MX VPU classes: {sorted(self.enabled_classes - valid)}")
        self.connected_sigmoid = None
        if connect_sigmoid_multiplies:
            if (fixed_weight_catalog is None or not connect_multiplies
                    or not {"elementwise", "unary"} <= self.enabled_classes
                    or not self.enabled_site_regions):
                raise ValueError("sigmoid connections need a catalog, site policy and both VPU classes")
            from .merlin_connected_sigmoid import MXConnectedSigmoidPostOffload

            self.connected_sigmoid = MXConnectedSigmoidPostOffload(fixed_weight_catalog, profile)
        if fixed_weight_catalog is not None:
            from .merlin_fixed_weight import MXFixedWeightPostOffload

            self.fixed_weight = MXFixedWeightPostOffload(fixed_weight_catalog)
        else:
            self.fixed_weight = None
        self.shared_activation = None
        if share_activation_packing:
            if fixed_weight_catalog is None:
                raise ValueError("shared activation packing requires a fixed weight catalog")
            from .merlin_shared_activation import MXSharedActivationPostOffload

            self.shared_activation = MXSharedActivationPostOffload(
                fixed_weight_catalog, max_live_bytes=activation_packing_budget_bytes)
        self.matrix_transport = None
        if matrix_vpu_transport:
            if fixed_weight_catalog is None:
                raise ValueError("matrix transport requires a fixed weight catalog")
            from .merlin_matrix_transport import MXMatrixTransportPostOffload

            self.matrix_transport = MXMatrixTransportPostOffload(fixed_weight_catalog)
        self.vpu_matrix_transport = None
        if vpu_matrix_transport:
            if fixed_weight_catalog is None or not share_activation_packing or not connect_multiplies:
                raise ValueError("VPU matrix transport needs a catalog, shared packing and connected multiplies")
            from .merlin_vpu_matrix_transport import MXVPUMatrixTransportPostOffload

            self.vpu_matrix_transport = MXVPUMatrixTransportPostOffload(fixed_weight_catalog)
        self.f32_rms_packing = None
        if fuse_f32_rms_packing:
            if fixed_weight_catalog is None or not share_activation_packing or fuse_rms_packing:
                raise ValueError("FP32 RMS packing requires a shared immutable matrix catalog")
            from .merlin_f32_rms_packing import MXF32RMSPackingPostOffload

            self.f32_rms_packing = MXF32RMSPackingPostOffload(fixed_weight_catalog)
        self.gated_projection_packing = None
        if fuse_gated_projections:
            if fixed_weight_catalog is None or not share_activation_packing:
                raise ValueError(
                    "gated projection packing requires a shared immutable matrix catalog"
                )
            from .merlin_gated_projection import MXGatedProjectionPostOffload

            self.gated_projection_packing = MXGatedProjectionPostOffload(
                fixed_weight_catalog
            )
        self.softmax_packing = None
        if fuse_softmax_packing:
            if fixed_weight_catalog is None or batch_rhs_catalog is None:
                raise ValueError("softmax packing requires a source-bound compact batch catalog")
            from .merlin_softmax_packing import MXSoftmaxPackingPostOffload

            self.softmax_packing = MXSoftmaxPackingPostOffload(fixed_weight_catalog)
        self.projection_groups = None
        if group_projections:
            if fixed_weight_catalog is None or not share_activation_packing:
                raise ValueError("projection grouping requires a shared packed-input catalog")
            from .merlin_projection_group import MXProjectionGroupPostOffload

            self.projection_groups = MXProjectionGroupPostOffload(fixed_weight_catalog)
        self.batch_rhs_reuse = None
        self.rms_packing = None
        if fuse_rms_packing:
            if fixed_weight_catalog is None or not vpu_matrix_transport:
                raise ValueError("RMS packing requires a catalog and private VPU matrix transport")
            from .merlin_rms_packing import MXRmsPackingPostOffload

            self.rms_packing = MXRmsPackingPostOffload(fixed_weight_catalog)
        if batch_rhs_catalog is not None:
            from .merlin_batch_rhs import MXBatchRHSReusePostOffload

            self.batch_rhs_reuse = MXBatchRHSReusePostOffload(batch_rhs_catalog)

    def __call__(self, prepared: Path, sidecar: Path, workdir: Path) -> Path:
        selected = prepared
        if self.enabled_site_regions:
            from .execution_policy import reconcile_vpu_regions

            resolved, _ = reconcile_vpu_regions(
                Path(prepared).read_text(), self.softmax.profile,
                self.enabled_site_regions)
            self.enabled_site_regions = resolved
            for name in ("softmax", "add", "elementwise", "unary", "reduce"):
                getattr(self, name).enabled_regions = resolved[name]
        for name in ("softmax", "add", "elementwise", "unary", "reduce"):
            if name == "elementwise" and self.connected_sigmoid is not None:
                selected, chains = self.connected_sigmoid(selected, sidecar, workdir)
                fused_elements = {region for chain in chains for region in chain["regions"]}
                fused_unary = {chain["unary_region"] for chain in chains}
                self.elementwise.enabled_regions = self.enabled_site_regions["elementwise"] - fused_elements
                self.unary.enabled_regions = self.enabled_site_regions["unary"] - fused_unary
            if (name in self.enabled_classes and self.enabled_site_regions.get(name) != frozenset()
                    and getattr(self, name).enabled_regions != frozenset()):
                selected = getattr(self, name)(selected, sidecar, workdir)
        if self.fixed_weight is not None:
            selected = self.fixed_weight(selected, sidecar, workdir)
        if self.shared_activation is not None:
            selected = self.shared_activation(selected, sidecar, workdir)
        if self.f32_rms_packing is not None:
            selected = self.f32_rms_packing(selected, sidecar, workdir)
        # Consume the original down-projection before private BF16 transport.
        if self.gated_projection_packing is not None:
            selected = self.gated_projection_packing(selected, sidecar, workdir)
        if self.matrix_transport is not None:
            selected = self.matrix_transport(selected, sidecar, workdir)
        if self.vpu_matrix_transport is not None:
            selected = self.vpu_matrix_transport(selected, sidecar, workdir)
        if self.rms_packing is not None:
            selected = self.rms_packing(selected, sidecar, workdir)
        if self.batch_rhs_reuse is not None:
            selected = self.batch_rhs_reuse(selected, sidecar, workdir)
        if self.softmax_packing is not None:
            selected = self.softmax_packing(selected, sidecar, workdir)
        if self.projection_groups is not None:
            selected = self.projection_groups(selected, sidecar, workdir)
        return selected
