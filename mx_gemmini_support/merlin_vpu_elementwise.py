"""Source-checked MX VPU binary arithmetic for captured graphs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from itertools import combinations
from math import prod
from pathlib import Path

from .target_profile import profile_sha256

_SOURCE_OPS = {
    "mul": ("aten.mul.Tensor", "arith.mulf"),
    "sub": ("aten.sub.Tensor", "arith.subf"),
    "add": ("aten.add.Tensor", "arith.addf"),
    "div": ("aten.div.Tensor", "arith.divf"),
}
_LAYER_NORM_OPS = {
    "arith.addf": "add", "arith.subf": "sub", "arith.mulf": "mul",
    "arith.divf": "div",
}


def _map(result_shape: tuple[int, ...], operand_shape: tuple[int, ...],
         axes: tuple[int, ...]) -> str:
    rank = len(result_shape)
    loop_axes = ", ".join(f"d{i}" for i in range(rank))
    mapped = ", ".join(
        ("0" if dim == 1 and result_shape[axes[i]] != 1
         else f"d{axes[i]}")
        for i, dim in enumerate(operand_shape))
    return f"affine_map<({loop_axes}) -> ({mapped})>"


def _matching_axes(result_shape: tuple[int, ...], operand_shape: tuple[int, ...],
                   source_map: str) -> tuple[int, ...] | None:
    for axes in combinations(range(len(result_shape)), len(operand_shape)):
        if (all(dim in (1, result_shape[axis])
                for axis, dim in zip(axes, operand_shape)) and
                _map(result_shape, operand_shape, axes) == source_map):
            return axes
    return None


def selected_elementwise_sites(mlir_text: str, profile: dict) -> list[dict]:
    """Require one pure scalar op and the exact source broadcast maps."""
    from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir

    if (not profile["resources"].get("vpu") or
            profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_columns") != 16):
        return []
    sites = []
    seen = set()
    generic_ordinals = defaultdict(int)
    for op in parse_linalg_mlir(mlir_text)["ops"]:
        prov = op["prov"]
        source_region = prov.get("region_id")
        primitive = (op["body_ops"][-1] if op["kind"] == "linalg.generic" and
                     (len(op["body_ops"]) == 1 or
                      (len(op["body_ops"]) == 2 and
                       op["body_ops"][0] == "arith.extf")) and
                     op["body_ops"][-1] in _LAYER_NORM_OPS else None)
        if source_region and primitive:
            ordinal = generic_ordinals[source_region, primitive]
            generic_ordinals[source_region, primitive] += 1
        else:
            ordinal = None
        source_op = prov.get("op")
        mixed_float = (source_op in {"mul", "add"} and
                       op["body_ops"] == ["arith.extf", _SOURCE_OPS[source_op][1]])
        if (source_op in _SOURCE_OPS and
                prov.get("aten") == _SOURCE_OPS[source_op][0] and
                (op["body_ops"] == [_SOURCE_OPS[source_op][1]] or mixed_float)):
            operation = source_op
        elif (source_op == "layer_norm" and
              prov.get("aten") == "aten.native_layer_norm.default" and
              len(op["body_ops"]) == 1 and
              op["body_ops"][0] in _LAYER_NORM_OPS):
            operation = _LAYER_NORM_OPS[op["body_ops"][0]]
        elif (source_op == "convolution_im2col_matmul" and
              prov.get("aten") == "aten.convolution.default" and
              op["body_ops"] == ["arith.addf"]):
            operation = "add"
        elif (source_op == "reduce_mean" and
              prov.get("aten") == "aten.mean.dim" and
              op["body_ops"] == ["arith.divf"]):
            operation = "div"
        else:
            continue
        if (op["kind"] != "linalg.generic" or len(op["ins"]) != 2 or
                len(op["results"]) != 1):
            continue
        body = op["scalar_body"] or {}
        operations = body.get("operations") or ()
        pure = (len(operations) == 1 and
                operations[0].get("operands") == ["arg:0", "arg:1"] and
                body.get("yields") == ["op:0:0"])
        mixed = (mixed_float and len(operations) == 2 and
                 operations[0].get("operands") == ["arg:0"] and
                 operations[1].get("operands") == ["op:0:0", "arg:1"] and
                 body.get("yields") == ["op:1:0"])
        if (not (pure or mixed) or body.get("captures") or
                body.get("effects") or operations[0].get("effects") or
                operations[0].get("regions") or
                any(step.get("effects") or step.get("regions") for step in operations)):
            continue
        result = op["results"][0]
        shape = tuple(result["shape"])
        dtype = result["dtype"]
        inputs = op["ins"]
        input_shapes = [tuple(item["shape"]) for item in inputs]
        input_axes = ([_matching_axes(shape, dims, source_map)
                       for dims, source_map in zip(input_shapes, op["indexing_maps"][:2])]
                      if len(shape) <= 4 else [])
        input_dtypes = [item["dtype"] for item in inputs]
        if (not 1 <= len(shape) <= 4 or any(type(dim) is not int or dim <= 0 for dim in shape)
                or dtype not in {"f32", "bf16"} or
                (input_dtypes != (["bf16", "f32"] if mixed else [dtype, dtype])) or
                op["iterator_types"] != ["parallel"] * len(shape) or
                len(op["indexing_maps"]) != 3 or
                any(axes is None for axes in input_axes) or
                op["indexing_maps"][2] !=
                _map(shape, shape, tuple(range(len(shape))))):
            continue
        if dtype == "f32" and prod(shape) == 1:
            # The VPU rounds f32 inputs and results through BF16. Scalar
            # state updates, including SmolVLA's timestep, need the source
            # f32 value and gain nothing from a one-element VPU launch.
            continue
        if source_op == "add" and len(shape) != 1:
            # The add catalog owns exact tensors and last-axis vector biases.
            # Other source-verified broadcasts, including attention masks,
            # use this general elementwise issuer.
            add_catalog_shape = (input_shapes[0] == shape and
                                 (input_shapes[1] == shape or
                                  input_shapes[1] == (shape[-1],)))
            if add_catalog_shape and not mixed_float:
                continue
        if source_op == "convolution_im2col_matmul" and operation != "add":
            continue
        full = [i for i, dims in enumerate(input_shapes) if dims == shape]
        swap = bool(full) and operation in {"mul", "add"} and 0 not in full
        lhs_index = 1 if swap else 0
        rhs_index = 0 if swap else 1
        lhs_shape = input_shapes[lhs_index]
        lhs_axes = input_axes[lhs_index]
        lhs_dtype = input_dtypes[lhs_index]
        rhs_shape = input_shapes[rhs_index]
        rhs_axes = input_axes[rhs_index]
        rhs_dtype = input_dtypes[rhs_index]
        if len(lhs_shape) > len(shape) or len(rhs_shape) > len(shape):
            continue
        region = f"{source_region}#{operation}{ordinal}"
        if not source_region or region in seen:
            raise ValueError("MX VPU elementwise has no unique source region")
        seen.add(region)
        name = (f"mx_vpu_{operation}_{dtype}_l{lhs_dtype}_r{rhs_dtype}_" +
                "_".join(map(str, shape)) +
                "_lhs_" + ("_".join(map(str, lhs_shape)) if lhs_shape else "scalar") +
                "_laxes_" + ("_".join(map(str, lhs_axes)) if lhs_axes else "none") +
                "_rhs_" + ("_".join(map(str, rhs_shape)) if rhs_shape else "scalar") +
                "_axes_" + "_".join(map(str, rhs_axes)))
        sites.append({
            "region": region, "source_region": source_region,
            "generic_ordinal": ordinal, "operation": operation, "shape": shape,
            "lhs_shape": lhs_shape, "lhs_axes": lhs_axes,
            "lhs_dtype": lhs_dtype, "rhs_dtype": rhs_dtype,
            "rhs_shape": rhs_shape, "dtype": dtype, "swap": swap,
            "rhs_axes": rhs_axes,
            "elements": prod(shape), "symbol": name,
            "kernel_symbol": name + "_kernel",
        })
    return sites


class MXVPUElementwisePostOffload:
    """Replace verified binary generics with destination-passing VPU calls."""

    def __init__(self, profile: dict, *, enabled_operations: frozenset[str] | None = None,
                 enabled_regions: frozenset[str] | None = None,
                 connect_multiplies: bool = False):
        self.profile = profile
        self.connect_multiplies = connect_multiplies
        valid = frozenset(("mul", "add", "sub", "div"))
        self.enabled_operations = (valid if enabled_operations is None
                                   else frozenset(enabled_operations))
        if not self.enabled_operations or not self.enabled_operations <= valid:
            raise ValueError("MX VPU elementwise operations must be a nonempty subset "
                             "of mul,add,sub,div")
        self.enabled_regions = (None if enabled_regions is None
                                else frozenset(enabled_regions))
        if self.enabled_regions is not None and not self.enabled_regions:
            raise ValueError("MX VPU elementwise regions must be nonempty")

    def __call__(self, prepared: Path, sidecar: Path, workdir: Path) -> Path:
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text
        from xdsl.dialects import func, tensor
        from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
        from xdsl.ir import Region

        prepared, workdir = Path(prepared), Path(workdir)
        source = prepared.read_text()
        all_sites = selected_elementwise_sites(source, self.profile)
        if (self.enabled_regions is not None and
                not self.enabled_regions <= {site["region"] for site in all_sites}):
            raise ValueError("MX VPU elementwise diagnostic names unknown source regions")
        sites = [site for site in all_sites
                 if site["operation"] in self.enabled_operations and
                 (self.enabled_regions is None or
                  site["region"] in self.enabled_regions)]
        if (self.enabled_regions is not None and
                {site["region"] for site in sites} != self.enabled_regions):
            raise ValueError("MX VPU elementwise regions conflict with the operation filter")
        if not sites:
            return prepared
        module = parse_mlir_file(prepared)
        forward = [op for op in module.walk()
                   if isinstance(op, func.FuncOp) and op.body.blocks and
                   op.sym_name.data == "forward"]
        if len(forward) != 1:
            raise ValueError("MX VPU elementwise rewrite needs one forward function")
        block = forward[0].body.block
        site_by_region = {site["region"]: site for site in sites}
        candidates = {}
        generic_ordinals = defaultdict(int)
        for op in block.ops:
            if op.name != "linalg.generic":
                continue
            source_region = getattr(op.attributes.get("prov.region_id"), "data", None)
            if not source_region:
                continue
            body_ops = [step.name for step in op.body.blocks[0].ops
                        if step.name != "linalg.yield"]
            if (not body_ops or
                    (len(body_ops) != 1 and
                     not (len(body_ops) == 2 and body_ops[0] == "arith.extf")) or
                    body_ops[-1] not in _LAYER_NORM_OPS):
                continue
            primitive = body_ops[-1]
            operation = _LAYER_NORM_OPS[primitive]
            region = f"{source_region}#{operation}{generic_ordinals[source_region, primitive]}"
            generic_ordinals[source_region, primitive] += 1
            if region in site_by_region:
                candidates.setdefault(region, []).append(op)
        if set(candidates) != set(site_by_region) or any(
                len(group) != 1 for group in candidates.values()):
            raise ValueError("MX VPU elementwise source regions changed after routing")
        declarations = {}
        records = []
        connected = []
        consumed = {}
        if self.connect_multiplies:
            from .merlin_connected_multiply import multiply_pairs

            for pair in multiply_pairs(module, sites):
                first, second = pair["first"], pair["second"]
                result_type = second.results[0].type
                destination = tensor.EmptyOp((), result_type)
                call = func.CallOp(pair["symbol"],
                                   [*first.inputs, second.inputs[pair["external_index"]], destination.results[0]],
                                   [result_type])
                block.insert_op_before(destination, second)
                block.insert_op_before(call, second)
                second.results[0].replace_all_uses_with(call.results[0])
                second.detach()
                second.erase()
                first.detach()
                first.erase()
                declarations[pair["symbol"]] = tuple(v.type for v in call.arguments)
                consumed.update({region: pair["symbol"] for region in pair["regions"]})
                connected.append({key: pair[key] for key in ("regions", "shape", "symbol")})
        for site in sites:
            if site["region"] in consumed:
                records.append({key: site[key] for key in
                                ("region", "operation", "dtype", "swap")} |
                               {"symbol": consumed[site["region"]], "source_symbol": site["symbol"]})
                continue
            op = candidates[site["region"]][0]
            args = list(op.inputs)
            if site["swap"]:
                args.reverse()
            shape_type = f"tensor<{'x'.join(map(str, site['shape']))}x{site['dtype']}>"
            lhs_type = (f"tensor<{'x'.join(map(str, site['lhs_shape']))}x{site['lhs_dtype']}>"
                        if site["lhs_shape"] else f"tensor<{site['lhs_dtype']}>")
            rhs_type = (f"tensor<{'x'.join(map(str, site['rhs_shape']))}x{site['rhs_dtype']}>"
                        if site["rhs_shape"] else f"tensor<{site['rhs_dtype']}>")
            if (len(args) != 2 or len(op.results) != 1 or
                    tuple(map(str, (args[0].type, args[1].type,
                                    op.results[0].type))) !=
                    (lhs_type, rhs_type, shape_type)):
                raise ValueError("MX VPU elementwise source tensor types changed")
            result_type = op.results[0].type
            destination = tensor.EmptyOp((), result_type)
            call = func.CallOp(site["symbol"], [*args, destination.results[0]],
                               [result_type])
            block.insert_op_before(destination, op)
            block.insert_op_before(call, op)
            op.results[0].replace_all_uses_with(call.results[0])
            op.detach()
            op.erase()
            signature = tuple(value.type for value in call.arguments)
            previous = declarations.setdefault(site["symbol"], signature)
            if previous != signature:
                raise ValueError("MX VPU elementwise symbol has conflicting signatures")
            records.append({key: site[key] for key in
                            ("region", "symbol", "operation", "dtype", "swap")})
        read = DictionaryAttr({"bufferization.access": StringAttr("read")})
        write = DictionaryAttr({"bufferization.access": StringAttr("write")})
        for symbol, signature in declarations.items():
            module.body.block.add_op(func.FuncOp(
                symbol, (signature, (signature[-1],)), Region(),
                visibility="private", arg_attrs=ArrayAttr([read] * (len(signature) - 1) + [write])))
        workdir.mkdir(parents=True, exist_ok=True)
        out = workdir / "mx_vpu_elementwise.mlir"
        out.write_text(text(module))
        receipt = {
            "schema": "mx_gemmini.merlin_vpu_elementwise_rewrite.v1",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "sidecar_sha256": hashlib.sha256(Path(sidecar).read_bytes()).hexdigest(),
            "profile_sha256": profile_sha256(self.profile),
            "numeric_policy": "f32_to_bf16_rne_vpu_to_f32_or_direct_bf16",
            "enabled_operations": sorted(self.enabled_operations),
            "enabled_regions": (sorted(self.enabled_regions)
                                if self.enabled_regions is not None else None),
            "regions": records,
            "connected_multiplies": connected,
        }
        (workdir / "mx_vpu_elementwise_rewrite.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return out
