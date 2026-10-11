"""MX-owned BF16 tensor boundary for source-bound dense catalog calls.

The prepared source still performs every surrounding operation, including
the BF16 addmm bias. Only zero-initialized BF16 matmuls selected by the MX
profile acquire f32 pointer operands for the existing dense conversion ABI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, StringAttr, TensorType, bf16, f32
from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
from xdsl.dialects.linalg.ops import FillOp, GenericOp, MatmulOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.dialects.builtin import AffineMapAttr

from .merlin_inventory import build_mx_worklist_from_merlin
from .model_precision_plan import build_model_precision_plan
from .target_profile import profile_sha256


def _wide_tensor(value):
    old_type = value.type
    if not isinstance(old_type, TensorType) or old_type.element_type != bf16:
        raise ValueError("MX BF16 preparation needs a ranked BF16 tensor")
    wide_type = TensorType(f32, old_type.get_shape())
    identity = AffineMapAttr(AffineMap.identity(len(old_type.get_shape())))
    parallel = IteratorTypeAttr(IteratorType.PARALLEL)
    empty = tensor.EmptyOp((), wide_type)
    body = Block(arg_types=[bf16, f32])
    cast = arith.ExtFOp(body.args[0], f32)
    body.add_ops([cast, YieldOp(cast.result)])
    convert = GenericOp(
        inputs=[value], outputs=[empty.results[0]], body=Region(body),
        indexing_maps=ArrayAttr([identity, identity]),
        iterator_types=ArrayAttr([parallel] * len(old_type.get_shape())),
        result_types=[wide_type],
    )
    return empty, convert


def _narrow_tensor(value, narrow_type):
    identity = AffineMapAttr(AffineMap.identity(len(narrow_type.get_shape())))
    parallel = IteratorTypeAttr(IteratorType.PARALLEL)
    empty = tensor.EmptyOp((), narrow_type)
    body = Block(arg_types=[f32, bf16])
    cast = arith.TruncFOp(body.args[0], bf16)
    body.add_ops([cast, YieldOp(cast.result)])
    convert = GenericOp(
        inputs=[value], outputs=[empty.results[0]], body=Region(body),
        indexing_maps=ArrayAttr([identity, identity]),
        iterator_types=ArrayAttr([parallel] * len(narrow_type.get_shape())),
        result_types=[narrow_type],
    )
    return empty, convert


class MXBF16CatalogPreparation:
    """Prepare exact profile-selected BF16 matmuls for the f32 MX catalog ABI."""

    def __init__(self, profile: dict):
        self.profile = profile

    def __call__(self, source_path: Path, workdir: Path) -> Path:
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.system.offload import catalog_source_contractions
        from merlin.xdsl_dialects._common import text

        source_path, workdir = Path(source_path), Path(workdir)
        source = source_path.read_bytes()
        module = parse_mlir_file(source_path)
        worklist = build_mx_worklist_from_merlin(source.decode(), self.profile)
        # Merlin's normal preparation widens BF16 matmuls to a BF16-input,
        # f32-accumulating generic before this target hook. Restore those
        # exact matrix sites to the rank-two planning inventory. The f32
        # accumulator and its downstream BF16 narrowing remain live.
        widened = {}
        for op, shape in catalog_source_contractions(
                module, "mx_gemmini", ("bf16", "bf16", "f32")):
            if not isinstance(op, GenericOp):
                continue
            marker = getattr(op.attributes.get("prov.transforms"), "data", None)
            if marker != "bf16_f32acc":
                continue
            region = getattr(op.attributes.get("prov.region_id"), "data", None)
            source_op = getattr(op.attributes.get("prov.op"), "data", None)
            fqn = getattr(op.attributes.get("prov.fqn"), "data", None)
            body = list(op.body.blocks[0].ops)
            if (not region or not fqn or source_op not in {"matmul", "addmm"}
                    or len(shape.parallel) != 2 or len(shape.reduction) != 1
                    or [step.name for step in body] != [
                        "arith.extf", "arith.extf", "arith.mulf",
                        "arith.addf", "linalg.yield"]
                    or body[2].operands[0] != body[0].results[0]
                    or body[2].operands[1] != body[1].results[0]
                    or body[3].operands[1] != body[2].results[0]
                    or body[4].operands[0] != body[3].results[0]):
                raise ValueError("MX BF16 prepared source lost Merlin's exact f32 accumulator form")
            if region in widened:
                raise ValueError(f"MX BF16 prepared source duplicated region {region}")
            dimensions = list(shape.parallel) + list(shape.reduction)
            widened[region] = (op, shape)
            worklist["rank2_matmuls"].append({
                "fqn": fqn, "region_id": region, "source_op": source_op,
                "shape_mnk": dimensions, "element_type": "bf16",
                "k_requires_32_element_scale_padding": dimensions[2] % 32 != 0,
                "status": "requires_explicit_mx_quantization_and_program_lowering",
            })
        if widened:
            worklist["other_contraction_regions"] = [
                site for site in worklist["other_contraction_regions"]
                if site["region_id"] not in widened]
        proposal = build_model_precision_plan(worklist, self.profile)
        selected = {site["region_id"]: site for site in proposal["sites"]
                    if site.get("element_type") == "bf16"
                    and site.get("selected_format") in {"mxfp4", "mxfp8"}}
        if len(selected) != sum(site.get("element_type") == "bf16"
                                and site.get("selected_format") in {"mxfp4", "mxfp8"}
                                for site in proposal["sites"]):
            raise ValueError("MX BF16 precision proposal has duplicate source regions")
        if not selected:
            return source_path
        candidates = catalog_source_contractions(module, "mx_gemmini", ("bf16", "bf16", "bf16"))
        candidates += list(widened.values())
        by_region = {}
        for op, shape in candidates:
            region = getattr(op.attributes.get("prov.region_id"), "data", None)
            if region in selected:
                by_region.setdefault(region, []).append((op, shape))
        if set(selected) != set(by_region) or any(len(group) != 1 for group in by_region.values()):
            missing = sorted(set(selected) - set(by_region))
            ambiguous = sorted(region for region, group in by_region.items() if len(group) != 1)
            raise ValueError(f"MX BF16 selected matmuls are not exact zero-init sources: "
                             f"missing={missing}, ambiguous={ambiguous}")

        records = []
        for region, site in selected.items():
            op, shape = by_region[region][0]
            if (not isinstance(op, (MatmulOp, GenericOp)) or site["source_op"] not in {"matmul", "addmm"}
                    or tuple(shape.parallel) + tuple(shape.reduction) != tuple(site["shape_mnk"])
                    or len(op.results) != 1 or not isinstance(op.results[0].type, TensorType)):
                raise ValueError(f"MX BF16 selected source {region} changed kind or geometry")
            old_type = op.results[0].type
            lhs_empty, lhs_wide = _wide_tensor(op.inputs[0])
            rhs_empty, rhs_wide = _wide_tensor(op.inputs[1])
            wide_type = TensorType(f32, old_type.get_shape())
            new_ops = [lhs_empty, lhs_wide, rhs_empty, rhs_wide]
            if isinstance(op, MatmulOp):
                init_empty = tensor.EmptyOp((), wide_type)
                zero = arith.ConstantOp(FloatAttr(0.0, f32))
                init = FillOp(inputs=[zero.results[0]], outputs=[init_empty.results[0]], res=[wide_type])
                new_ops.extend([init_empty, zero, init])
                wide_init = init.results[0]
            else:
                wide_init = op.outputs[0]
            wide = MatmulOp(
                inputs=[lhs_wide.results[0], rhs_wide.results[0]], outputs=[wide_init], res=[wide_type],
                attributes=dict(op.attributes) | {"mx.source_bf16_widened": StringAttr("f32_catalog_abi")},
            )
            new_ops.append(wide)
            if isinstance(op, MatmulOp):
                narrow_empty, narrow = _narrow_tensor(wide.results[0], old_type)
                new_ops.extend([narrow_empty, narrow])
                replacement = narrow.results[0]
            else:
                replacement = wide.results[0]
            block = op.parent_block()
            for new_op in new_ops:
                block.insert_op_before(new_op, op)
            op.results[0].replace_all_uses_with(replacement)
            block.detach_op(op)
            records.append({"region": region, "shape_mnk": site["shape_mnk"],
                            "selected_format": site["selected_format"],
                            "source_element_type": "bf16", "catalog_element_type": "f32",
                            "prepared_source_form": "f32acc_generic" if isinstance(op, GenericOp)
                                                    else "bf16_matmul"})

        module.verify()
        workdir.mkdir(parents=True, exist_ok=True)
        prepared = workdir / "mx_bf16_catalog.mlir"
        prepared.write_text(text(module), encoding="utf-8")
        (workdir / "mx_bf16_catalog.json").write_text(json.dumps({
            "schema": "mx_gemmini.bf16_catalog_preparation.v1",
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "profile_sha256": profile_sha256(self.profile),
            "selected_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
            "widened_operations": len(records), "records": records,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return prepared
