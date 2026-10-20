"""Bind explicit Merlin FP32 host softmax arithmetic to captured source regions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .merlin_vpu_softmax import _HELPERS, _PATTERN, selected_softmax_sites
from .target_profile import profile_sha256


class MXHostSoftmaxPostOffload:
    """Replace closed host softmax regions under an explicit finite-input policy."""

    def __init__(self, profile: dict, *, policy: str, enabled_regions: frozenset[str]):
        from merlin.llvmlower.cpu_softmax import (
            CPU_SOFTMAX_F32_AVX2_POLICY,
            LIBM_SOFTMAX_F32_FINITE_POLICY,
            LIBM_SOFTMAX_F32_RNE_POLICY,
        )

        if policy not in {
            CPU_SOFTMAX_F32_AVX2_POLICY,
            LIBM_SOFTMAX_F32_FINITE_POLICY,
            LIBM_SOFTMAX_F32_RNE_POLICY,
        }:
            raise ValueError("unsupported host softmax arithmetic policy")
        self.profile = profile
        self.policy = policy
        self.enabled_regions = enabled_regions

    def __call__(self, prepared: Path, sidecar: Path, workdir: Path) -> Path:
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.llvmlower.cpu_softmax import (
            LIBM_SOFTMAX_F32_FINITE_POLICY,
            LIBM_SOFTMAX_F32_RNE_POLICY,
            build_cpu_softmax_lastdim,
            build_cpu_softmax_maximum,
            build_libm_softmax_exp_rne,
        )
        from merlin.xdsl_dialects._common import text
        from xdsl.dialects import func
        from xdsl.ir.affine import AffineExpr, AffineMap

        source = Path(prepared).read_text()
        selected = {
            row["region"]: row
            for row in selected_softmax_sites(source, self.profile)
            if row["region"] in self.enabled_regions
        }
        if set(selected) != set(self.enabled_regions):
            raise ValueError("host softmax policy names missing source regions")
        if not selected:
            return prepared
        module = parse_mlir_file(prepared)
        forward = [
            op
            for op in module.walk()
            if isinstance(op, func.FuncOp)
            and op.sym_name.data == "forward"
            and op.body.blocks
        ]
        if len(forward) != 1:
            raise ValueError("host softmax needs one forward function")
        block = forward[0].body.block
        ops = list(block.ops)
        records = []
        identity = AffineMap.identity(4)
        broadcast = AffineMap(
            4,
            0,
            tuple(AffineExpr.dimension(i) for i in range(3))
            + (AffineExpr.constant(0),),
        )
        for region, row in selected.items():
            indices = [
                i
                for i, op in enumerate(ops)
                if getattr(op.attributes.get("prov.region_id"), "data", None) == region
            ]
            if not indices:
                raise ValueError("host softmax source region disappeared")
            members = ops[indices[0] : indices[-1] + 1]
            if tuple(op.name for op in members) != _PATTERN:
                raise ValueError("host softmax source decomposition changed")
            for op in members:
                marker = getattr(op.attributes.get("prov.region_id"), "data", None)
                if marker != region and (marker is not None or op.name not in _HELPERS):
                    raise ValueError("host softmax contains foreign operations")
            # Check scalar bodies, operand order, initial values and indexing;
            # provenance alone does not establish softmax semantics.
            for index, scalar_name, count in [
                (2, "arith.maximumf", 2),
                (6, "arith.subf", 3),
                (8, "math.exp", 2),
                (11, "arith.addf", 2),
                (15, "arith.divf", 3),
            ]:
                op = members[index]
                body = op.regions[0].block
                steps = list(body.ops)
                if (
                    len(body.args) != count
                    or len(steps) != 2
                    or steps[0].name != scalar_name
                    or steps[1].name != "linalg.yield"
                    or tuple(steps[0].operands)
                    != tuple(body.args[: 1 if index == 8 else 2])
                    or tuple(steps[1].operands) != tuple(steps[0].results)
                    or steps[0].attributes
                    or any(
                        getattr(value, "data", ())
                        for value in steps[0].properties.values()
                    )
                ):
                    raise ValueError("host softmax scalar arithmetic changed")
                if index in (2, 11):
                    if tuple(op.dimensions.get_values()) != (3,):
                        raise ValueError("host softmax reduction axis changed")
                else:
                    expected = (
                        (identity, identity)
                        if index == 8
                        else (identity, broadcast, identity)
                    )
                    if (
                        tuple(a.data for a in op.indexing_maps) != expected
                        or any(a.data.value != "parallel" for a in op.iterator_types)
                        or tuple(op.outputs) != (members[index - 1].results[0],)
                        or len(op.inputs) != (1 if index == 8 else 2)
                    ):
                        raise ValueError("host softmax indexing changed")
            for index, value in [(0, float("-inf")), (9, 0.0)]:
                attr = members[index].properties["value"]
                if attr.value.data != value:
                    raise ValueError("host softmax reduction initialization changed")
            maximum, shifted, exps, total, result = [
                members[i] for i in (2, 6, 8, 11, 15)
            ]
            score = shifted.inputs[0]
            if (
                maximum.input != score
                or tuple(exps.inputs) != (shifted.results[0],)
                or total.input != exps.results[0]
                or result.inputs[0] != exps.results[0]
                or members[1].operands[0] != members[0].results[0]
                or members[10].operands[0] != members[9].results[0]
                or maximum.init != members[1].results[0]
                or total.init != members[10].results[0]
            ):
                raise ValueError("host softmax dataflow changed")
            for reduction, collapse, expand, operand in [
                (maximum, members[3], members[4], shifted.inputs[1]),
                (total, members[12], members[13], result.inputs[1]),
            ]:
                if (
                    collapse.operands[0] != reduction.results[0]
                    or expand.operands[0] != collapse.results[0]
                    or operand != expand.results[0]
                    or tuple(expand.results[0].type.get_shape())
                    != (*row["shape"][:-1], 1)
                ):
                    raise ValueError("host softmax broadcast dataflow changed")
            inside = set(members)
            if score.owner in inside or any(
                use.operation not in inside
                for op in members[:-1]
                for value in op.results
                for use in value.uses
            ):
                raise ValueError("host softmax has an escaping intermediate")
            if self.policy in {
                LIBM_SOFTMAX_F32_FINITE_POLICY,
                LIBM_SOFTMAX_F32_RNE_POLICY,
            }:
                body = maximum.regions[0].block
                original_maximum = body.first_op
                finite_maximum = build_cpu_softmax_maximum(
                    *original_maximum.operands,
                    maximum_policy="finite",
                    assume_finite_inputs=True,
                )
                body.insert_op_before(finite_maximum, original_maximum)
                original_maximum.result.replace_all_uses_with(finite_maximum.result)
                original_maximum.detach()
                original_maximum.erase()
                if self.policy == LIBM_SOFTMAX_F32_RNE_POLICY:
                    body = exps.regions[0].block
                    original_exp = body.first_op
                    exp_ops, exp_result = build_libm_softmax_exp_rne(
                        original_exp.operands[0],
                        domain="nonpositive",
                        rounding_mode="rne",
                        floating_effects="unobserved",
                    )
                    body.insert_ops_before(exp_ops, original_exp)
                    original_exp.result.replace_all_uses_with(exp_result)
                    original_exp.detach()
                    original_exp.erase()
                records.append({"region": region, "shape": list(row["shape"])})
                continue
            new_ops, replacement = build_cpu_softmax_lastdim(
                score,
                backend_policy=self.policy,
                assume_finite_inputs=True,
                maximum_policy="finite",
            )
            block.insert_ops_before(new_ops, members[0])
            result.results[0].replace_all_uses_with(replacement)
            for op in reversed(members):
                op.detach()
                op.erase()
            records.append({"region": region, "shape": list(row["shape"])})
        module.verify()
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        output = workdir / "mx_host_softmax.mlir"
        output.write_text(text(module))
        receipt = {
            "schema": "mx_gemmini.host_softmax_rewrite.v1",
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "sidecar_sha256": hashlib.sha256(Path(sidecar).read_bytes()).hexdigest(),
            "profile_sha256": profile_sha256(self.profile),
            "numeric_policy": self.policy,
            "finite_input_contract": True,
            "finite_maximum_specialization": True,
            "regions": records,
        }
        if self.policy == LIBM_SOFTMAX_F32_RNE_POLICY:
            receipt["exponential_contract"] = {
                "domain": "nonpositive",
                "rounding_mode": "rne",
                "floating_effects": "unobserved",
                "underflow_cutoff": -104.0,
                "underflow_result_bits": 0,
            }
        output.with_name("mx_host_softmax_rewrite.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        return output
