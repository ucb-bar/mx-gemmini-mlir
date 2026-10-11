"""Live shape planning shares the exact source-qualified command schedule."""

from __future__ import annotations

from pathlib import Path

import pytest

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.dense_contraction import emit_dense_contraction, plan_dense_shape
from mx_gemmini_support.planned_runtime_issuer import emit_planned_runtime_issuer
from mx_gemmini_support.runtime_issuer import emit_runtime_issuer
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/model2mlir_small_llama_mlp_fp4_padding_e9ded36_266c593"
PROFILE = load_profile(
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")


@pytest.mark.parametrize("site", ["g", "dn"])
def test_shape_planned_commands_equal_payload_qualified_commands(site: str) -> None:
    archive = ARCHIVE / site
    manifest, resources = load_bundle(archive / "bundle")
    program = lower_bound_source((archive / "projection.payload_bound.mlir").read_text(),
                                 PROFILE, manifest, resources)
    qualified = emit_runtime_issuer(program, manifest)
    planned = emit_planned_runtime_issuer(
        PROFILE, tuple(manifest["shape_mnk"]), tuple(manifest["tile_mnk"]),
        precision="FP4")
    assert planned.source == qualified.source
    assert planned.buffer_bytes == qualified.buffer_bytes
    assert planned.plan == program.plan


def test_shape_planner_handles_new_multi_tile_fp8_without_payload() -> None:
    planned = emit_planned_runtime_issuer(
        PROFILE, (16, 128, 128), (16, 64, 64), precision="FP8")
    assert planned.plan["bf16_output_layout"] == "row_major_bf16"
    assert len(planned.plan["output_tiles"]) == 2
    assert planned.buffer_bytes["output_bf16"] == 4096
    assert "output_bf16" in planned.source
    assert ".insn r 0x7b" in planned.source


def test_shape_planner_rejects_profile_without_direct_fp4() -> None:
    fp8_profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json")
    with pytest.raises(ValueError, match="compute mode"):
        emit_planned_runtime_issuer(
            fp8_profile, (32, 64, 64), (32, 64, 64), precision="FP4")


@pytest.mark.parametrize("site,logical", [
    ("g", (8, 344, 128)), ("dn", (8, 128, 344)),
])
def test_dense_contraction_recipe_recovers_qualified_projection(
        site: str, logical: tuple[int, int, int]) -> None:
    archive = ARCHIVE / site
    manifest, resources = load_bundle(archive / "bundle")
    qualified = emit_runtime_issuer(
        lower_bound_source((archive / "projection.payload_bound.mlir").read_text(),
                           PROFILE, manifest, resources), manifest)
    generated = emit_dense_contraction(
        PROFILE, logical, precision="FP4", symbol="mx_dense_projection",
        issuer_symbol="mx_issue_encoded")
    assert generated.shape_plan.padded_shape_mnk == tuple(manifest["shape_mnk"])
    assert generated.shape_plan.tile_mnk == tuple(manifest["tile_mnk"])
    assert generated.issuer.source == qualified.source
    assert generated.kernel.logical_shape_mnk == logical
    assert generated.shape_plan.estimated_commands == len(
        lower_bound_source((archive / "projection.payload_bound.mlir").read_text(),
                           PROFILE, manifest, resources).steps)


def test_dense_shape_planning_records_multi_tile_fp8_attention() -> None:
    planned = plan_dense_shape(PROFILE, (8, 127, 128), precision="FP8")
    assert planned.padded_shape_mnk == (16, 128, 128)
    assert planned.tile_mnk == (16, 16, 128)
    assert planned.output_tiles == 8


def test_full_vocabulary_scale_selects_compact_issuer() -> None:
    planned = plan_dense_shape(PROFILE, (128, 151936, 1536), precision="FP4")
    assert planned.estimated_commands > 100_000
    generated = emit_dense_contraction(
        PROFILE, (128, 151936, 1536), precision="FP4",
        symbol="large_projection", issuer_symbol="large_issue")
    assert generated.issuer_mode == "compact_loop"
    assert len(generated.issuer.source) < 7000
    assert generated.shape_plan.output_tiles == 18992
