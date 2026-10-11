"""The model-derived physical schedule can accept live encoded buffers."""

from __future__ import annotations

from pathlib import Path

import pytest

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.runtime_issuer import emit_runtime_issuer
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/model2mlir_tinyllama_projection_a042643_266c593"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
FP4_ARCHIVE = ROOT / "docs/evidence/model2mlir_four_model_fp4_vpu_a042643_266c593/tinyllama"
FP4_PROFILE = load_profile(
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")


def _projection():
    manifest, resources = load_bundle(ARCHIVE / "bundle")
    program = lower_bound_source(
        (ARCHIVE / "projection.payload_bound.mlir").read_text(),
        PROFILE, manifest, resources)
    return manifest, program


def test_model_projection_exports_live_encoded_operand_abi() -> None:
    manifest, program = _projection()
    issuer = emit_runtime_issuer(program, manifest)
    assert issuer.shape_mnk == (16, 32, 2048)
    assert issuer.precision == "FP8"
    assert issuer.buffer_bytes == {
        "activation": 32768, "activation_scales": 1024,
        "weight": 65536, "weight_scales": 2048,
        "output_bf16": 1024, "scratch_output_scales": 2048,
    }
    signature = issuer.source.split("void mx_issue_encoded(", 1)[1].split(")", 1)[0]
    assert signature == (
        "const void *activation, const void *activation_scales, "
        "const void *weight, const void *weight_scales, "
        "const void *output_bf16, const void *scratch_output_scales"
    )
    assert "golden_bf16" not in issuer.source
    assert issuer.source.count(".insn r 0x7b") == sum(
        step.command.__class__.__name__ == "Command" for step in program.steps)


def test_runtime_issuer_refuses_changed_payload_and_non_bf16_readout() -> None:
    manifest, program = _projection()
    changed = {**manifest, "shape_mnk": [32, 32, 2048]}
    with pytest.raises(ValueError, match="identity differ"):
        emit_runtime_issuer(program, changed)
    from dataclasses import replace

    with pytest.raises(ValueError, match="BF16 matrix readout"):
        emit_runtime_issuer(replace(program, output_format="fp8_e4m3"), manifest)


def test_fp4_projection_exports_the_same_runtime_buffer_abi() -> None:
    manifest, resources = load_bundle(FP4_ARCHIVE / "bundle")
    program = lower_bound_source(
        (FP4_ARCHIVE / "projection.payload_bound.mlir").read_text(),
        FP4_PROFILE, manifest, resources)
    issuer = emit_runtime_issuer(program, manifest)
    assert issuer.precision == "FP4"
    assert issuer.shape_mnk == (32, 2048, 2048)
    assert issuer.buffer_bytes["weight"] == 2_097_152
    assert issuer.buffer_bytes["output_bf16"] == 131_072
    assert "golden_bf16" not in issuer.source
