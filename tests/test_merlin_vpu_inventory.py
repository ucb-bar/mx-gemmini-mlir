"""The model builder accounts for direct and converted softmax boundaries."""

from pathlib import Path

import pytest

from mx_gemmini_support.merlin_vpu_inventory import audit_softmax_boundaries
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")


def test_bf16_rank_two_source_softmax_cannot_silently_fall_back() -> None:
    pytest.importorskip("merlin.targetgen.contract.linalg_iface")
    source = (ROOT / "docs/evidence/nicolas_vpu_softmax_model2mlir_329718b/"
              "softmax.model2mlir.mlir").read_text()
    inventory = audit_softmax_boundaries(source, PROFILE)
    assert inventory["softmax_regions"] == 1
    assert inventory["direct_profile_matches_unlowered"] == 1
    assert inventory["host_owned_regions"] == 0
    assert inventory["regions"][0]["input_shape"] == [16, 32]
    assert inventory["regions"][0]["input_dtype"] == "bf16"


def test_f32_attention_softmax_requires_conversion_before_vpu_placement() -> None:
    pytest.importorskip("merlin.targetgen.contract.linalg_iface")
    source = (ROOT / "docs/evidence/nicolas_attn_vpu_source_baseline_e9ded36_0344bcb/"
              "attention.model2mlir.mlir").read_text()
    inventory = audit_softmax_boundaries(source, PROFILE)
    assert inventory["softmax_regions"] == 1
    assert inventory["direct_profile_matches_unlowered"] == 0
    assert inventory["host_owned_regions"] == 1
    assert inventory["regions"][0]["input_shape"] == [1, 32, 64]
    assert inventory["regions"][0]["row_width"] == 64
    assert inventory["regions"][0]["reason"] == (
        "f32_or_higher_rank_requires_conversion_tiling_and_numeric_policy")
