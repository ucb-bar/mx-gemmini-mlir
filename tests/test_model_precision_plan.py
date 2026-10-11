"""Model precision intent must follow the selected RTL profile."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.model_precision_plan import build_model_precision_plan
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984"
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"
VPU = load_profile(PROFILES / "MxE4M3Fp4VpuGemminiRocketConfig.json")
ALL = load_profile(PROFILES / "MxGemminiRocketConfig.json")


@pytest.mark.parametrize("model,attention_pairs", [
    ("tinyllama", 2),
    ("deepseek_r1_distill_qwen_1_5b", 1),
    ("gemma2_2b", 2),
    ("smolvla", 12),
])
def test_model_precision_is_graph_bound_and_vpu_legal(model: str,
                                                       attention_pairs: int) -> None:
    frontend = gzip.decompress((ARCHIVE / f"{model}.mlir.gz").read_bytes()).decode()
    worklist = build_model2mlir_worklist(frontend, VPU)
    plan = build_model_precision_plan(worklist, VPU)
    assert plan["status"] == "proposal_only_no_whole_model_executable"
    assert plan["profile_has_vpu"] is True
    assert plan["source_mlir_sha256"] == worklist["source_mlir_sha256"]
    assert len(plan["sites"]) == worklist["contraction_region_count"]
    assert len([s for s in plan["sites"] if s["role"] == "qk"]) == attention_pairs
    assert len([s for s in plan["sites"] if s["role"] == "pv"]) == attention_pairs
    assert all(s["selected_format"] == "mxfp8" and
               s["status"] == "requires_fp8_batched_lowering"
               for s in plan["sites"] if s["role"] in {"qk", "pv"})
    assert all(s["required_operand_boundary"] ==
               "quantize_q_and_k_to_mxfp8_with_e8m0_scales"
               for s in plan["sites"] if s["role"] == "qk")
    assert all(s["required_operand_boundary"] ==
               "quantize_softmax_probabilities_and_v_to_mxfp8_with_e8m0_scales"
               for s in plan["sites"] if s["role"] == "pv")
    assert all(s["selected_format"] == "mxfp4"
               for s in plan["sites"] if s["role"] == "regular_matmul" and
               s["shape_mnk"][1] % 32 == s["shape_mnk"][2] % 32 == 0)
    assert not any(s["selected_format"] == "mxfp6" for s in plan["sites"])


def test_attention_uses_fp8_even_when_non_vpu_profile_supports_fp6() -> None:
    frontend = gzip.decompress((ARCHIVE / "tinyllama.mlir.gz").read_bytes()).decode()
    worklist = build_model2mlir_worklist(frontend, ALL)
    plan = build_model_precision_plan(worklist, ALL)
    assert plan["profile_has_vpu"] is False
    assert all(s["selected_format"] == "mxfp8" and
               s["status"] == "requires_fp8_batched_lowering"
               for s in plan["sites"] if s["role"] in {"qk", "pv"})


def test_precision_plan_rejects_profile_swap() -> None:
    frontend = gzip.decompress((ARCHIVE / "tinyllama.mlir.gz").read_bytes()).decode()
    worklist = build_model2mlir_worklist(frontend, ALL)
    with pytest.raises(ValueError, match="worklist and MX profile do not match"):
        build_model_precision_plan(worklist, VPU)
