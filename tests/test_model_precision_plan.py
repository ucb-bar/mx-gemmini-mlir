"""Model precision intent must follow the selected RTL profile."""

from __future__ import annotations

import gzip
import hashlib
import json
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


def test_current_small_llama_attention_and_padding_proposal() -> None:
    archive = ROOT / "docs/evidence/model2mlir_small_llama_precision_e9ded36_266c593"
    graph = gzip.decompress((archive / "small_llama.model2mlir.mlir.gz").read_bytes()).decode()
    receipt = json.loads((archive / "receipt.json").read_text())
    assert hashlib.sha256(graph.encode()).hexdigest() == receipt["frontend_mlir_sha256"]
    worklist = build_model2mlir_worklist(graph, VPU)
    plan = build_model_precision_plan(worklist, VPU)
    assert worklist == json.loads((archive / "worklist.json").read_text())
    assert plan == json.loads((archive / "precision_proposal.json").read_text())
    assert plan["selected_format_counts"] == {"host": 6, "mxfp4": 9, "mxfp8": 4}
    attention = [site for site in plan["sites"] if site["role"] in {"qk", "pv"}]
    assert [(site["fqn"], site["role"]) for site in attention] == [
        ("blocks.0.attn", "qk"), ("blocks.0.attn", "pv"),
        ("blocks.1.attn", "qk"), ("blocks.1.attn", "pv")]
    assert all(site["selected_format"] == "mxfp8" and
               site["mx_padded_shape_bmnk"] == [4, 32, 32, 32]
               for site in attention)
    mlp = {site["fqn"]: site for site in plan["sites"]
           if site["role"] == "regular_matmul" and ".mlp." in (site["fqn"] or "")}
    assert len(mlp) == 6
    assert mlp["blocks.0.mlp.g"]["mx_padded_shape_mnk"] == [32, 352, 128]
    assert mlp["blocks.0.mlp.dn"]["mx_padded_shape_mnk"] == [32, 128, 352]
    assert all(site["selected_format"] == "host" and
               site["post_padding_candidate_format"] == "mxfp4"
               for site in mlp.values())
    incomplete = dict(worklist)
    incomplete["other_contraction_regions"] = [site for site in
        worklist["other_contraction_regions"] if site["region_id"] != "matmul_4"]
    incomplete_plan = build_model_precision_plan(incomplete, VPU)
    assert not any(site["role"] in {"qk", "pv"} and
                   site["fqn"] == "blocks.0.attn"
                   for site in incomplete_plan["sites"])
