"""Merlin's shared graph reader must give the same MX placement decisions."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from mx_gemmini_support.merlin_inventory import build_mx_worklist_from_merlin
from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.model_precision_plan import build_model_precision_plan
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")


@pytest.mark.parametrize("model", (
    "tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma2_2b", "smolvla"))
def test_merlin_inventory_preserves_mx_precision_plan(model: str) -> None:
    pytest.importorskip("merlin.targetgen.contract.linalg_iface")
    source = gzip.decompress((ARCHIVE / f"{model}.mlir.gz").read_bytes()).decode()
    from_merlin = build_mx_worklist_from_merlin(source, PROFILE)
    from_local = build_model2mlir_worklist(source, PROFILE)
    assert from_merlin["source_mlir_sha256"] == from_local["source_mlir_sha256"]
    assert from_merlin["contraction_region_count"] == from_local["contraction_region_count"]
    assert from_merlin["rank2_matmuls"] == from_local["rank2_matmuls"]
    assert from_merlin["rank3_batch_matmuls"] == from_local["rank3_batch_matmuls"]
    assert [(x["region_id"], x["source_op"], x["fqn"])
            for x in from_merlin["other_contraction_regions"]] == [
                (x["region_id"], x["source_op"], x["fqn"])
                for x in from_local["other_contraction_regions"]]
    graph_plan = build_model_precision_plan(from_merlin, PROFILE)
    local_plan = build_model_precision_plan(from_local, PROFILE)
    if model != "smolvla":
        assert graph_plan == local_plan
    else:
        # The older text worklist can route these source-compatible batched
        # matmuls, but cannot identify their QK/PV operand policies without
        # the structural SSA edges Merlin supplies.
        assert len(from_merlin["attention_pairs"]) >= 2
        assert graph_plan["selected_format_counts"] == local_plan["selected_format_counts"]
        changed = [(new["region_id"], new["role"], old["role"])
                   for new, old in zip(graph_plan["sites"], local_plan["sites"])
                   if new != old]
        assert changed == [
            ("matmul_101", "qk", "regular_batched_matmul"),
            ("matmul_102", "pv", "regular_batched_matmul"),
            ("matmul_113", "qk", "regular_batched_matmul"),
            ("matmul_114", "pv", "regular_batched_matmul"),
        ]
