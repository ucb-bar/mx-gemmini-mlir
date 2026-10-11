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
    assert build_model_precision_plan(from_merlin, PROFILE) == build_model_precision_plan(from_local, PROFILE)
