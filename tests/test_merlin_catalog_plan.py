"""Catalog bindings identify exact source operations in Merlin's MLIR."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest

from mx_gemmini_support.merlin_catalog_plan import plan_merlin_catalog
from mx_gemmini_support import merlin_catalog_plan
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/model2mlir_small_llama_precision_e9ded36_266c593"
PROFILE = load_profile(
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")


def test_small_llama_catalog_plan_binds_every_selected_site(tmp_path, monkeypatch) -> None:
    pytest.importorskip("merlin.system.offload")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(ROOT))
    source = tmp_path / "small_llama.mlir"
    source.write_bytes(gzip.decompress((ARCHIVE / "small_llama.model2mlir.mlir.gz").read_bytes()))
    plan = plan_merlin_catalog(source, PROFILE)
    assert plan["status"] == "exact_source_plan_no_executable_catalog"
    assert plan["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert plan["selected_operations"] == 19
    assert plan["unique_kernels"] == 6
    assert {row["region"] for row in plan["bindings"]} == {
        f"matmul_{index}" for index in range(19)}
    assert len({row["source_operation_ordinal"] for row in plan["bindings"]}) == 19
    assert {tuple(row["target_dtypes"]) for row in plan["bindings"]} == {
        ("f4E2M1FN", "f4E2M1FN", "bf16"),
        ("f8E4M3FN", "f8E4M3FN", "bf16"),
    }


def test_catalog_plan_reports_precision_proposals_outside_f32_source_abi(
        tmp_path, monkeypatch) -> None:
    pytest.importorskip("merlin.system.offload")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(ROOT))
    source = tmp_path / "small_llama.mlir"
    source.write_bytes(gzip.decompress((ARCHIVE / "small_llama.model2mlir.mlir.gz").read_bytes()))
    original = merlin_catalog_plan.build_model_precision_plan

    def with_unbound(worklist, profile):
        result = original(worklist, profile)
        result["sites"].append({"region_id": "conv_unbound", "source_op": "convolution_im2col_matmul",
                                "element_type": "bf16", "selected_format": "mxfp8"})
        return result

    monkeypatch.setattr(merlin_catalog_plan, "build_model_precision_plan", with_unbound)
    plan = plan_merlin_catalog(source, PROFILE)
    assert plan["selected_operations"] == 19
    assert plan["precision_proposals"] == 20
    assert plan["status"] == "exact_source_plan_partial_precision_coverage"
    assert plan["unbound_precision_proposals"] == [{
        "region": "conv_unbound", "source_op": "convolution_im2col_matmul",
        "element_type": "bf16", "selected_format": "mxfp8",
        "reason": "no zero-initialized direct f32 catalog contraction at this source region",
    }]


def test_checkpoint_cache_splits_fixed_sites_but_shares_issuer(tmp_path, monkeypatch) -> None:
    pytest.importorskip("merlin.system.offload")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(ROOT))
    source = tmp_path / "small_llama.mlir"
    source.write_bytes(gzip.decompress((ARCHIVE / "small_llama.model2mlir.mlir.gz").read_bytes()))
    baseline = plan_merlin_catalog(source, PROFILE)
    manifest = {
        "2": {"kind": "param", "dtype": "float32", "shape": [128, 128], "weight": "q.weight"},
        "3": {"kind": "param", "dtype": "float32", "shape": [128, 128], "weight": "k.weight"},
        "4": {"kind": "input", "dtype": "float32", "shape": [128, 128], "weight": "not_constant"},
    }
    cached = plan_merlin_catalog(source, PROFILE, checkpoint_manifest=manifest,
                                 cache_immutable_weights=True)
    assert cached["cached_checkpoint_kernels"] == 2
    assert cached["unique_kernels"] == baseline["unique_kernels"] + 2
    assert cached["unique_issuers"] == baseline["unique_kernels"]
    by_region = {row["region"]: row for row in cached["bindings"]}
    assert by_region["matmul_0"]["symbol"] != by_region["matmul_1"]["symbol"]
    cache_rows = [row for row in cached["kernels"] if row["cache_immutable_weight"]]
    assert {row["checkpoint_parameter_arg"] for row in cache_rows} == {2, 3}
    assert len({row["issuer_symbol"] for row in cache_rows}) == 1
