"""Archived model QK heads must retain graph provenance and exact Spike output."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.model_projection import portable_projection_shape
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import (
    load_bundle, manifest_sha256, validate_model2mlir_attention_qk,
)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/model2mlir_four_model_fp8_qk_vpu_a042643_266c593"
FRONTEND = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")
MODELS = ("tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma2_2b", "smolvla")


@pytest.mark.parametrize("model", MODELS)
def test_model_qk_head_matches_source_site_and_pinned_spike(model: str) -> None:
    folder = ARCHIVE / model
    summary = json.loads((folder / "summary.json").read_text())
    inputs = json.loads((folder / "input_receipt.json").read_text())
    manifest, resources = load_bundle(folder / "bundle")
    obj = json.loads((folder / "object/object_manifest.json").read_text())
    run = json.loads((folder / "run/artifact_manifest.json").read_text())
    frontend = gzip.decompress((FRONTEND / f"{model}.mlir.gz").read_bytes()).decode()
    worklist = build_model2mlir_worklist(frontend, PROFILE)
    site = next(row for row in worklist["rank3_batch_matmuls"]
                if row["region_id"] == summary["site"]["region_id"])
    batch, m, n, k = site["shape_bmnk"]
    padded_m, padded_n, padded_k = summary["padded_shape_mnk"]
    assert summary["status"] == "qk_head_matched_on_spike"
    assert summary["site"]["source_mlir_sha256"] == worklist["source_mlir_sha256"]
    assert summary["site"]["source_shape_bmnk"] == [batch, m, n, k]
    assert summary["site"]["fqn"] == site["fqn"] == inputs["fqn"]
    assert summary["site"]["batch_index"] == inputs["batch_index"] == 0
    assert summary["site"]["input_sha256"] == inputs["activation_sha256"]
    assert summary["site"]["weight_sha256"] == inputs["weight_sha256"]
    assert manifest["origin"] == "model2mlir_attention_qk_head"
    assert manifest["precision"] == "FP8"
    assert manifest["model2mlir_attention_qk"] == summary["site"]
    assert (padded_m, padded_n, padded_k) == (
        ((m + 15) // 16) * 16, ((n + 31) // 32) * 32, k)
    assert summary["model_output_values"] == m * n
    assert summary["compared_bf16_values"] == padded_m * padded_n
    assert len(resources["golden_bf16"]) == 2 * padded_m * padded_n
    assert portable_projection_shape((folder / "qk.model2mlir.mlir").read_text()) == (
        padded_m, padded_n, k)
    bound = (folder / "qk.payload_bound.mlir").read_text()
    assert verify_ir(bound, PROFILE)["contracts"] == 1
    physical = gzip.decompress((folder / "object/physical_program.json.gz").read_bytes())
    assert lower_bound_source(bound, PROFILE, manifest, resources).receipt() == json.loads(
        physical)
    assert hashlib.sha256(physical).hexdigest() == obj["physical_program_sha256"]
    assert obj["payload_manifest_sha256"] == manifest_sha256(manifest)
    assert obj["object_sha256"] == summary["object_sha256"]
    assert obj["allocated_data_section_bytes"] == 0
    assert run["status"] == "model_attention_qk_matched_on_pinned_spike"
    assert run["compared_bf16_outputs"] == padded_m * padded_n
    assert run["elf_sha256"] == summary["elf_sha256"]
    assert hashlib.sha256((folder / "run/artifact_manifest.json").read_bytes()).hexdigest() == (
        summary["run_receipt_sha256"])
    assert f"lowered MX {padded_m}x{padded_n}x{k}: 0 BF16 mismatches" in (
        folder / "run/spike.log").read_text()


def test_qk_manifest_rejects_other_precision_and_forged_geometry() -> None:
    manifest, _ = load_bundle(ARCHIVE / "tinyllama/bundle")
    changed = json.loads(json.dumps(manifest))
    changed["precision"] = "FP4"
    with pytest.raises(ValueError, match="distinct source contract"):
        validate_model2mlir_attention_qk(changed)
    changed = json.loads(json.dumps(manifest))
    changed["tile_mnk"][2] = 16
    with pytest.raises(ValueError, match="padding or K tile"):
        validate_model2mlir_attention_qk(changed)
    changed = json.loads(json.dumps(manifest))
    changed["model2mlir_attention_qk"]["golden_model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source site"):
        validate_model2mlir_attention_qk(changed)


def test_index_names_only_four_qk_heads() -> None:
    index = json.loads((ARCHIVE / "index.json").read_text())
    assert index["scope"] == (
        "one head of first QK batched matmul per reduced random-weight model; "
        "no attention epilogue or PV")
    assert [row["model"] for row in index["rows"]] == list(MODELS)
    for row in index["rows"]:
        summary = json.loads((ARCHIVE / row["model"] / "summary.json").read_text())
        assert all(row[key] == summary[key] for key in row if key != "model")
