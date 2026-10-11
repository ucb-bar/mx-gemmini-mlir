"""Four actual model-derived FP4 projections must remain auditable."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.source_payload import load_bundle, manifest_sha256
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/model2mlir_four_model_fp4_vpu_a042643_266c593"
FRONTEND = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")


@pytest.mark.parametrize("model", [
    "tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma2_2b", "smolvla",
])
def test_full_width_fp4_projection_matches_pinned_spike(model: str) -> None:
    folder = ARCHIVE / model
    summary = json.loads((folder / "summary.json").read_text())
    inputs = json.loads((folder / "input_receipt.json").read_text())
    manifest, resources = load_bundle(folder / "bundle")
    obj = json.loads((folder / "object/object_manifest.json").read_text())
    run = json.loads((folder / "run/artifact_manifest.json").read_text())
    frontend = gzip.decompress((FRONTEND / f"{model}.mlir.gz").read_bytes()).decode()
    worklist = build_model2mlir_worklist(frontend, PROFILE)
    site = next(row for row in worklist["rank2_matmuls"]
                if row["region_id"] == summary["site"]["region_id"])
    m, n, k = site["shape_mnk"]
    assert summary["status"] == "complete_projection_matched_on_spike"
    assert summary["precision"] == manifest["precision"] == "FP4"
    assert summary["site"]["source_mlir_sha256"] == worklist["source_mlir_sha256"]
    assert summary["site"]["source_shape_mnk"] == [m, n, k]
    assert summary["site"]["column_count"] == n
    assert summary["site"]["row_count"] == m
    assert summary["site"]["input_sha256"] == inputs["activation_sha256"]
    assert summary["site"]["weight_sha256"] == inputs["weight_sha256"]
    assert summary["excluded_addmm_bias_epilogue"] == (site["source_op"] == "addmm")
    assert manifest["model2mlir_projection"] == summary["site"]
    assert summary["padded_shape_mnk"] == manifest["shape_mnk"]
    assert len(resources["golden_bf16"]) == summary["compared_bf16_values"] * 2
    assert summary["model_output_values"] == m * n
    assert obj["payload_manifest_sha256"] == manifest_sha256(manifest)
    assert obj["object_sha256"] == summary["object_sha256"]
    assert obj["allocated_data_section_bytes"] == 0
    physical = gzip.decompress((folder / "object/physical_program.json.gz").read_bytes())
    assert hashlib.sha256(physical).hexdigest() == obj["physical_program_sha256"]
    assert run["status"] == "model_projection_matched_on_pinned_spike"
    assert run["compared_bf16_outputs"] == summary["compared_bf16_values"]
    assert run["elf_sha256"] == summary["elf_sha256"]
    assert hashlib.sha256((folder / "run/artifact_manifest.json").read_bytes()).hexdigest() == (
        summary["run_receipt_sha256"])
    assert f"lowered MX {manifest['shape_mnk'][0]}x{n}x{k}: 0 BF16 mismatches" in (
        folder / "run/spike.log").read_text()
    assert verify_ir((folder / "projection.payload_bound.mlir").read_text(), PROFILE)[
        "contracts"] == 1


def test_index_has_only_full_width_fp4_model_projections() -> None:
    index = json.loads((ARCHIVE / "index.json").read_text())
    assert index["scope"] == (
        "full-width layer-0 q_proj matmul only; reduced random-weight captures")
    assert len(index["rows"]) == 4
    for row in index["rows"]:
        summary = json.loads((ARCHIVE / row["model"] / "summary.json").read_text())
        assert all(row[key] == summary[key] for key in row if key != "model")
