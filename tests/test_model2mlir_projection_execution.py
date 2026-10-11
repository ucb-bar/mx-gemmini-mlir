"""The model-derived MX stripe must retain source identity and exact hardware output."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.model_projection import portable_projection_shape
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import (load_bundle, manifest_sha256,
                                              validate_model2mlir_projection)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/model2mlir_tinyllama_projection_a042643_266c593"
FULL_ARCHIVE = ROOT / "docs/evidence/model2mlir_tinyllama_full_qproj_a042643_266c593"
FULL_GRAPH = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984/tinyllama.mlir.gz"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_executed_projection_is_a_real_full_model_site() -> None:
    summary = json.loads((ARCHIVE / "summary.json").read_text())
    source = gzip.decompress(FULL_GRAPH.read_bytes()).decode()
    worklist = build_model2mlir_worklist(source, PROFILE)
    site = next(site for site in worklist["rank2_matmuls"]
                if site["region_id"] == "matmul_1")
    selected = summary["site"]
    assert summary["status"] == "full_k_projection_stripe_matched_on_spike"
    assert summary["scope"] == "one model2MLIR matmul output stripe; other model operations uncompiled"
    assert selected["source_mlir_sha256"] == worklist["source_mlir_sha256"]
    assert (selected["fqn"], selected["source_shape_mnk"], selected["source_op"]) == (
        site["fqn"], site["shape_mnk"], site["source_op"])
    assert selected["row_count"] == 8 and selected["column_count"] == 32
    assert summary["padded_shape_mnk"] == [16, 32, 2048]
    assert (summary["compared_bf16_values"], summary["model_output_values"],
            summary["padding_output_values"]) == (512, 256, 256)
    assert portable_projection_shape((ARCHIVE / "projection.model2mlir.mlir").read_text()) == (
        16, 32, 2048)
    inputs = json.loads((ARCHIVE / "input_receipt.json").read_text())
    assert inputs["activation_sha256"] == selected["input_sha256"]
    assert inputs["seed"] == 20261010


def test_model_payload_reproduces_physical_commands_and_spike_receipt() -> None:
    summary = json.loads((ARCHIVE / "summary.json").read_text())
    manifest, resources = load_bundle(ARCHIVE / "bundle")
    bound = (ARCHIVE / "projection.payload_bound.mlir").read_text()
    report = verify_ir(bound, PROFILE)
    assert report["contracts"] == 1 and report["source_resources"] == 4
    assert manifest["origin"] == "model2mlir_projection_slice"
    assert manifest["model2mlir_projection"] == summary["site"]
    assert "source_header_sha256" not in manifest
    physical = lower_bound_source(bound, PROFILE, manifest, resources)
    assert physical.receipt() == json.loads((ARCHIVE / "object/physical_program.json").read_text())
    obj = json.loads((ARCHIVE / "object/object_manifest.json").read_text())
    run = json.loads((ARCHIVE / "run/artifact_manifest.json").read_text())
    assert obj["payload_manifest_sha256"] == manifest_sha256(manifest)
    assert obj["object_sha256"] == summary["object_sha256"]
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert obj["allocated_data_section_bytes"] == 0
    assert run["status"] == "model_projection_matched_on_pinned_spike"
    assert run["compared_bf16_outputs"] == 512
    assert run["elf_sha256"] == summary["elf_sha256"]
    assert _sha(ARCHIVE / "run/artifact_manifest.json") == summary["run_receipt_sha256"]
    assert "lowered MX 16x32x2048: 0 BF16 mismatches" in (
        ARCHIVE / "run/spike.log").read_text()


def test_model_payload_rejects_provenance_and_shape_forgery() -> None:
    manifest, _ = load_bundle(ARCHIVE / "bundle")
    changed = json.loads(json.dumps(manifest))
    changed["model2mlir_projection"]["golden_model_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="site or slice"):
        validate_model2mlir_projection(changed)
    changed = json.loads(json.dumps(manifest))
    changed["shape_mnk"][0] = 32
    with pytest.raises(ValueError, match="padded shape"):
        validate_model2mlir_projection(changed)
    changed = json.loads(json.dumps(manifest))
    changed["source_header_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="distinct source contract"):
        validate_model2mlir_projection(changed)


def test_complete_q_projection_is_one_compiled_mx_program() -> None:
    summary = json.loads((FULL_ARCHIVE / "summary.json").read_text())
    manifest, resources = load_bundle(FULL_ARCHIVE / "bundle")
    inputs = json.loads((FULL_ARCHIVE / "input_receipt.json").read_text())
    bound = (FULL_ARCHIVE / "projection.payload_bound.mlir").read_text()
    assert summary["status"] == "complete_projection_matched_on_spike"
    assert summary["scope"] == (
        "one complete rank-two model projection; other model operations uncompiled")
    assert summary["site"]["source_shape_mnk"] == [8, 2048, 2048]
    assert summary["site"]["column_count"] == 2048
    assert summary["site"]["weight_sha256"] == inputs["weight_sha256"]
    assert summary["padded_shape_mnk"] == [16, 2048, 2048]
    assert (summary["compared_bf16_values"], summary["model_output_values"],
            summary["padding_output_values"]) == (32768, 16384, 16384)
    assert manifest["tile_mnk"] == [16, 32, 128]
    assert portable_projection_shape((FULL_ARCHIVE / "projection.model2mlir.mlir").read_text()) == (
        16, 2048, 2048)
    report = verify_ir(bound, PROFILE)
    assert report["contracts"] == 1 and report["source_resources"] == 4
    physical = lower_bound_source(bound, PROFILE, manifest, resources)
    archived = gzip.decompress((FULL_ARCHIVE / "object/physical_program.json.gz").read_bytes())
    receipt = json.loads(archived)
    assert physical.receipt() == receipt
    assert len(receipt["steps"]) == 36998
    assert len(receipt["plan"]["output_tiles"]) == 64
    obj = json.loads((FULL_ARCHIVE / "object/object_manifest.json").read_text())
    run = json.loads((FULL_ARCHIVE / "run/artifact_manifest.json").read_text())
    assert obj["physical_program_sha256"] == hashlib.sha256(archived).hexdigest()
    assert obj["issuer_opt_level"] == "-O1"
    assert obj["object_sha256"] == summary["object_sha256"]
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert run["compiler_revision"].startswith("816c5dd")
    assert run["status"] == "model_projection_matched_on_pinned_spike"
    assert run["compared_bf16_outputs"] == 32768
    assert run["elf_sha256"] == summary["elf_sha256"]
    assert _sha(FULL_ARCHIVE / "run/artifact_manifest.json") == summary["run_receipt_sha256"]
    assert "lowered MX 16x2048x2048: 0 BF16 mismatches" in (
        FULL_ARCHIVE / "run/spike.log").read_text()
