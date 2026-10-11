"""Archived model-derived FP4 programs must preserve source and output bounds."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.model_projection import portable_projection_shape
from mx_gemmini_support.source_payload import (load_bundle,
                                               validate_model2mlir_projection)
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/model2mlir_small_llama_mlp_fp4_padding_e9ded36_266c593"
GRAPH = (ROOT / "docs/evidence/model2mlir_small_llama_precision_e9ded36_266c593/"
         "small_llama.model2mlir.mlir.gz")
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name,region,shape,padded,logical", [
    ("g", "matmul_6", [8, 344, 128], [32, 352, 128], 2752),
    ("dn", "matmul_8", [8, 128, 344], [32, 128, 352], 1024),
])
def test_complete_model_derived_projection_matched_on_spike(
        name: str, region: str, shape: list[int], padded: list[int], logical: int) -> None:
    site_dir = ARCHIVE / name
    manifest, _ = load_bundle(site_dir / "bundle")
    summary = json.loads((site_dir / "summary.json").read_text())
    obj = json.loads((site_dir / "object/object_manifest.json").read_text())
    run = json.loads((site_dir / "run/artifact_manifest.json").read_text())
    worklist = build_model2mlir_worklist(gzip.decompress(GRAPH.read_bytes()).decode(),
                                         PROFILE)
    source_site = next(site for site in worklist["rank2_matmuls"]
                       if site["region_id"] == region)
    assert manifest["model2mlir_projection"] == summary["site"]
    assert summary["site"]["fqn"] == source_site["fqn"]
    assert summary["site"]["source_shape_mnk"] == source_site["shape_mnk"] == shape
    assert summary["site"]["column_count"] == shape[1]
    assert summary["precision"] == manifest["precision"] == "FP4"
    assert manifest["shape_mnk"] == summary["padded_shape_mnk"] == padded
    assert portable_projection_shape((site_dir / "projection.model2mlir.mlir").read_text()) == tuple(padded)
    assert (summary["model_output_values"], summary["compared_bf16_values"]) == (
        logical, padded[0] * padded[1])
    assert summary["padding_output_values"] == padded[0] * padded[1] - logical
    assert summary["status"] == "complete_projection_matched_on_spike"
    assert summary["run_status"] == run["status"] == "model_projection_matched_on_pinned_spike"
    assert run["compared_bf16_outputs"] == padded[0] * padded[1]
    assert obj["object_sha256"] == summary["object_sha256"] == _sha(site_dir / "object/mx_issue.o")
    assert run["elf_sha256"] == summary["elf_sha256"] == _sha(site_dir / "run/mx_program.elf")
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert f"lowered MX {padded[0]}x{padded[1]}x{padded[2]}: 0 BF16 mismatches" in (
        site_dir / "run/spike.log").read_text()
    golden = np.frombuffer((site_dir / "bundle/golden_bf16.bin").read_bytes(),
                           dtype="<u2").reshape(padded[0], padded[1])
    assert np.count_nonzero(golden[shape[0]:]) == 0
    assert np.count_nonzero(golden[:shape[0], shape[1]:]) == 0


@pytest.mark.parametrize("name,axis", [("g", 1), ("dn", 2)])
def test_padding_extent_is_bound_to_source_shape(name: str, axis: int) -> None:
    manifest, _ = load_bundle(ARCHIVE / name / "bundle")
    changed = json.loads(json.dumps(manifest))
    changed["shape_mnk"][axis] -= 32
    with pytest.raises(ValueError, match="padded shape"):
        validate_model2mlir_projection(changed)


def test_captured_operands_identify_archived_model_sites() -> None:
    inputs = ARCHIVE / "inputs"
    manifest = json.loads((inputs / "manifest.json").read_text())
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    assert receipt["status"] == "two_complete_projections_matched_on_pinned_spike"
    assert receipt["input_manifest_sha256"] == _sha(inputs / "manifest.json")
    assert receipt["source_model_mlir_sha256"] == _sha(GRAPH)
    assert manifest["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert _sha(inputs / "input_ids.npy") == manifest["input_ids_npy_sha256"]
    for name in ("g", "dn"):
        site = manifest["sites"][name]
        assert _sha(inputs / f"{name}.activation.npy") == site["activation_npy_sha256"]
        assert _sha(inputs / f"{name}.weight.npy") == site["weight_npy_sha256"]
        summary = json.loads((ARCHIVE / name / "summary.json").read_text())
        assert summary["site"]["input_sha256"] == site["activation_sha256"]
        assert summary["site"]["weight_sha256"] == site["weight_sha256"]
        assert receipt["sites"][name]["object_sha256"] == summary["object_sha256"]
        assert receipt["sites"][name]["elf_sha256"] == summary["elf_sha256"]
