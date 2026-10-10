"""Pin the independent data model used for five missing DIM16 MX cells."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_generated_modes_266c593"
MODES = {
    "e2m3_e4m3s", "e3m2_e3m2", "e4m3s_e2m3",
    "e4m3s_e4m3", "e4m3_e4m3s",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generated_headers_have_pinned_model_provenance() -> None:
    manifest = json.loads(
        (EVIDENCE / "mx_gemmini_generated_modes_manifest.json").read_text())
    assert manifest["schema"] == "mx_gemmini.nicolas_generated_asymmetric_headers.v1"
    assert manifest["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert manifest["microxcaling_revision"] == (
        "7bc41952de394f5cc5e782baf132e7c7542eb4e4")
    assert manifest["baseline_sha256"] == (
        "b4c7f87d11a0e096bc8ccfe2234c88c992ac5d2bccdb7b4d09c9aca0d770d8dc")
    assert manifest["wrapper_sha256"] == _sha(
        ROOT / "tools/generate_nicolas_missing_headers.py")
    assert set(manifest["headers_sha256"]) == MODES
    for name, digest in manifest["headers_sha256"].items():
        assert _sha(EVIDENCE / f"matmul_data_asym_{name}.h") == digest


def test_generated_mode_spike_evidence_and_remaining_stock_model_gap() -> None:
    manifest_sha = _sha(EVIDENCE / "mx_gemmini_generated_modes_manifest.json")
    profile = json.loads((ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                          "MxAllAsymGemminiRocketConfig.json").read_text())
    legal = {json.dumps(cell, sort_keys=True) for cell in profile["legal_compute"]}
    generated_passes: set[str] = set()
    failure = None
    for name in sorted(MODES):
        folder = EVIDENCE / name
        receipt = json.loads((folder / "first.json").read_text())
        resource_manifest = json.loads((folder / "resource_manifest.json").read_text())
        cell = resource_manifest["recipe"]["compute"]
        encoded = json.dumps(cell, sort_keys=True)
        assert encoded in legal
        assert resource_manifest["origin"] == "nicolas_generated_header_specialization"
        assert receipt["source_generation_manifest_sha256"] == manifest_sha
        assert receipt["source_header_sha256"] == _sha(
            EVIDENCE / f"matmul_data_asym_{name}.h")
        assert receipt["compiler_revision"].startswith("fc9c077")
        assert receipt["rtl_revision"].startswith("266c593")
        assert receipt["model2mlir_revision"].startswith("7485a829")
        assert receipt["compared_bf16_outputs"] == 4096
        assert receipt["files_sha256"]["asymmetric_bound.mlir"] == _sha(
            folder / "asymmetric_bound.mlir")
        assert receipt["files_sha256"]["model2mlir.mlir"] == _sha(
            folder / "model2mlir.mlir")
        assert receipt["physical_program_sha256"] == _sha(
            folder / "physical_program.json")
        assert receipt["resource_manifest_sha256"] == _sha(
            folder / "resource_manifest.json")
        assert receipt["generated_c_sha256"] == _sha(folder / "mx_issue.c")
        assert receipt["spike_log_sha256"] == _sha(folder / "stock_spike.log")
        if name == "e4m3s_e4m3":
            assert receipt["status"] == "source_golden_failed_on_pinned_spike"
            assert receipt["spike_exit_code"] == 1
            assert "4094 BF16 mismatches" in (folder / "stock_spike.log").read_text()
            failure = encoded
        else:
            assert receipt["status"] == "source_golden_matched_on_pinned_spike"
            assert receipt["spike_exit_code"] == 0
            assert "0 BF16 mismatches" in (folder / "stock_spike.log").read_text()
            reproduced = json.loads((folder / "repro.json").read_text())
            for run in (receipt, reproduced):
                run["build_log_sha256"].pop("link.log")
            assert receipt == reproduced
            generated_passes.add(encoded)
    assert len(generated_passes) == 4

    checked_in = json.loads((ROOT / "docs/evidence/"
                             "nicolas_asym_matrix_dim16_all_plus_fp4_266c593/"
                             "matrix_first.json").read_text())
    checked_in_modes = {json.dumps(row["compute"], sort_keys=True)
                        for row in checked_in["rows"]}
    radiance_direct = json.dumps({
        "activation_format": "fp8_e4m3", "activation_projection": "direct",
        "weight_format": "fp8_e4m3", "weight_projection": "direct",
        "pe_mode": 8,
    }, sort_keys=True)
    qualified = checked_in_modes | generated_passes | {radiance_direct}
    assert len(qualified) == 35
    assert legal - qualified == {failure}
    index = json.loads((EVIDENCE / "qualification.json").read_text())
    assert (index["legal_mode_count"], index["checked_in_source_passes"],
            index["radiance_source_passes"], index["generated_stock_spike_passes"],
            index["generated_stock_spike_failures"],
            index["distinct_stock_spike_passes"]) == (36, 30, 1, 4, 1, 35)
    assert {json.dumps(cell, sort_keys=True) for cell in
            index["unqualified_legal_compute"]} == {failure}
    assert index["generation_manifest_sha256"] == manifest_sha
    for row in index["generated_rows"]:
        folder = EVIDENCE / row["mode"]
        assert row["first_receipt_sha256"] == _sha(folder / "first.json")
        if row["repro_receipt_sha256"] is not None:
            assert row["repro_receipt_sha256"] == _sha(folder / "repro.json")


def test_local_spike_fix_diagnostic_is_kept_separate() -> None:
    folder = EVIDENCE / "e4m3s_e4m3"
    diagnostic = json.loads((folder / "patch_diagnostic.json").read_text())
    stock = json.loads((folder / "first.json").read_text())
    assert diagnostic["compiler_revision"] == stock["compiler_revision"]
    assert diagnostic["elf_sha256"] == stock["elf_sha256"]
    assert diagnostic["source_header_sha256"] == stock["source_header_sha256"]
    assert diagnostic["stock_spike_log_sha256"] == stock["spike_log_sha256"]
    assert diagnostic["patched_spike_log_sha256"] == _sha(folder / "patched_spike.log")
    assert diagnostic["patch_sha256"] == _sha(
        EVIDENCE / "spike_weight_lut_quad_candidate.patch")
    assert (diagnostic["stock_mismatches"], diagnostic["patched_mismatches"],
            diagnostic["compared_bf16_outputs"]) == (4094, 0, 4096)
