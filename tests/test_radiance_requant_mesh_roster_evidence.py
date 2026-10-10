"""Verify the repeated DIM8/DIM16/DIM32 Radiance requant Spike results."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.source_payload import validate_target_mesh_reference
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_mx_requant_mesh_roster_266c593"
INDEX = json.loads((EVIDENCE / "index.json").read_text())
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _asset(root: Path, name: str) -> bytes:
    return gzip.decompress((root / f"{name}.gz").read_bytes())


def test_requant_archive_covers_all_source_drivers_on_three_meshes():
    assert INDEX["schema"] == "mx_gemmini.radiance_requant_mesh_spike_matrix.v1"
    assert INDEX["status"] == "three_requant_rosters_matched_twice_on_pinned_spike"
    assert INDEX["compiler_revision"] == "13417df20c73480a51736af38a56141438d9a3a5"
    assert INDEX["source_revision"] == "80f84caedbabc663a7433c1da4455b936cca41f3"
    assert INDEX["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert INDEX["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert INDEX["profile_count"] == 3
    assert INDEX["case_count_per_run"] == 24
    assert INDEX["compared_quantized_output_bytes_per_run"] == 270336
    assert INDEX["compared_output_scales_per_run"] == 9984
    assert {entry["mesh_dim"]: entry["case_count"] for entry in INDEX["profiles"]} == {
        8: 8, 16: 8, 32: 8}


def test_requant_archive_binds_every_code_scale_elf_and_spike_log():
    for entry in INDEX["profiles"]:
        dim = entry["mesh_dim"]
        root = EVIDENCE / f"dim{dim}"
        profile = load_profile(PROFILES / f"{entry['profile_name']}.json")
        assert profile["geometry"]["mesh_columns"] == dim
        assert profile_sha256(profile) == entry["profile_sha256"]
        first = (root / "qualification.json").read_bytes()
        second = (root / "qualification_repro.json").read_bytes()
        assert _sha(first) == entry["qualification_sha256"]
        assert _sha(second) == entry["qualification_repro_sha256"]
        a, b = json.loads(first), json.loads(second)
        assert a["compiler_revision"] == b["compiler_revision"] == INDEX["compiler_revision"]
        assert a["compared_quantized_output_bytes"] == b[
            "compared_quantized_output_bytes"] == entry[
                "compared_quantized_output_bytes_per_run"]
        assert a["compared_output_scales"] == b["compared_output_scales"] == (
            entry["compared_output_scales_per_run"])
        assert len(a["rows"]) == len(b["rows"]) == entry["case_count"]
        assert {row["precision"] for row in a["rows"]} == {"FP4", "FP6", "FP8"}
        for row_a, row_b, case in zip(a["rows"], b["rows"], entry["cases"]):
            assert row_a["driver"] == row_b["driver"] == case["driver"]
            assert row_a["precision"] == row_b["precision"] == case["precision"]
            assert row_a["compared_quantized_output_bytes"] == row_b[
                "compared_quantized_output_bytes"] == case[
                    "compared_quantized_output_bytes"]
            assert row_a["compared_output_scales"] == row_b[
                "compared_output_scales"] == case["compared_output_scales"]
            normalized_a, normalized_b = row_a.copy(), row_b.copy()
            normalized_a.pop("artifact_manifest_sha256")
            normalized_b.pop("artifact_manifest_sha256")
            assert normalized_a == normalized_b
            case_root = root / "cases" / Path(case["driver"]).stem
            receipt_first = (case_root / "artifact_manifest.json").read_bytes()
            receipt_second = (case_root / "artifact_manifest_repro.json").read_bytes()
            assert _sha(receipt_first) == row_a["artifact_manifest_sha256"] == (
                case["artifact_manifest_sha256"])
            assert _sha(receipt_second) == row_b["artifact_manifest_sha256"] == (
                case["artifact_manifest_repro_sha256"])
            x, y = json.loads(receipt_first), json.loads(receipt_second)
            assert x["spike_exit_code"] == y["spike_exit_code"] == 0
            assert x["status"] == y["status"] == (
                "radiance_header_matched_on_pinned_spike" if dim == 16 else
                "target_mesh_reference_matched_on_pinned_spike")
            basis = "source" if dim == 16 else "target"
            code_field = (f"compared_{basis}_fp6_packed_bytes" if
                          row_a["precision"] == "FP6" else
                          f"compared_{basis}_fp8_codes")
            assert x[code_field] == y[code_field] == case[
                "compared_quantized_output_bytes"]
            assert x[f"compared_{basis}_e8m0_scales"] == y[
                f"compared_{basis}_e8m0_scales"] == case["compared_output_scales"]
            if dim != 16:
                assert x["spike_extension_name"] == y["spike_extension_name"] == (
                    f"gemmini_dim{dim}")
            normalized_x, normalized_y = x.copy(), y.copy()
            normalized_x.pop("build_log_sha256")
            normalized_y.pop("build_log_sha256")
            assert normalized_x == normalized_y
            for name, digest in (("mx_issue.c", row_a["generated_issue_sha256"]),
                                 ("physical_program.json", row_a["physical_program_sha256"]),
                                 ("mx_program.elf", row_a["elf_sha256"]),
                                 ("payload_bound.mlir", row_a["payload_bound_mlir_sha256"]),
                                 ("profile_bound.mlir", row_a["profile_bound_mlir_sha256"])):
                assert _sha(_asset(case_root, name)) == digest
            for name in ("mx_issue.c", "physical_program.json"):
                assert _sha(_asset(case_root, name)) == x["files_sha256"][name]
            for name in ("mx_driver.c", "mx_data.S"):
                assert _sha((case_root / name).read_bytes()) == x["files_sha256"][name]
            log = (case_root / "spike.log").read_bytes()
            assert _sha(log) == _sha((case_root / "spike_repro.log").read_bytes()) == (
                case["spike_log_sha256"])
            assert b"0 E8M0 scale mismatches" in log
            assert (b"0 Radiance FP6 packed-index mismatches" if
                    row_a["precision"] == "FP6" else
                    b"0 Radiance FP8 code mismatches") in log
            manifest_bytes = (case_root / "bundle_manifest.json").read_bytes()
            assert _sha(manifest_bytes) == row_a["bundle_manifest_sha256"]
            manifest = json.loads(manifest_bytes)
            code_name = "source_fp6_packed" if row_a["precision"] == "FP6" else "golden_fp8"
            nicolas_name = {"FP8": "nicolas_fp8", "FP4": "nicolas_fp4",
                            "FP6": "nicolas_fp6"}[row_a["precision"]]
            for asset_name, resource_name in (("expected_codes.bin", code_name),
                                              ("expected_scales.bin", "golden_output_scales"),
                                              ("nicolas_codes.bin", nicolas_name),
                                              ("nicolas_scales.bin", "nicolas_output_scales"),
                                              ("golden_bf16.bin", "golden_bf16")):
                assert _sha(_asset(case_root, asset_name)) == (
                    manifest["resources"][resource_name]["sha256"])
            if row_a["precision"] == "FP6":
                assert _sha(_asset(case_root, "output_lut.bin")) == (
                    manifest["resources"]["output_lut"]["sha256"])
            if dim == 16:
                assert _asset(case_root, "source_codes.bin") == _asset(
                    case_root, "expected_codes.bin")
                assert _asset(case_root, "source_scales.bin") == _asset(
                    case_root, "expected_scales.bin")
            else:
                validate_target_mesh_reference(manifest)
                assert manifest["target_mesh_reference"] == row_a["target_mesh_reference"]
                assert manifest["target_quant_reference"] == row_a["target_quant_reference"]
                policy = manifest["target_quant_reference"]
                assert _sha(_asset(case_root, "source_codes.bin")) == policy[
                    "source_code_sha256"]
                assert _sha(_asset(case_root, "source_scales.bin")) == policy[
                    "source_scale_sha256"]
                assert _sha(_asset(case_root, "expected_codes.bin")) == policy[
                    "target_code_sha256"]
                assert _sha(_asset(case_root, "expected_scales.bin")) == policy[
                    "target_scale_sha256"]
                assert _asset(case_root, "source_codes.bin") != _asset(
                    case_root, "expected_codes.bin")
                assert _sha(_asset(case_root, "source_golden_bf16.bin")) == row_a[
                    "source_golden_sha256"]
                assert _sha(_asset(case_root, "golden_bf16.bin")) == row_a[
                    "target_golden_sha256"]
