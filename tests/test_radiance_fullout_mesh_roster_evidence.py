"""Check the two-run Nicolas Spike archive for the complete BF16 MX roster."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.source_payload import validate_target_mesh_reference
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_mx_fullout_mesh_roster_266c593"
INDEX = json.loads((EVIDENCE / "index.json").read_text())
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _asset(root: Path, name: str) -> bytes:
    return gzip.decompress((root / f"{name}.gz").read_bytes())


def test_fullout_mesh_archive_covers_two_runs_and_separate_vpu_profile():
    assert INDEX["schema"] == "mx_gemmini.radiance_fullout_mesh_spike_matrix.v1"
    assert INDEX["status"] == (
        "fullout_mesh_rosters_and_vpu_selected_cases_matched_twice_on_pinned_spike")
    assert INDEX["compiler_revision"] == "a541a65aaff2704ca3ac1697c0d8fb1c903a9fc5"
    assert INDEX["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert INDEX["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert INDEX["profile_count"] == 3
    assert INDEX["case_count_per_run"] == 48
    assert INDEX["compared_bf16_outputs_per_run"] == 774144
    assert {entry["mesh_dim"]: entry["case_count"] for entry in INDEX["profiles"]} == {
        8: 23, 16: 2, 32: 23}


def test_fullout_mesh_archive_matches_every_generated_program_and_spike_log():
    for entry in INDEX["profiles"]:
        dim = entry["mesh_dim"]
        label = "vpu" if dim == 16 else f"dim{dim}"
        root = EVIDENCE / label
        profile = load_profile(PROFILES / f"{entry['profile_name']}.json")
        assert profile["geometry"]["mesh_columns"] == dim
        assert profile["resources"]["vpu"] is (dim == 16)
        assert profile_sha256(profile) == entry["profile_sha256"]
        first = (root / "qualification.json").read_bytes()
        second = (root / "qualification_repro.json").read_bytes()
        assert _sha(first) == entry["qualification_sha256"]
        assert _sha(second) == entry["qualification_repro_sha256"]
        a, b = json.loads(first), json.loads(second)
        assert a["compiler_revision"] == b["compiler_revision"] == INDEX["compiler_revision"]
        assert a["compared_bf16_outputs"] == b["compared_bf16_outputs"] == (
            entry["compared_bf16_outputs_per_run"])
        assert len(a["rows"]) == len(b["rows"]) == entry["case_count"]
        for row_a, row_b, case in zip(a["rows"], b["rows"], entry["cases"]):
            assert row_a["driver"] == row_b["driver"] == case["driver"]
            assert row_a["precision"] == row_b["precision"] == case["precision"]
            assert row_a["compared_bf16_outputs"] == row_b["compared_bf16_outputs"] == (
                case["compared_bf16_outputs"])
            assert row_a["elf_sha256"] == row_b["elf_sha256"] == case["elf_sha256"]
            assert row_a["spike_log_sha256"] == row_b["spike_log_sha256"] == (
                case["spike_log_sha256"])
            normalized_a, normalized_b = row_a.copy(), row_b.copy()
            normalized_a.pop("artifact_manifest_sha256")
            normalized_b.pop("artifact_manifest_sha256")
            assert normalized_a == normalized_b
            case_root = root / "cases" / case["label"]
            receipt_first = (case_root / "artifact_manifest.json").read_bytes()
            receipt_second = (case_root / "artifact_manifest_repro.json").read_bytes()
            assert _sha(receipt_first) == row_a["artifact_manifest_sha256"] == (
                case["artifact_manifest_sha256"])
            assert _sha(receipt_second) == row_b["artifact_manifest_sha256"] == (
                case["artifact_manifest_repro_sha256"])
            x, y = json.loads(receipt_first), json.loads(receipt_second)
            assert x["spike_exit_code"] == y["spike_exit_code"] == 0
            assert x["compared_bf16_outputs"] == y["compared_bf16_outputs"] == (
                case["compared_bf16_outputs"])
            if dim != 16:
                assert x["spike_extension_name"] == y["spike_extension_name"] == (
                    f"gemmini_dim{dim}")
            else:
                assert "spike_extension_name" not in x
            assert x["status"] == y["status"] == (
                "source_golden_matched_on_pinned_spike" if dim == 16 else
                "target_mesh_reference_matched_on_pinned_spike")
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
            assert b"0 BF16 mismatches" in log
            manifest = json.loads((case_root / "bundle_manifest.json").read_text())
            assert _sha((case_root / "bundle_manifest.json").read_bytes()) == (
                row_a["bundle_manifest_sha256"])
            assert _sha(_asset(case_root, "golden_bf16.bin")) == (
                manifest["resources"]["golden_bf16"]["sha256"])
            if dim != 16:
                validate_target_mesh_reference(manifest)
                assert manifest["target_mesh_reference"] == row_a["target_mesh_reference"]
                assert manifest["target_mesh_reference"]["schema"] == (
                    "mx_gemmini.radiance_target_mesh_reference.v2")
                assert _sha(_asset(case_root, "source_golden_bf16.bin")) == (
                    row_a["source_golden_sha256"])
                assert _sha(_asset(case_root, "golden_bf16.bin")) == (
                    row_a["target_golden_sha256"])
