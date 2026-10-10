"""Check all Radiance FP8/FP4 source drivers on Nicolas's VPU-enabled MX profile."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_mx_vpu_legal_roster_266c593"
INDEX = json.loads((EVIDENCE / "index.json").read_text())
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
FRONTEND = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22/frontend"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _asset(root: Path, name: str) -> bytes:
    return gzip.decompress((root / f"{name}.gz").read_bytes())


def test_vpu_profile_roster_scope_and_fp6_rejection():
    assert INDEX["schema"] == "mx_gemmini.radiance_vpu_legal_roster_spike_matrix.v1"
    assert INDEX["status"] == (
        "vpu_profile_fp8_fp4_source_rosters_matched_twice_on_pinned_spike")
    assert INDEX["compiler_revision"] == "e9965937463d412442d957826fe019c6f29e5b88"
    assert INDEX["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert INDEX["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert INDEX["source_revision"] == "80f84caedbabc663a7433c1da4455b936cca41f3"
    assert INDEX["legal_precisions"] == ["FP4", "FP8"]
    assert INDEX["case_count_per_run"] == 24
    assert {row["kind"]: row["case_count"] for row in INDEX["rosters"]} == {
        "fullout": 18, "requant": 6}
    profile = load_profile(PROFILE)
    assert profile["geometry"]["mesh_columns"] == 16
    assert profile["resources"]["vpu"] and not profile["resources"]["lut"]
    assert profile_sha256(profile) == INDEX["profile_sha256"]
    rejection_bytes = (EVIDENCE / "fp6_rejection.json").read_bytes()
    assert _sha(rejection_bytes) == INDEX["unsupported_fp6_rejection_sha256"]
    rejection = json.loads(rejection_bytes)
    assert rejection["status"] == "unsupported_profile_rejected"
    assert rejection["profile_sha256"] == INDEX["profile_sha256"]
    handoff = (FRONTEND /
               "mxgemm.fp6.singletile.tm128tn128tk128.fullout/mx_gemm.handoff.mlir")
    assert _sha(handoff.read_bytes()) == rejection["handoff_sha256"]
    with pytest.raises(ValueError, match="0 legal fp6_e3m2/lut choices") as error:
        bind_handoff(handoff.read_text(), profile)
    assert str(error.value) == rejection["diagnostic"]


def test_vpu_profile_rosters_reproduce_issued_elves_and_spike_results():
    for roster in INDEX["rosters"]:
        kind = roster["kind"]
        root = EVIDENCE / kind
        first = (root / "qualification.json").read_bytes()
        second = (root / "qualification_repro.json").read_bytes()
        assert _sha(first) == roster["qualification_sha256"]
        assert _sha(second) == roster["qualification_repro_sha256"]
        a, b = json.loads(first), json.loads(second)
        assert a["compiler_revision"] == b["compiler_revision"] == INDEX["compiler_revision"]
        assert a["profile_sha256"] == b["profile_sha256"] == INDEX["profile_sha256"]
        assert a["selected_precisions"] == b["selected_precisions"] == ["FP4", "FP8"]
        assert len(a["rows"]) == len(b["rows"]) == roster["case_count"]
        if kind == "fullout":
            assert a["compared_bf16_outputs"] == b["compared_bf16_outputs"] == (
                roster["compared_bf16_outputs_per_run"]) == 294912
        else:
            assert a["compared_quantized_output_bytes"] == b[
                "compared_quantized_output_bytes"] == roster[
                    "compared_quantized_output_bytes_per_run"] == 73728
            assert a["compared_output_scales"] == b["compared_output_scales"] == (
                roster["compared_output_scales_per_run"]) == 2304
        for row_a, row_b, case in zip(a["rows"], b["rows"], roster["cases"]):
            assert row_a["driver"] == row_b["driver"] == case["driver"]
            assert row_a["precision"] == row_b["precision"] == case["precision"]
            assert row_a["precision"] in {"FP4", "FP8"}
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
            normalized_x, normalized_y = x.copy(), y.copy()
            normalized_x.pop("build_log_sha256")
            normalized_y.pop("build_log_sha256")
            assert normalized_x == normalized_y
            assert x["spike_exit_code"] == y["spike_exit_code"] == 0
            assert x["elf_sha256"] == y["elf_sha256"] == row_a["elf_sha256"]
            assert x["spike_log_sha256"] == y["spike_log_sha256"] == (
                row_a["spike_log_sha256"])
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
            program = json.loads(_asset(case_root, "physical_program.json"))
            assert all(step["phase"] != "vpu" for step in program["steps"])
            log = (case_root / "spike.log").read_bytes()
            assert _sha(log) == _sha((case_root / "spike_repro.log").read_bytes()) == (
                case["spike_log_sha256"])
            manifest_bytes = (case_root / "bundle_manifest.json").read_bytes()
            assert _sha(manifest_bytes) == row_a["bundle_manifest_sha256"]
            manifest = json.loads(manifest_bytes)
            assert manifest["origin"] == "radiance_source_header_specialization"
            assert _sha(_asset(case_root, "golden_bf16.bin")) == (
                manifest["resources"]["golden_bf16"]["sha256"])
            if kind == "fullout":
                assert x["status"] == "source_golden_matched_on_pinned_spike"
                assert x["compared_bf16_outputs"] == case["compared_bf16_outputs"]
                assert b"0 BF16 mismatches" in log
            else:
                assert x["status"] == "radiance_header_matched_on_pinned_spike"
                assert x["compared_source_fp8_codes"] == (
                    case["compared_quantized_output_bytes"])
                assert x["compared_source_e8m0_scales"] == (
                    case["compared_output_scales"])
                assert b"0 Radiance FP8 code mismatches" in log
                assert b"0 E8M0 scale mismatches" in log
                assert _sha(_asset(case_root, "expected_codes.bin")) == (
                    manifest["resources"]["golden_fp8"]["sha256"])
                assert _sha(_asset(case_root, "expected_scales.bin")) == (
                    manifest["resources"]["golden_output_scales"]["sha256"])
