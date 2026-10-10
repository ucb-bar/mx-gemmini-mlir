"""Verify direct stock-Spike receipts for the remaining Rocket MX wrappers."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_mx_rocket_wrapper_matrix import _output_identity


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_rocket_wrapper_matrix_266c593"
INDEX = json.loads((EVIDENCE / "index.json").read_text())
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_remaining_rocket_wrapper_matrix_has_eight_named_profiles() -> None:
    assert INDEX["schema"] == "mx_gemmini.rocket_wrapper_spike_matrix.v1"
    assert INDEX["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert INDEX["profile_count"] == len(INDEX["rows"]) == 8
    assert INDEX["compared_bf16_outputs_per_run"] == 32768
    assert {row["profile_name"] for row in INDEX["rows"]} == {
        "MxDim8AllGemminiRocketConfig", "MxDim32AllGemminiRocketConfig",
        "MxDim32GemminiRocketConfig", "MxE2M3OnlyGemminiRocketConfig",
        "MxE3M2OnlyGemminiRocketConfig", "MxE5M2OnlyGemminiRocketConfig",
        "TestMxGemminiRocketConfig", "TestRequantizerLutMxGemminiRocketConfig",
    }
    assert {row["compiler_revision"] for row in INDEX["rows"]} == {
        "3e96fff6b478b2301b55724e2ed6aad5936c980c"}
    assert {row["model2mlir_revision"] for row in INDEX["rows"]} == {
        "e9ded36eb85abf2d9097ac4dc11457c825853388"}


def test_archived_programs_and_both_spike_runs_match_source_goldens() -> None:
    for row in INDEX["rows"]:
        profile = load_profile(PROFILES / f"{row['profile_name']}.json")
        assert profile["chipyard_config"] is not None
        assert profile["geometry"]["mesh_rows"] == row["mesh_dim"]
        assert profile_sha256(profile) == row["profile_sha256"]
        folder = EVIDENCE / row["slug"]
        receipts = [json.loads((folder / f"receipt_{label}.json").read_text())
                    for label in ("first", "repro")]
        for label, receipt in zip(("first", "repro"), receipts):
            assert receipt["status"] == "source_golden_matched_on_pinned_spike"
            assert receipt["spike_exit_code"] == 0
            assert receipt["compared_bf16_outputs"] == 4096
            assert receipt["profile_name"] == row["profile_name"]
            assert receipt["profile_sha256"] == row["profile_sha256"]
            assert receipt.get("source_driver_sha256") == row["source_driver_sha256"]
            assert receipt.get("source_generation_manifest_sha256") == row[
                "source_generation_manifest_sha256"]
            assert receipt["source_header_sha256"] == row["source_header_sha256"]
            assert receipt["elf_sha256"] == row["artifact_sha256"][
                "physical/asymmetric_program.elf"]
            assert receipt["physical_program_sha256"] == row["artifact_sha256"][
                "physical/physical_program.json"]
            log = (folder / f"spike_{label}.log").read_bytes()
            assert _sha(log) == receipt["spike_log_sha256"] == row[
                "artifact_sha256"]["physical/spike.log"]
            assert b"0 BF16 mismatches" in log
        for relative, digest in row["artifact_sha256"].items():
            if relative == "physical/spike.log":
                continue
            compressed = folder / (relative.replace("/", "__") + ".gz")
            assert _sha(gzip.decompress(compressed.read_bytes())) == digest


def test_baseline_comparison_tracks_generated_outputs_across_tool_commits() -> None:
    from copy import deepcopy

    reproduced = json.loads((EVIDENCE / "reproduction_7809c82.json").read_text())
    assert _output_identity(reproduced) == _output_identity(INDEX)
    assert {row["compiler_revision"] for row in reproduced["rows"]} == {
        "7809c820e4590b5603e244359363e7c5834d271a"}
    assert {row["compiler_source_closure_sha256"] for row in reproduced["rows"]} != {
        row["compiler_source_closure_sha256"] for row in INDEX["rows"]}
    later = deepcopy(INDEX)
    later["rows"][0]["compiler_revision"] = "f" * 40
    later["rows"][0]["compiler_source_closure_sha256"] = "f" * 64
    assert _output_identity(later) == _output_identity(INDEX)
    later["rows"][0]["artifact_sha256"]["physical/asymmetric_program.elf"] = "0" * 64
    assert _output_identity(later) != _output_identity(INDEX)
