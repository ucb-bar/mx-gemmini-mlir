"""Check the direct stock-Spike hardware requantizer receipts and artifacts."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_requantizer_wrapper import _identity


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_requantizer_wrapper_266c593"
INDEX = json.loads((EVIDENCE / "index.json").read_text())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_three_requantized_output_modes_use_the_named_profile() -> None:
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/"
               "TestRequantizerLutMxGemminiRocketConfig.json")
    assert INDEX["schema"] == "mx_gemmini.nicolas_requantizer_wrapper.v1"
    assert INDEX["profile_name"] == profile["name"]
    assert INDEX["profile_sha256"] == profile_sha256(profile)
    assert INDEX["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert INDEX["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert profile["resources"]["requantizer"] is True
    assert [(row["precision"], row["compared_codes_or_bytes"],
             row["compared_e8m0_scales"]) for row in INDEX["rows"]] == [
                 ("fp8", 4096, 128), ("fp4", 2048, 128), ("fp6", 8192, 512)]


def test_archived_mlir_commands_elf_and_spike_logs_match_receipts() -> None:
    formats = {"fp8": "fp8_e4m3", "fp4": "fp4_e2m1", "fp6": "fp6_e3m2"}
    for row in INDEX["rows"]:
        precision = row["precision"]
        folder = EVIDENCE / precision
        receipt = json.loads((folder / "artifact_manifest.json").read_text())
        artifacts = {}
        for relative, digest in row["artifact_sha256"].items():
            archived = folder / f"{relative.replace('/', '__')}.gz"
            data = gzip.decompress(archived.read_bytes())
            assert _sha(data) == digest
            artifacts[relative] = data
        assert row["status"] == receipt["status"] == (
            "nicolas_oracle_matched_on_pinned_spike")
        assert receipt["spike_exit_code"] == 0
        assert receipt["profile_sha256"] == INDEX["profile_sha256"]
        assert receipt["source_driver_sha256"] == row["source_driver_sha256"]
        assert receipt["source_header_sha256"] == row["source_header_sha256"]
        assert receipt["elf_sha256"] == row["artifact_sha256"]["build/mx_program.elf"]
        assert receipt["spike_log_sha256"] == row["artifact_sha256"]["build/spike.log"]
        assert receipt["golden_basis"] == row["golden_basis"]
        assert row["fp6_quantized_readout_derived_from_fullout"] == (precision == "fp6")
        assert b'"mx_gemmini.readout_quantized"' in artifacts["payload_bound.mlir"]
        assert b'"mx_gemmini.host_requantize"' not in artifacts["payload_bound.mlir"]
        program = json.loads(artifacts["build/physical_program.json"])
        assert program["profile_sha256"] == INDEX["profile_sha256"]
        assert program["output_format"] == formats[precision]
        assert program["mode"] == "spike_serial"
        functs = {step["command"].get("funct") for step in program["steps"]}
        assert {0, 2, 3, 8, 9, 24, 26, 27} <= functs
        log = artifacts["build/spike.log"]
        assert b"0 E8M0 scale mismatches" in log
        assert {"fp8": b"0 FP8 code mismatches",
                "fp4": b"0 FP4 packed-code mismatches",
                "fp6": b"0 FP6 packed-index mismatches"}[precision] in log


def test_baseline_identity_tracks_executable_output() -> None:
    reproduced = json.loads((EVIDENCE / "reproduction_50f168b.json").read_text())
    assert _identity(reproduced) == _identity(INDEX)
    changed = deepcopy(INDEX)
    changed["rows"][0]["compiler_revision"] = "f" * 40
    assert _identity(changed) == _identity(INDEX)
    changed["rows"][0]["artifact_sha256"]["build/mx_program.elf"] = "0" * 64
    assert _identity(changed) != _identity(INDEX)
