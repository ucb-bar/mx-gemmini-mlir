"""Recheck the plain Rocket MX profile across the three Radiance precisions."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_plain_mx_profile_trio_266c593"
FRONTEND_INDEX = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22/frontend/index.json"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"
COUNTS = {"fp8": 16384, "fp4": 4096, "fp6": 16384}


def _read(root: Path, name: str) -> bytes:
    path = root / name
    return path.read_bytes() if path.exists() else gzip.decompress(
        (root / f"{name}.gz").read_bytes())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_three_precision_summary_and_fresh_checkout():
    first, repro, fresh = [json.loads((EVIDENCE / name).read_text()) for name in
                           ("index.json", "index_repro.json", "index_fresh.json")]
    assert first["schema"] == "mx_gemmini.radiance_plain_mx_profile_source_ladder.v1"
    assert first["status"] == "three_source_precisions_matched_on_pinned_spike"
    assert first["compiler_revision"] == "48fbb894b34dba95b1e04cccb18385632a8a7f0e"
    assert first["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert first["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert first["frontend_index_sha256"] == _sha(FRONTEND_INDEX.read_bytes())
    assert first["profile_sha256"] == profile_sha256(load_profile(PROFILE))
    assert first["compared_bf16_outputs"] == sum(COUNTS.values()) == 36864
    assert {row["precision"].lower() for row in first["rows"]} == set(COUNTS)
    materialization = json.loads((EVIDENCE / "fresh_materialization.json").read_text())
    assert materialization["drivers"] == 31
    assert any(path.endswith("mxgemm.data.fp4.m64n64k128.h")
               for path in materialization["generated_headers"])

    for other in (repro, fresh):
        a, b = deepcopy(first), deepcopy(other)
        for index in (a, b):
            for row in index["rows"]:
                row.pop("artifact_manifest_sha256")
        assert a == b


@pytest.mark.parametrize("precision", ("fp8", "fp4", "fp6"))
def test_source_capture_physical_program_and_spike_receipts(precision: str, tmp_path):
    root = EVIDENCE / precision
    index = json.loads((EVIDENCE / "index.json").read_text())
    row = next(row for row in index["rows"] if row["precision"].lower() == precision)
    frontend = json.loads(_read(root, "frontend_receipt.json"))
    assert row["driver"] == frontend["source_driver"]
    assert row["driver_sha256"] == frontend["source_driver_sha256"]
    assert row["header_sha256"] == frontend["source_data_header_sha256"]
    assert row["handoff_mlir_sha256"] == _sha(_read(root, "handoff.mlir"))
    assert frontend["source_mlir_sha256"] == _sha(_read(root, "frontend.mlir"))
    profile = load_profile(PROFILE)
    selected = bind_handoff(_read(root, "handoff.mlir").decode(), profile)
    assert selected == _read(root, "profile_bound.mlir").decode()
    assert row["profile_bound_mlir_sha256"] == _sha(selected.encode())
    manifest, resources = load_bundle(root / "bundle")
    bound = _read(root, "payload_bound.mlir").decode()
    assert row["payload_bound_mlir_sha256"] == _sha(bound.encode())
    assert row["bundle_manifest_sha256"] == _sha(_read(root, "bundle/manifest.json"))
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.source_golden_preserving
    assert program.derived_expected_bf16 is None
    assert program.receipt() == json.loads(_read(root, "build/physical_program.json"))
    assert row["physical_program_sha256"] == _sha(
        _read(root, "build/physical_program.json"))
    emitted = write_standalone_sources(tmp_path / precision, program, resources)
    for name in ("mx_issue.c", "mx_driver.c", "mx_data.S", "physical_program.json"):
        assert emitted["files_sha256"][name] == _sha(_read(root, f"build/{name}"))
    assert row["generated_issue_sha256"] == _sha(_read(root, "build/mx_issue.c"))
    assert row["elf_sha256"] == _sha(_read(root, "build/mx_program.elf"))
    assert row["spike_log_sha256"] == _sha(_read(root, "build/spike.log"))
    assert row["compared_bf16_outputs"] == COUNTS[precision]
    assert row["status"] == "source_golden_matched_on_pinned_spike"
    assert "0 BF16 mismatches" in _read(root, "build/spike.log").decode()

    receipts = [_read(root, f"build/{name}") for name in (
        "artifact_manifest.json", "artifact_manifest_repro.json",
        "artifact_manifest_fresh.json")]
    summaries = [json.loads((EVIDENCE / name).read_text()) for name in (
        "index.json", "index_repro.json", "index_fresh.json")]
    for data, summary in zip(receipts, summaries):
        selected_row = next(item for item in summary["rows"]
                            if item["precision"].lower() == precision)
        assert _sha(data) == selected_row["artifact_manifest_sha256"]
    artifacts = [json.loads(data) for data in receipts]
    for artifact in artifacts:
        assert artifact["status"] == "source_golden_matched_on_pinned_spike"
        assert artifact["spike_exit_code"] == 0
        assert artifact["compared_bf16_outputs"] == COUNTS[precision]
        assert artifact["elf_sha256"] == row["elf_sha256"]
        assert artifact["profile_sha256"] == index["profile_sha256"]
    stripped = []
    for artifact in artifacts:
        stable = dict(artifact)
        stable.pop("build_log_sha256")
        stripped.append(stable)
    assert stripped[0] == stripped[1] == stripped[2]
