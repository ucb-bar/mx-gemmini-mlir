"""Check direct Spike evidence for selected legal modes in six MX profiles."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.bind_payload import bind_payload
from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_selected_mx_profiles_266c593"
FRONTEND = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22/frontend"
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(root: Path, name: str) -> bytes:
    path = root / name
    return path.read_bytes() if path.exists() else gzip.decompress(
        (root / f"{name}.gz").read_bytes())


MATRIX = json.loads((EVIDENCE / "index.json").read_text())
CASES = [(entry["profile_name"], precision.lower())
         for entry in MATRIX["profiles"] for precision in entry["cases"]]


def test_selected_mx_profile_matrix_is_pinned_and_reproduced():
    assert MATRIX["schema"] == "mx_gemmini.radiance_selected_mx_profiles_spike_matrix.v1"
    assert MATRIX["status"] == "six_selected_profiles_matched_on_pinned_spike"
    assert MATRIX["compiler_revision"] == "82a8bae7ef0413b1978c95fc52bab1f18260cf66"
    assert MATRIX["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert MATRIX["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert MATRIX["profile_count"] == 6
    assert MATRIX["case_count"] == len(CASES) == 12
    assert MATRIX["compared_bf16_outputs"] == 147456
    assert sum(row["compared_bf16_outputs"] for row in MATRIX["profiles"]) == 147456
    for entry in MATRIX["profiles"]:
        root = EVIDENCE / entry["profile_name"]
        first = (root / "qualification.json").read_bytes()
        second = (root / "qualification_repro.json").read_bytes()
        assert _sha(first) == entry["qualification_sha256"]
        assert _sha(second) == entry["qualification_repro_sha256"]
        a, b = json.loads(first), json.loads(second)
        assert a["profile_name"] == b["profile_name"] == entry["profile_name"]
        assert a["profile_sha256"] == b["profile_sha256"] == entry["profile_sha256"]
        assert a["compiler_revision"] == b["compiler_revision"] == MATRIX[
            "compiler_revision"]
        x, y = deepcopy(a), deepcopy(b)
        for summary in (x, y):
            for row in summary["rows"]:
                row.pop("artifact_manifest_sha256")
        assert x == y


@pytest.mark.parametrize("profile_name,precision", CASES)
def test_selected_profile_case_rederives_commands_and_source_output(
        profile_name: str, precision: str, tmp_path):
    root = EVIDENCE / profile_name / precision
    profile = load_profile(PROFILES / f"{profile_name}.json")
    summary = json.loads((EVIDENCE / profile_name / "qualification.json").read_text())
    row = next(row for row in summary["rows"]
               if row["precision"].lower() == precision)
    assert summary["profile_sha256"] == profile_sha256(profile)
    stem = Path(row["driver"]).stem
    handoff = (FRONTEND / stem / "mx_gemm.handoff.mlir").read_text()
    selected = bind_handoff(handoff, profile)
    assert selected == _read(root, "profile_bound.mlir").decode()
    assert row["profile_bound_mlir_sha256"] == _sha(selected.encode())
    manifest, resources = load_bundle(root / "bundle")
    bound = bind_payload(selected, profile, manifest)
    assert bound == _read(root, "payload_bound.mlir").decode()
    assert row["payload_bound_mlir_sha256"] == _sha(bound.encode())
    assert row["bundle_manifest_sha256"] == _sha(_read(root, "bundle/manifest.json"))
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.source_golden_preserving
    assert program.receipt() == json.loads(_read(root, "build/physical_program.json"))
    assert row["physical_program_sha256"] == _sha(
        _read(root, "build/physical_program.json"))
    emitted = write_standalone_sources(tmp_path / "emitted", program, resources)
    for name in ("mx_issue.c", "mx_driver.c", "physical_program.json"):
        assert emitted["files_sha256"][name] == _sha(_read(root, f"build/{name}"))
    assert row["generated_issue_sha256"] == _sha(_read(root, "build/mx_issue.c"))
    assert row["elf_sha256"] == _sha(_read(root, "build/mx_program.elf"))
    assert row["spike_log_sha256"] == _sha(_read(root, "build/spike.log"))
    assert row["status"] == "source_golden_matched_on_pinned_spike"
    assert "0 BF16 mismatches" in _read(root, "build/spike.log").decode()
    receipts = [_read(root, name) for name in (
        "build/artifact_manifest.json", "build/artifact_manifest_repro.json")]
    second = json.loads((EVIDENCE / profile_name / "qualification_repro.json").read_text())
    repro_row = next(item for item in second["rows"]
                     if item["precision"].lower() == precision)
    assert _sha(receipts[0]) == row["artifact_manifest_sha256"]
    assert _sha(receipts[1]) == repro_row["artifact_manifest_sha256"]
    a, b = (json.loads(data) for data in receipts)
    for artifact in (a, b):
        assert artifact["status"] == "source_golden_matched_on_pinned_spike"
        assert artifact["spike_exit_code"] == 0
        assert artifact["compared_bf16_outputs"] == row["compared_bf16_outputs"]
        assert artifact["elf_sha256"] == row["elf_sha256"]
        assert artifact["profile_sha256"] == summary["profile_sha256"]
    a.pop("build_log_sha256")
    b.pop("build_log_sha256")
    assert a == b


def test_fp4_only_profile_rejects_fp8_capture():
    profile = load_profile(PROFILES / "MxFp4OnlyGemminiRocketConfig.json")
    stem = "mxgemm.fp8.m128n128k512.tm128tn128tk128.fullout"
    with pytest.raises(ValueError):
        bind_handoff((FRONTEND / stem / "mx_gemm.handoff.mlir").read_text(), profile)
