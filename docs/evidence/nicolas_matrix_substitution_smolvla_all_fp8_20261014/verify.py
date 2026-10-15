"""Verify the archived equal-output all-FP8 SmolVLA issuer comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from tools.compare_linked_session_cycles import compare_sessions  # noqa: E402


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify() -> None:
    manifest = json.loads((HERE / "archive_manifest.json").read_text())
    assert manifest["schema"] == "mx_gemmini.smolvla_all_fp8_issuer_archive.v1"
    for name, digest in manifest["files"].items():
        assert sha(HERE / name) == digest, name

    comparison = json.loads((HERE / "comparison.json").read_text())
    recomputed = compare_sessions(
        HERE / "compiler_link.json",
        HERE / "compiler/mx_session_run.json",
        HERE / "nicolas_link.json",
        HERE / "nicolas/mx_session_run.json",
    )
    assert comparison == recomputed
    assert comparison["output_digest_parity"] is True
    assert comparison["changed_stage_objects"] == [
        "flow_denoise/device_catalog/mx_catalog.o",
        "prefix_encode/device_catalog/mx_catalog.o",
    ]
    assert comparison["reference_cycles"] == 180_219_637_915
    assert comparison["candidate_cycles"] == 179_451_403_096
    assert [(row["name"], row["steps"]) for row in comparison["programs"]] == [
        ("prefix_encode", 1), ("flow_denoise", 10), ("action_decode", 1)
    ]

    baseline = HERE.parent / "smolvla_all_fp8_full_action_20261014"
    assert sha(HERE / "compiler_link.json") == sha(baseline / "link.json")
    assert sha(HERE / "compiler/mx_session_run.json") == sha(baseline / "run.json")
    print("12 steps, 33 output hashes, FP8 baseline, and issuer-only changes verified")


if __name__ == "__main__":
    verify()
