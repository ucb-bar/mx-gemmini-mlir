"""Keep the current-source 31-driver Spike replay tied to its published checkout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/radiance_mx_gemm_current_9b3a6b9_80f84ca"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_current_radiance_roster_has_identical_executable_artifacts():
    index = _read(ARCHIVE / "index.json")
    assert index["status"] == "all_31_source_goldens_reproduced_on_pinned_spike"
    assert index["drivers"] == 31
    assert (index["fullout_drivers"], index["requant_drivers"]) == (23, 8)
    assert index["fresh_checkout_artifacts_matched"] is True
    assert index["source_revision"] == "80f84caedbabc663a7433c1da4455b936cca41f3"
    assert index["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert index["compared_codes_or_bf16_values"] == 466944
    assert index["compared_e8m0_scales"] == 3328
    for filename, descriptor in index["files"].items():
        path = ARCHIVE / filename
        assert path.stat().st_size == descriptor["bytes"]
        assert _sha(path) == descriptor["sha256"]
    front = [_read(ARCHIVE / name / "frontend/index.json") for name in ("local", "fresh")]
    spike = [_read(ARCHIVE / name / "spike/index.json") for name in ("local", "fresh")]
    assert front[0] == front[1]
    assert all(item["captured_drivers"] == 31 for item in front)
    assert all(item["covered_drivers"] == 31 for item in spike)
    assert all(item["compiler_revision"] == index["compiler_revision"] for item in spike)
    for archived, first, second in zip(index["rows"], spike[0]["rows"], spike[1]["rows"]):
        assert archived["driver"] == first["driver"] == second["driver"]
        assert archived["artifact_agreement_except_build_logs"] is True
        assert first["status"] == second["status"]
        for key in ("payload_bound_mlir_sha256", "elf_sha256", "spike_log_sha256",
                    "comparison", "compared_count"):
            assert first[key] == second[key]
        assert archived["spike_log_sha256"] == first["spike_log_sha256"]
        stem = Path(archived["driver"]).stem
        receipts = [_read(ARCHIVE / name / "receipts" / f"{stem}.json")
                    for name in ("local", "fresh")]
        assert {key for key in receipts[0] if receipts[0][key] != receipts[1][key]} == {
            "build_log_sha256"}
        for key in ("files_sha256", "object_sha256", "extension_sha256",
                    "elf_sha256", "spike_log_sha256", "bound_mlir_sha256"):
            assert receipts[0][key] == receipts[1][key]
        assert _sha(ARCHIVE / "spike_logs" / f"{stem}.log") == (
            archived["spike_log_sha256"])
