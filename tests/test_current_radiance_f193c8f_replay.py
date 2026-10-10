"""Audit the current compiler's complete Radiance GEMM source replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_mx_gemm_f193c8f_80f84ca"
BASELINE = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22"
COMPILER = "f193c8ff0c6bd9165b08359841e945c3b4282261"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_current_radiance_roster_replays_all_31_outputs() -> None:
    reproduction = _json(EVIDENCE / "reproduction.json")
    frontend = _json(EVIDENCE / "frontend/index.json")
    spike = _json(EVIDENCE / "spike/index.json")
    old_frontend = _json(BASELINE / "frontend/index.json")
    old_spike = _json(BASELINE / "spike/index.json")
    assert reproduction["status"] == "all_31_source_goldens_reproduced_on_pinned_spike"
    assert reproduction["compiler_revision"] == frontend["compiler_revision"] == (
        spike["compiler_revision"])
    assert spike["compiler_revision"] == COMPILER
    assert reproduction["source_revision"] == frontend["source_revision"] == (
        spike["source_revision"])
    assert spike["source_revision"] == "80f84caedbabc663a7433c1da4455b936cca41f3"
    assert reproduction["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert reproduction["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert reproduction["frontend_index_sha256"] == spike["frontend_index_sha256"] == _sha(
        EVIDENCE / "frontend/index.json")
    assert reproduction["spike_index_sha256"] == _sha(EVIDENCE / "spike/index.json")
    assert reproduction["baseline_frontend_index_sha256"] == _sha(
        BASELINE / "frontend/index.json")
    assert reproduction["baseline_spike_index_sha256"] == _sha(
        BASELINE / "spike/index.json")
    assert frontend["captured_drivers"] == spike["covered_drivers"] == 31
    assert frontend["fullout_drivers"] == spike["fullout_drivers"] == 23
    assert frontend["requant_drivers"] == spike["requant_drivers"] == 8
    assert len(frontend["rows"]) == len(spike["rows"]) == 31
    assert _json(EVIDENCE / "headers.json")["drivers"] == 31

    for row, old in zip(frontend["rows"], old_frontend["rows"]):
        assert row["driver"] == old["driver"]
        for key in ("driver_sha256", "header_sha256", "shape_mnk", "tile_mnk",
                    "precision", "quant_output", "profile_sha256",
                    "source_mlir_sha256", "bound_mlir_sha256"):
            assert row[key] == old[key]
        receipt = EVIDENCE / "frontend" / row["receipt"]
        assert row["receipt_sha256"] == _sha(receipt)

    total_outputs = total_scales = 0
    for row, old in zip(spike["rows"], old_spike["rows"]):
        assert row["driver"] == old["driver"]
        for key in ("source_driver_sha256", "source_header_sha256",
                    "profile_bound_mlir_sha256", "payload_bound_mlir_sha256",
                    "elf_sha256", "spike_log_sha256", "comparison",
                    "compared_count", "status"):
            assert row[key] == old[key]
        receipt_path = EVIDENCE / "spike" / row["receipt"]
        receipt = _json(receipt_path)
        log_path = receipt_path.parent / "spike.log"
        assert row["receipt_sha256"] == _sha(receipt_path)
        assert row["spike_log_sha256"] == receipt["spike_log_sha256"] == _sha(log_path)
        assert row["elf_sha256"] == receipt["elf_sha256"]
        assert receipt["compiler_revision"] == COMPILER
        assert receipt["spike_exit_code"] == 0
        assert receipt["status"] == row["status"]
        assert receipt[row["comparison"]] == row["compared_count"]
        total_outputs += row["compared_count"]
        total_scales += row.get("compared_source_e8m0_scales", 0)
    assert total_outputs == 466944
    assert total_scales == 3328
