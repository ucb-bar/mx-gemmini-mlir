"""Check the current ucb-bar Radiance MX GEMM roster on Nicolas Spike."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_mx_gemm_upstream_82be2c7_20261010"
FRESH = ROOT / "docs/evidence/radiance_mx_gemm_fresh_4fc4d3a_82be2c7"
BASELINE = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_upstream_radiance_reproduces_all_31_mx_gemms() -> None:
    index = _read(EVIDENCE / "index.json")
    report = _read(EVIDENCE / "reproduction.json")
    front = _read(EVIDENCE / "frontend_index.json")
    spike = _read(EVIDENCE / "spike_index.json")
    headers = _read(EVIDENCE / "headers.json")
    old_front = _read(BASELINE / "frontend/index.json")
    old_spike = _read(BASELINE / "spike/index.json")
    assert index["schema"] == "mx_gemmini.upstream_radiance_mx_gemm_reproduction.v1"
    assert index["source_revision"] == report["source_revision"] == (
        "82be2c71e5a0ca5d764e08e1b2c6a0e638bff401")
    assert index["model2mlir_revision"] == report["model2mlir_revision"] == (
        "e9ded36eb85abf2d9097ac4dc11457c825853388")
    assert index["rtl_revision"] == report["rtl_revision"] == (
        "266c593f2cb51d7e3fe83fc0317072b585ac3c52")
    assert index["compiler_revision"] == report["compiler_revision"] == (
        "73c652562b238a3d97b2f826fa4d0a251c9821bb")
    assert report["status"] == "all_31_source_goldens_reproduced_on_pinned_spike"
    assert report["compatible_source_revision"] is True
    assert report["baseline_frontend_index_sha256"] == _sha(BASELINE / "frontend/index.json")
    assert report["baseline_spike_index_sha256"] == _sha(BASELINE / "spike/index.json")
    assert report["frontend_index_sha256"] == _sha(EVIDENCE / "frontend_index.json")
    assert report["spike_index_sha256"] == _sha(EVIDENCE / "spike_index.json")
    assert (index["covered_drivers"], report["covered_drivers"],
            front["captured_drivers"], spike["covered_drivers"],
            headers["drivers"]) == (31, 31, 31, 31, 31)
    assert (report["fullout_drivers"], report["requant_drivers"]) == (23, 8)
    assert spike["frontend_index_sha256"] == _sha(EVIDENCE / "frontend_index.json")
    for name, digest in index["files_sha256"].items():
        assert _sha(EVIDENCE / name) == digest
    for actual, previous, header in zip(front["rows"], old_front["rows"],
                                        headers["rows"], strict=True):
        for key in ("driver", "driver_sha256", "header_sha256", "shape_mnk",
                    "tile_mnk", "precision", "quant_output", "profile_sha256",
                    "source_mlir_sha256", "bound_mlir_sha256"):
            assert actual[key] == previous[key], (actual["driver"], key)
        assert actual["driver"] == header["driver"]
        assert actual["driver_sha256"] == header["driver_sha256"]
        assert actual["header_sha256"] == header["header_sha256"]
    for actual, previous in zip(spike["rows"], old_spike["rows"], strict=True):
        for key in ("driver", "source_driver_sha256", "source_header_sha256",
                    "profile_bound_mlir_sha256", "payload_bound_mlir_sha256",
                    "elf_sha256", "spike_log_sha256", "comparison",
                    "compared_count", "status"):
            assert actual[key] == previous[key], (actual["driver"], key)
        assert actual["status"].endswith("_matched_on_pinned_spike")


def test_latest_published_compiler_fresh_checkout_reproduces_upstream_roster() -> None:
    index = _read(FRESH / "index.json")
    report = _read(FRESH / "reproduction.json")
    front = _read(FRESH / "frontend_index.json")
    spike = _read(FRESH / "spike_index.json")
    headers = _read(FRESH / "headers.json")
    previous_front = _read(EVIDENCE / "frontend_index.json")
    previous_spike = _read(EVIDENCE / "spike_index.json")
    assert index["schema"] == "mx_gemmini.radiance_mx_gemm_fresh_checkout_upstream.v1"
    assert index["status"] == report["status"] == (
        "all_31_source_goldens_reproduced_on_pinned_spike")
    assert index["compiler_revision"] == report["compiler_revision"] == (
        "4fc4d3a50ea6d1483f5d80b43e65219dc1fe5497")
    assert index["radiance_source_revision"] == report["source_revision"] == (
        "82be2c71e5a0ca5d764e08e1b2c6a0e638bff401")
    assert (index["driver_count"], index["fullout_drivers"],
            index["requant_drivers"], index["total_compared_elements"]) == (
            31, 23, 8, 466944)
    assert (front["captured_drivers"], spike["covered_drivers"],
            headers["drivers"]) == (31, 31, 31)
    assert report["compatible_source_revision"] is True
    assert report["frontend_index_sha256"] == _sha(FRESH / "frontend_index.json")
    assert report["spike_index_sha256"] == _sha(FRESH / "spike_index.json")
    for name, digest in index["files_sha256"].items():
        assert _sha(FRESH / name) == digest
    for actual, previous in zip(front["rows"], previous_front["rows"], strict=True):
        for key in ("driver", "driver_sha256", "header_sha256", "shape_mnk",
                    "tile_mnk", "precision", "quant_output", "profile_sha256",
                    "source_mlir_sha256", "bound_mlir_sha256"):
            assert actual[key] == previous[key], (actual["driver"], key)
    for actual, previous in zip(spike["rows"], previous_spike["rows"], strict=True):
        for key in ("driver", "source_driver_sha256", "source_header_sha256",
                    "profile_bound_mlir_sha256", "payload_bound_mlir_sha256",
                    "elf_sha256", "spike_log_sha256", "comparison",
                    "compared_count", "status"):
            assert actual[key] == previous[key], (actual["driver"], key)
    assert sum(row["compared_count"] for row in spike["rows"]) == 466944
