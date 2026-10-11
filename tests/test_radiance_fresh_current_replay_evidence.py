"""The current published compiler must reproduce the complete source roster."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/radiance_mx_gemm_fresh_2b55be0_82be2c7"
BASELINE = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fresh_current_checkout_reproduced_all_source_outputs() -> None:
    index = _read(ARCHIVE / "index.json")
    receipt = _read(ARCHIVE / "reproduction.json")
    headers = _read(ARCHIVE / "headers.json")
    frontend = _read(ARCHIVE / "frontend/index.json")
    spike = _read(ARCHIVE / "spike/index.json")
    old_frontend = _read(BASELINE / "frontend/index.json")
    old_spike = _read(BASELINE / "spike/index.json")
    assert index["schema"] == "mx_gemmini.fresh_current_readonly_roster.v1"
    assert index["status"] == "fresh_checkout_all_31_source_goldens_reproduced_on_pinned_spike"
    assert index["compiler_revision"] == "2b55be0a26af24721b76a23b1396fbe6ef9f5cde"
    assert index["fresh_task"] == "mx-current-fresh-replay"
    assert len(index["mx_opt_sha256"]) == 64
    assert index["source_status_before_sha256"] == index["source_status_after_sha256"]
    for name, digest in index["files_sha256"].items():
        assert _sha(ARCHIVE / name) == digest
    assert receipt["status"] == "all_31_source_goldens_reproduced_on_pinned_spike"
    assert receipt["compiler_revision"] == index["compiler_revision"]
    assert receipt["read_only_source"] is True
    assert (receipt["covered_drivers"], receipt["fullout_drivers"],
            receipt["requant_drivers"]) == (31, 23, 8)
    assert receipt["frontend_index_sha256"] == _sha(ARCHIVE / "frontend/index.json")
    assert receipt["spike_index_sha256"] == _sha(ARCHIVE / "spike/index.json")
    assert receipt["baseline_frontend_index_sha256"] == _sha(BASELINE / "frontend/index.json")
    assert receipt["baseline_spike_index_sha256"] == _sha(BASELINE / "spike/index.json")
    assert headers["drivers"] == 31 and headers["generated_headers"] == []
    assert len(frontend["rows"]) == len(spike["rows"]) == 31
    assert frontend["compiler_revision"] == spike["compiler_revision"] == index["compiler_revision"]
    assert spike["frontend_index_sha256"] == _sha(ARCHIVE / "frontend/index.json")
    for actual, old in zip(frontend["rows"], old_frontend["rows"]):
        assert actual["driver"] == old["driver"]
        for key in ("driver_sha256", "header_sha256", "shape_mnk", "tile_mnk",
                    "precision", "quant_output", "profile_sha256",
                    "source_mlir_sha256", "bound_mlir_sha256"):
            assert actual[key] == old[key]
    for actual, old in zip(spike["rows"], old_spike["rows"]):
        assert actual["driver"] == old["driver"]
        for key in ("source_driver_sha256", "source_header_sha256",
                    "profile_bound_mlir_sha256", "payload_bound_mlir_sha256",
                    "elf_sha256", "spike_log_sha256", "comparison",
                    "compared_count", "status"):
            assert actual[key] == old[key]
