"""Current compiler replay must match all 31 archived source and Spike results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/radiance_mx_gemm_readonly_7e92ed2_82be2c7"
BASELINE = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_readonly_replay_matches_every_source_and_spike_case() -> None:
    index = _read(ARCHIVE / "index.json")
    receipt = _read(ARCHIVE / "reproduction.json")
    headers = _read(ARCHIVE / "headers.json")
    front = _read(ARCHIVE / "frontend/index.json")
    spike = _read(ARCHIVE / "spike/index.json")
    old_front = _read(BASELINE / "frontend/index.json")
    old_spike = _read(BASELINE / "spike/index.json")
    assert index["schema"] == "mx_gemmini.radiance_readonly_roster_archive.v1"
    assert index["status"] == "all_31_source_goldens_reproduced_without_source_writes"
    assert index["source_status_unchanged"] is True
    assert index["source_status_before_sha256"] == index["source_status_after_sha256"]
    assert len(index["archived_files_sha256"]) == 66
    for name, digest in index["archived_files_sha256"].items():
        assert _sha(ARCHIVE / name) == digest
    assert receipt["status"] == "all_31_source_goldens_reproduced_on_pinned_spike"
    assert receipt["read_only_source"] is True
    assert receipt["compatible_source_revision"] is True
    assert receipt["compiler_revision"] == index["compiler_revision"]
    assert receipt["source_revision"] == index["source_revision"]
    assert receipt["model2mlir_revision"] == index["model2mlir_revision"]
    assert receipt["rtl_revision"] == index["rtl_revision"]
    assert receipt["frontend_index_sha256"] == _sha(ARCHIVE / "frontend/index.json")
    assert receipt["spike_index_sha256"] == _sha(ARCHIVE / "spike/index.json")
    assert receipt["baseline_frontend_index_sha256"] == _sha(BASELINE / "frontend/index.json")
    assert receipt["baseline_spike_index_sha256"] == _sha(BASELINE / "spike/index.json")
    assert headers["drivers"] == 31 and headers["generated_headers"] == []
    assert len(headers["rows"]) == len(front["rows"]) == len(spike["rows"]) == 31
    assert (receipt["fullout_drivers"], receipt["requant_drivers"]) == (23, 8)
    assert spike["frontend_index_sha256"] == _sha(ARCHIVE / "frontend/index.json")

    for header, actual, old in zip(headers["rows"], front["rows"], old_front["rows"]):
        assert header["origin"] == "existing"
        assert header["driver"] == actual["driver"] == old["driver"]
        assert header["driver_sha256"] == actual["driver_sha256"]
        assert header["header_sha256"] == actual["header_sha256"]
        for key in ("driver_sha256", "header_sha256", "shape_mnk", "tile_mnk",
                    "precision", "quant_output", "profile_sha256",
                    "source_mlir_sha256", "bound_mlir_sha256"):
            assert actual[key] == old[key]
        new_capture = _read(ARCHIVE / "frontend" / actual["receipt"])
        old_capture = _read(BASELINE / "frontend" / old["receipt"])
        assert actual["receipt_sha256"] == _sha(ARCHIVE / "frontend" / actual["receipt"])
        for key in ("source_generator_sha256", "source_layout", "source_plan_error"):
            assert new_capture[key] == old_capture[key]

    for actual, old in zip(spike["rows"], old_spike["rows"]):
        assert actual["driver"] == old["driver"]
        for key in ("source_driver_sha256", "source_header_sha256",
                    "profile_bound_mlir_sha256", "payload_bound_mlir_sha256",
                    "elf_sha256", "spike_log_sha256", "comparison",
                    "compared_count", "status"):
            assert actual[key] == old[key]
        new_run = _read(ARCHIVE / "spike" / actual["receipt"])
        old_run = _read(BASELINE / "spike" / old["receipt"])
        assert actual["receipt_sha256"] == _sha(ARCHIVE / "spike" / actual["receipt"])
        assert new_run["spike_exit_code"] == 0
        assert new_run["compiler_revision"] == index["compiler_revision"]
        for key in ("files_sha256", "object_sha256", "extension_sha256",
                    "elf_sha256", "spike_log_sha256", "bound_mlir_sha256"):
            assert new_run[key] == old_run[key]
