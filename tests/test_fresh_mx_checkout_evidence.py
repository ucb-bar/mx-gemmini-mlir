"""Check the fresh-clone build and source-roster reproduction receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.qualify_mx_rocket_wrapper_matrix import _output_identity


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/mx_fresh_checkout_205e178"
ROSTER_BASELINE = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22"
WRAPPER_BASELINE = ROOT / "docs/evidence/nicolas_rocket_wrapper_matrix_266c593/index.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fresh_clone_rebuilt_dialect_and_reproduced_source_outputs() -> None:
    index = _read(EVIDENCE / "index.json")
    assert index["schema"] == "mx_gemmini.fresh_checkout_reproduction.v1"
    assert index["status"] == "fresh_checkout_built_and_source_rosters_reproduced"
    assert index["compiler_revision"] == "205e178b26b4dec69f29aa50f727f8788af1fffd"
    assert index["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["radiance_source_revision"] == "80f84caedbabc663a7433c1da4455b936cca41f3"
    assert len(index["mx_opt_sha256"]) == 64
    assert (index["roster_drivers"], index["rocket_wrappers"]) == (31, 8)
    for relative, digest in index["files_sha256"].items():
        assert _sha(EVIDENCE / relative) == digest

    receipt = _read(EVIDENCE / "radiance_roster_reproduction.json")
    frontend = _read(EVIDENCE / "radiance_frontend_index.json")
    spike = _read(EVIDENCE / "radiance_spike_index.json")
    old_frontend = _read(ROSTER_BASELINE / "frontend/index.json")
    old_spike = _read(ROSTER_BASELINE / "spike/index.json")
    assert receipt["status"] == "all_31_source_goldens_reproduced_on_pinned_spike"
    assert receipt["compiler_revision"] == index["compiler_revision"]
    assert (receipt["covered_drivers"], receipt["fullout_drivers"],
            receipt["requant_drivers"]) == (31, 23, 8)
    assert receipt["frontend_index_sha256"] == _sha(
        EVIDENCE / "radiance_frontend_index.json")
    assert receipt["spike_index_sha256"] == _sha(EVIDENCE / "radiance_spike_index.json")
    assert receipt["baseline_frontend_index_sha256"] == _sha(
        ROSTER_BASELINE / "frontend/index.json")
    assert receipt["baseline_spike_index_sha256"] == _sha(
        ROSTER_BASELINE / "spike/index.json")
    assert len(frontend["rows"]) == len(spike["rows"]) == 31
    for actual, baseline in zip(frontend["rows"], old_frontend["rows"]):
        for key in ("driver", "driver_sha256", "header_sha256", "shape_mnk",
                    "tile_mnk", "precision", "quant_output", "profile_sha256",
                    "source_mlir_sha256", "bound_mlir_sha256"):
            assert actual[key] == baseline[key]
    for actual, baseline in zip(spike["rows"], old_spike["rows"]):
        for key in ("driver", "source_driver_sha256", "source_header_sha256",
                    "profile_bound_mlir_sha256", "payload_bound_mlir_sha256",
                    "elf_sha256", "spike_log_sha256", "comparison",
                    "compared_count", "status"):
            assert actual[key] == baseline[key]

    wrappers = _read(EVIDENCE / "rocket_wrapper_index.json")
    assert len(wrappers["rows"]) == 8
    assert {row["compiler_revision"] for row in wrappers["rows"]} == {
        index["compiler_revision"]}
    assert _output_identity(wrappers) == _output_identity(_read(WRAPPER_BASELINE))
