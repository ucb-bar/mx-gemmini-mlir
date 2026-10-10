"""Audit current Radiance source against the complete pinned MX Spike roster."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.compile_mx import _source_closure


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_mx_gemm_80f84ca_upstream_repro_20261010"
BASELINE = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_current_radiance_roster_reproduces_all_31_pinned_spike_artifacts():
    archive = _read(EVIDENCE / "index.json")
    assert archive["schema"] == "mx_gemmini.current_radiance_roster_reproduction_archive.v1"
    assert archive["baseline"] == str(BASELINE.relative_to(ROOT))
    for name, digest in archive["files_sha256"].items():
        assert _sha(EVIDENCE / name) == digest

    report = _read(EVIDENCE / "reproduction.json")
    front = _read(EVIDENCE / "frontend/index.json")
    spike = _read(EVIDENCE / "spike/index.json")
    old_front = _read(BASELINE / "frontend/index.json")
    old_spike = _read(BASELINE / "spike/index.json")
    headers = _read(EVIDENCE / "headers.json")
    assert report["status"] == "all_31_source_goldens_reproduced_on_pinned_spike"
    assert report["compatible_source_revision"] is True
    assert report["source_revision"] == front["source_revision"] == (
        "80f84caedbabc663a7433c1da4455b936cca41f3")
    assert report["baseline_source_revision"] == old_front["source_revision"] == (
        "ee22e0b87180436cd0fa1583c411a3cf49d7586a")
    assert report["model2mlir_revision"] == front["model2mlir_revision"] == (
        "e9ded36eb85abf2d9097ac4dc11457c825853388")
    assert report["compiler_revision"] == (
        "f5829b0db54a19b6a456f11868e44c6e907ab662")
    assert report["rtl_revision"] == front["rtl_revision"] == spike["rtl_revision"] == (
        "266c593f2cb51d7e3fe83fc0317072b585ac3c52")
    assert report["baseline_frontend_index_sha256"] == _sha(BASELINE / "frontend/index.json")
    assert report["baseline_spike_index_sha256"] == _sha(BASELINE / "spike/index.json")
    assert report["frontend_index_sha256"] == _sha(EVIDENCE / "frontend/index.json")
    assert report["spike_index_sha256"] == _sha(EVIDENCE / "spike/index.json")
    assert report["covered_drivers"] == front["captured_drivers"] == spike["covered_drivers"] == 31
    assert report["fullout_drivers"] == front["fullout_drivers"] == spike["fullout_drivers"] == 23
    assert report["requant_drivers"] == front["requant_drivers"] == spike["requant_drivers"] == 8
    assert headers["drivers"] == 31
    assert spike["frontend_index_sha256"] == _sha(EVIDENCE / "frontend/index.json")
    source_closure = _source_closure(
        ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
        sorted((ROOT / "tools").glob("*.py")))

    for new_f, old_f, new_s, old_s, header in zip(
            front["rows"], old_front["rows"], spike["rows"], old_spike["rows"],
            headers["rows"], strict=True):
        name = new_f["driver"]
        assert name == old_f["driver"] == new_s["driver"] == old_s["driver"] == header["driver"]
        for key in ("driver_sha256", "header_sha256", "shape_mnk", "tile_mnk",
                    "precision", "quant_output", "profile_sha256",
                    "source_mlir_sha256", "bound_mlir_sha256"):
            assert new_f[key] == old_f[key], (name, key)
        assert header["driver_sha256"] == new_f["driver_sha256"]
        assert header["header_sha256"] == new_f["header_sha256"]
        current_capture = EVIDENCE / "frontend" / new_f["receipt"]
        baseline_capture = BASELINE / "frontend" / old_f["receipt"]
        assert new_f["receipt_sha256"] == _sha(current_capture)
        current = _read(current_capture)
        baseline = _read(baseline_capture)
        for key in ("source_generator_sha256", "source_layout", "source_plan_error"):
            assert current[key] == baseline[key], (name, key)
        old_case = BASELINE / "frontend" / Path(name).stem
        assert _sha(old_case / "mx_gemm.model2mlir.mlir") == new_f["source_mlir_sha256"]
        assert _sha(old_case / "mx_gemm.profile_bound.mlir") == new_f["bound_mlir_sha256"]

        for key in ("source_driver_sha256", "source_header_sha256",
                    "profile_bound_mlir_sha256", "payload_bound_mlir_sha256",
                    "elf_sha256", "spike_log_sha256", "comparison",
                    "compared_count", "status"):
            assert new_s[key] == old_s[key], (name, key)
        assert new_s["frontend_receipt_sha256"] == new_f["receipt_sha256"]
        current_manifest = EVIDENCE / "spike" / new_s["receipt"]
        baseline_manifest = BASELINE / "spike" / old_s["receipt"]
        assert new_s["receipt_sha256"] == _sha(current_manifest)
        compiled = _read(current_manifest)
        previous = _read(baseline_manifest)
        assert compiled["compiler_revision"] == report["compiler_revision"]
        assert compiled["compiler_source_closure_sha256"] == source_closure
        for key in ("files_sha256", "object_sha256", "extension_sha256",
                    "elf_sha256", "spike_log_sha256", "bound_mlir_sha256"):
            assert compiled[key] == previous[key], (name, key)
        assert compiled["spike_exit_code"] == 0
        assert compiled["status"] == new_s["status"]
        old_build = BASELINE / "spike" / Path(name).stem / "build"
        assert _sha(old_build / "spike.log") == compiled["spike_log_sha256"]
        assert _sha(old_build / "physical_program.json") == (
            compiled["files_sha256"]["physical_program.json"])
