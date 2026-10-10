"""Keep every Radiance MX precision driver tied to a source-output Spike receipt."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROSTER = ROOT / "docs/evidence/radiance_mx_gemm_source_parity_266c593.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_radiance_mx_gemm_source_parity_roster():
    index = json.loads(ROSTER.read_text())
    assert index["schema"] == "mx_gemmini.radiance_mx_gemm_source_parity.v1"
    assert index["source_revision"] == "d4732fb41c2d55088050c0399a4c6243b53e5fbd"
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["model2mlir_revision"] == (
        "7485a829c0195af0ec42820837d609e62e466564")
    rows = index["rows"]
    assert (index["covered_drivers"], index["fullout_drivers"],
            index["requant_drivers"]) == (31, 23, 8)
    assert len(rows) == 31
    names = [row["driver"] for row in rows]
    assert len(names) == len(set(names))
    source_root = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))
    if source_root.is_dir():
        source_files = sorted((source_root / "kernels/gemm_mxgemmini").glob(
            "mxgemm.fp*.cpp"))
        assert {str(path.relative_to(source_root)) for path in source_files} == set(names)
    for row in rows:
        name = row["driver"]
        quant = ".requant." in name
        expected = ("radiance_header_matched_on_pinned_spike" if quant else
                    "source_golden_matched_on_pinned_spike")
        assert row["status"] == expected
        assert row["comparison"] == (
            "compared_source_fp6_packed_bytes" if quant and ".fp6." in name else
            "compared_source_fp8_codes" if quant else "compared_bf16_outputs")
        m, n, _ = row["shape_mnk"]
        assert row["compared_count"] == m * n // (2 if quant and ".fp6." in name else 1)
        if quant:
            assert row["compared_source_e8m0_scales"] == m * n // 32
        receipt_path = ROOT / row["receipt"]
        assert receipt_path.is_relative_to(ROOT / "docs/evidence")
        assert _sha(receipt_path) == row["receipt_sha256"]
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == expected
        assert receipt["source_driver_sha256"] == row["driver_sha256"]
        assert receipt["source_header_sha256"] == row["source_header_sha256"]
        assert receipt["shape_mnk"] == row["shape_mnk"]
        assert receipt[row["comparison"]] == row["compared_count"]
        if quant:
            assert receipt["compared_source_e8m0_scales"] == (
                row["compared_source_e8m0_scales"])
        assert receipt["rtl_revision"] == index["rtl_revision"]
        capture_path = ROOT / row["model2mlir_capture_receipt"]
        assert capture_path.is_relative_to(ROOT / "docs/evidence")
        assert _sha(capture_path) == row["model2mlir_capture_receipt_sha256"]
        capture = json.loads(capture_path.read_text())
        assert capture["source_driver_sha256"] == row["driver_sha256"]
        assert capture["model2mlir_revision"] == index["model2mlir_revision"]
        if source_root.is_dir():
            assert _sha(source_root / name) == row["driver_sha256"]
