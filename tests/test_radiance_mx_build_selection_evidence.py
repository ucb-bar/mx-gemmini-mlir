"""Keep compiler parity and Radiance Makefile selection distinct."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.source_build_selection import selected_mx_gemm_drivers


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
ROSTER = EVIDENCE / "radiance_mx_gemm_latest_e9ded36_ee22"
REPORT = EVIDENCE / "radiance_mx_gemm_build_selection_80f84ca.json"
MAKEFILE = EVIDENCE / "radiance_mx_gemm_build_selection_80f84ca.Makefile"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_selection_audit_covers_all_spike_matched_drivers():
    report = json.loads(REPORT.read_text())
    frontend = json.loads((ROSTER / "frontend/index.json").read_text())
    spike = json.loads((ROSTER / "spike/index.json").read_text())
    assert report["schema"] == "mx_gemmini.radiance_mx_build_selection_audit.v1"
    assert report["source_revision"] == "80f84caedbabc663a7433c1da4455b936cca41f3"
    assert report["roster_source_revision"] == frontend["source_revision"] == spike[
        "source_revision"]
    assert report["frontend_index_sha256"] == _sha(ROSTER / "frontend/index.json")
    assert report["spike_index_sha256"] == _sha(ROSTER / "spike/index.json")
    assert report["source_makefile_sha256"] == _sha(MAKEFILE)
    selected = selected_mx_gemm_drivers(MAKEFILE.read_text())
    rows = {row["driver"]: row for row in report["rows"]}
    frontend_rows = {row["driver"]: row for row in frontend["rows"]}
    spike_rows = {row["driver"]: row for row in spike["rows"]}
    assert set(rows) == set(frontend_rows) == set(spike_rows)
    assert (report["named_driver_count"], report["source_makefile_selected_count"],
            report["source_recipe_only_count"]) == (31, 18, 13)
    for driver, row in rows.items():
        name = Path(driver).name
        assert row["source_makefile_selected"] == (name in selected)
        assert row["source_driver_sha256"] == frontend_rows[driver]["driver_sha256"]
        assert row["source_driver_sha256"] == spike_rows[driver]["source_driver_sha256"]
        assert row["spike_status"] == spike_rows[driver]["status"]
        assert row["precision"] == frontend_rows[driver]["precision"]
        assert row["shape_mnk"] == frontend_rows[driver]["shape_mnk"]
        assert row["tile_mnk"] == frontend_rows[driver]["tile_mnk"]
    assert not rows["kernels/gemm_mxgemmini/"
                    "mxgemm.fp8.m256n256k256.tm128tn128tk256.fullout.cpp"][
                        "source_makefile_selected"]


@pytest.mark.parametrize("makefile", (
    "MU_SRCS = a.cpp\nMU_SRCS = b.cpp\n",
    "MU_SRCS = a.cpp \\\n",
    "MU_SRCS = a.cpp a.cpp\n",
    "MU_SRCS = $(SOURCES)\n",
))
def test_build_selection_parser_rejects_ambiguous_makefiles(makefile):
    with pytest.raises(ValueError):
        selected_mx_gemm_drivers(makefile)
