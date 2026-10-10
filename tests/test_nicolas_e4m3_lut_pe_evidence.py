"""Keep the E4M3 LUT-enable PE RTL result narrower than full-loop parity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.qualify_nicolas_e4m3_lut_pe import (
    CONTROLLER_SHA256, GEMMINI_REVISION, MXGEN_REVISION, TEST, TEST_SHA256)


EVIDENCE = (Path(__file__).resolve().parents[1] /
            "docs/evidence/nicolas_e4m3_lut_pe_266c593")


def test_pinned_nicolas_mode9_pe_rtl_result():
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    log = (EVIDENCE / "test.log").read_bytes()
    assert receipt["schema"] == "mx_gemmini.nicolas_e4m3_lut_pe_rtl_test.v1"
    assert receipt["status"] == "pe_mode9_lut_enable_arithmetic_passed"
    assert "no LUT DMA, ExecuteController simulation" in receipt["scope"]
    assert "full RoCC loop, or FPGA claim" in receipt["scope"]
    assert (receipt["gemmini_revision"], receipt["mxgen_revision"]) == (
        GEMMINI_REVISION, MXGEN_REVISION)
    assert (receipt["test_source_sha256"], receipt["controller_source_sha256"]) == (
        TEST_SHA256, CONTROLLER_SHA256)
    assert receipt["command"] == ["./mill", "test.testOnly", TEST]
    assert receipt["test_exit_code"] == 0
    assert receipt["test_log_sha256"] == hashlib.sha256(log).hexdigest()
    assert b"Total number of tests run: 1" in log
    assert b"Tests: succeeded 1, failed 0" in log
    assert b"All tests passed." in log
