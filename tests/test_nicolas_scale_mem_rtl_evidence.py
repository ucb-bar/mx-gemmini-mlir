"""Audit scoped production RTL scale-half tests and their compiler command link."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_scale_mem_rtl_266c593"
PROBE = ROOT / "tools/rtl_scale_half_probe"
FP6_PHYSICAL = (ROOT / "docs/evidence/radiance_fp6_alternating_80f84ca/"
                "physical_program.json")
VALUES = [(0, 0, 11), (0, 1, 14), (1, 0, 41), (1, 1, 44), (0, 0, 11)]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_scale_memory_half_selectors_and_fp6_command_bits() -> None:
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    assert receipt["schema"] == "mx_gemmini.nicolas_scale_half_rtl_probe.v1"
    assert receipt["status"] == (
        "scale_half_selectors_matched_on_actual_scaling_factor_mem_rtl")
    assert receipt["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert receipt["rtl_scale_mem_sha256"] == (
        "ed44c21427c8fd9e0bd74ea072be7dd971a593827ff59dcd97d0ea6acff2b549")
    assert receipt["scale_sram_depth"] == 16
    assert receipt["scale_sram_width_bits"] == 128
    assert [(row["activation_half"], row["weight_half"], row["combined_e8m0"])
            for row in receipt["selectors"]] == VALUES
    for name, digest in receipt["probe_inputs_sha256"].items():
        assert _sha(PROBE / name) == digest, name

    probes = receipt["probes"]
    assert [(probe["name"], probe["mesh_rows"],
             probe["reset_between_selector_cases"]) for probe in probes] == [
                 ("dim16", 16, True), ("wave4", 4, False)]
    for probe in probes:
        log = EVIDENCE / f"{probe['name']}.log"
        assert _sha(log) == probe["test_log_sha256"]
        output = log.read_text()
        assert "All tests passed." in output
        assert "Total number of tests run: 1" in output
        observed = [(int(a), int(w), int(value)) for a, w, value in
                    re.findall(r"act=(\d+) weight=(\d+) combined_e8m0=(\d+)", output)]
        assert observed == VALUES

    physical = json.loads(FP6_PHYSICAL.read_text())
    selectors = [(step["wave"], (step["command"]["rs1"]["or_bits"] >> 60) & 3)
                 for step in physical["steps"] if step["phase"] == "select_scales"
                 and "funct" in step["command"]]
    assert selectors == [(wave, 3 if wave & 1 else 0) for wave in range(16)]
    assert {(act, weight) for act, weight, _ in VALUES} == {
        (0, 0), (0, 1), (1, 0), (1, 1)}
