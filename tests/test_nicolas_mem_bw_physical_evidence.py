"""Audit Nicolas's seven-phase MX memory benchmark replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_mem_bw_physical_public_4a24503_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output(log: bytes) -> bytes:
    match = re.search(rb"MX_MEM_DUMP_BEGIN\n([0-9a-f]{32768})\nMX_MEM_DUMP_END", log)
    assert match is not None
    return bytes.fromhex(match.group(1).decode())


def test_source_and_generated_memory_phases_match_entire_mvout() -> None:
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    physical = json.loads((EVIDENCE / "physical_program.json").read_text())
    assert receipt["compiler_revision"] == (
        "4a24503caa45877528cad81d115f4e3c579e4f12")
    assert receipt["source_sha256"] == (
        "d23c749184ede0db1cb0b4ee67f3e29d8d80ddee4ddf589dd949fc7c40a3b922")
    assert receipt["status"] == (
        "source_memory_phases_and_full_mvout_matched_on_pinned_spike")
    assert receipt["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert receipt["phase_count"] == 7
    assert receipt["compared_mvout_bytes"] == 16384
    assert (receipt["source_exit_code"] == receipt["source_full_exit_code"] ==
            receipt["compiled_exit_code"] == 0)
    assert receipt["object_sha256"] == _sha(EVIDENCE / "mx_issue.o")
    assert receipt["issuer_c_sha256"] == _sha(EVIDENCE / "mx_issue.c")
    assert receipt["physical_program_sha256"] == _sha(EVIDENCE / "physical_program.json")
    for name, field in (("source", "source_spike_log_sha256"),
                        ("source_full", "source_full_spike_log_sha256"),
                        ("compiled", "compiled_spike_log_sha256")):
        assert receipt[field] == _sha(EVIDENCE / name / "spike.log")
    source = (EVIDENCE / "source_full/spike.log").read_bytes()
    compiled = (EVIDENCE / "compiled/spike.log").read_bytes()
    source_output, compiled_output = _output(source), _output(compiled)
    assert len(source_output) == len(compiled_output) == 16384
    assert source_output == compiled_output
    assert (hashlib.sha256(source_output).hexdigest() ==
            receipt["source_full_output_sha256"] ==
            receipt["compiled_output_sha256"])
    for log in (source, compiled):
        assert b"mx_mem_bw PASSED (0 mismatches in spot check)" in log
        assert len(re.findall(rb"^MEMBW ", log, re.M)) == 7
    assert [len(physical["phases"][name]) for name in
            ("setup", "a64", "b16", "scale", "mvout")] == [2, 17, 65, 1, 65]
    assert [physical["phases"][name][-1]["funct"] for name in
            ("a64", "b16", "scale", "mvout")] == [2, 2, 27, 3]
