"""Audit the published unified object compiler and full source parity replay."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/mx_compile_object_dispatch_d3156e4"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_all_three_graph_families_have_reproducible_data_free_objects():
    index = _read(EVIDENCE / "index.json")
    assert index["schema"] == "mx_gemmini.single_object_compiler_replay.v1"
    assert index["fresh_compiler_revision"].startswith("d3156e4")
    assert index["status"] == (
        "three_graph_families_and_three_source_precisions_reproduced_on_spike")
    assert [case["lowering_family"] for case in index["cases"]] == [
        "source_contract", "resident_pair", "resident_vpu_pair"]
    for case in index["cases"]:
        directory = EVIDENCE / case["family"]
        assert all(_file_sha(directory / name) == digest
                   for name, digest in case["files_sha256"].items())
        local, fresh = (_read(directory / name) for name in
                        ("compile_manifest.json", "fresh_compile_manifest.json"))
        obj, fresh_obj = (_read(directory / name) for name in
                          ("object_manifest.json", "fresh_object_manifest.json"))
        assert local["lowering_family"] == fresh["lowering_family"] == case["lowering_family"]
        assert local["compiler_source_closure_sha256"] == (
            fresh["compiler_source_closure_sha256"])
        assert (obj["object_sha256"] == fresh_obj["object_sha256"] ==
                case["object_sha256"] == _file_sha(directory / "mx_issue.o"))
        assert obj["allocated_data_section_bytes"] == 0
        assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
        assert _sha(gzip.decompress((directory / "mx_issue.c.gz").read_bytes())) == (
            case["issuer_c_sha256"])
        assert _sha(gzip.decompress((directory / "physical_program.json.gz").read_bytes())) == (
            case["physical_program_sha256"])
    fp4_issuer = gzip.decompress((EVIDENCE / "fp4/mx_issue.c.gz").read_bytes())
    prior_fp4 = gzip.decompress((ROOT / "docs/evidence/radiance_plain_mx_profile_trio_266c593/"
                                  "fp4/build/mx_issue.c.gz").read_bytes())
    assert fp4_issuer == prior_fp4
    for family, baseline in (
            ("pair", "nicolas_resident_pair_object_1eebfc5/m96/object_manifest.json"),
            ("vpu", "nicolas_resident_vpu_object_6c9ed40/object_manifest.json")):
        assert _read(EVIDENCE / family / "object_manifest.json")["object_sha256"] == (
            _read(ROOT / "docs/evidence" / baseline)["object_sha256"])


def test_fresh_source_objects_match_every_fp4_fp6_fp8_spike_output():
    index = _read(EVIDENCE / "index.json")
    assert index["source_precision_fresh_compiler_revision"].startswith("0c9931a")
    assert index["source_precision_compared_bf16_outputs"] == 36864
    assert [(row["precision"], row["compared_bf16_outputs"]) for row in
            index["source_precision_replay"]] == [
                ("FP4", 4096), ("FP6", 16384), ("FP8", 16384)]
    for row in index["source_precision_replay"]:
        fmt = row["precision"].lower()
        directory = EVIDENCE / "source_precision" / fmt
        assert all(_file_sha(directory / name) == digest
                   for name, digest in row["files_sha256"].items())
        local = _read(directory / "local_compile_manifest.json")
        fresh = _read(directory / "compile_manifest.json")
        obj = _read(directory / "object_manifest.json")
        qualified = _read(directory / "qualification_manifest.json")
        assert (local["object_sha256"] == fresh["object_sha256"] ==
                obj["object_sha256"] == row["object_sha256"])
        assert obj["allocated_data_section_bytes"] == 0
        assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
        assert _file_sha(directory / "mx_issue.o") == row["object_sha256"]
        assert _sha(gzip.decompress((directory / "mx_issue.c.gz").read_bytes())) == (
            row["issuer_c_sha256"])
        assert _sha(gzip.decompress((directory / "physical_program.json.gz").read_bytes())) == (
            row["physical_program_sha256"])
        assert _sha(gzip.decompress((directory / "mx_program.elf.gz").read_bytes())) == (
            row["elf_sha256"])
        assert qualified["status"] == "source_golden_matched_on_pinned_spike"
        assert qualified["issuer_origin"] == "linkable_source_object"
        assert qualified["spike_exit_code"] == 0
        assert qualified["compared_bf16_outputs"] == row["compared_bf16_outputs"]
        assert qualified["object_sha256"]["mx_issue.o"] == row["object_sha256"]
        assert _file_sha(directory / "spike.log") == row["spike_log_sha256"]
        baseline = (ROOT / "docs/evidence/radiance_plain_mx_profile_trio_266c593" /
                    fmt / "build/spike.log")
        assert _file_sha(baseline) == row["baseline_spike_log_sha256"] == (
            row["spike_log_sha256"])
    vpu = _read(EVIDENCE / "vpu_spike_manifest.json")
    assert _file_sha(EVIDENCE / "vpu_spike_manifest.json") == (
        index["vpu_spike_manifest_sha256"])
    assert (_file_sha(EVIDENCE / "vpu_spike.log") ==
            index["vpu_spike_log_sha256"] == vpu["spike_log_sha256"])
    assert vpu["issuer_origin"] == "linkable_resident_vpu_object"
    assert vpu["compared_bf16_values"] == 4096
    assert vpu["compared_fp8_codes"] == 8192
    assert vpu["compared_e8m0_scales"] == 256
