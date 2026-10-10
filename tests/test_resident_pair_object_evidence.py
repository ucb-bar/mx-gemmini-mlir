"""Recheck archived linkable resident-pair objects and stock Spike results."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_resident_pair_object_1eebfc5"
FRESH = ROOT / "docs/evidence/nicolas_resident_pair_object_fresh_5172afa"
INPUTS = ("a1_activation", "a1_scales", "b1_weight", "b1_scales",
          "b2_weight", "b2_scales")
OUTPUTS = ("c1_scales", "c1_tiled_observed", "c2_scales", "c2_tiled")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("m", (16, 96, 128))
def test_source_bound_resident_pair_object_matches_spike(m: int) -> None:
    index = json.loads((EVIDENCE / "index.json").read_text())
    case = index["cases"][str(m)]
    directory = EVIDENCE / f"m{m}"

    def read(name: str) -> bytes:
        plain = directory / name
        return plain.read_bytes() if plain.is_file() else gzip.decompress(
            (directory / f"{name}.gz").read_bytes())

    for name, digest in case["files_sha256"].items():
        assert _sha(read(name)) == digest, (m, name)
    obj = json.loads(read("object_manifest.json"))
    result = json.loads(read("qualification_manifest.json"))
    source = json.loads(read("source_reference_manifest.json"))
    physical = json.loads(read("physical_program.json"))
    assert index["schema"] == "mx_gemmini.resident_pair_object_archive.v1"
    assert obj["schema"] == "mx_gemmini.resident_pair_linkable_object.v1"
    assert result["schema"] == "mx_gemmini.resident_pair_object_spike.v1"
    assert obj["compiler_revision"] == result["compiler_revision"] == (
        "1eebfc5b7290ad72f57d4b859936acbcc98d6a9a")
    assert obj["shape_mnk"] == result["shape_mnk"] == case["shape_mnk"] == [m, 128, 128]
    assert obj["profile_sha256"] == result["profile_sha256"] == (
        index["profile_sha256"])
    assert obj["rtl_revision"] == result["rtl_revision"] == index["rtl_revision"]
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["defined_symbol"] == "mx_issue" and obj["undefined_symbols"] == []
    assert obj["object_sha256"] == result["object_sha256"] == _sha(read("mx_issue.o"))
    assert obj["issuer_c_sha256"] == result["source_issuer_sha256"] == (
        _sha(read("mx_issue.c")))
    assert obj["issuer_h_sha256"] == _sha(read("mx_issue.h"))
    assert obj["physical_program_sha256"] == _sha(read("physical_program.json"))
    assert result["object_manifest_sha256"] == _sha(read("object_manifest.json"))
    assert result["source_reference_manifest_sha256"] == (
        _sha(read("source_reference_manifest.json")))
    assert result["linked_elf_sha256"] == _sha(read("linked.elf"))
    assert result["spike_log_sha256"] == _sha(read("spike.log"))
    assert result["status"] == case["status"] == (
        "source_connected_object_matched_on_pinned_spike")
    assert result["spike_exit_code"] == source["spike_exit_code"] == 0
    assert (result["compared_c1_fp8_codes"], result["compared_c1_e8m0_scales"],
            result["compared_c2_fp8_codes"], result["compared_c2_e8m0_scales"]) == (
            m * 128, m * 4, m * 128, m * 4)
    assert (f"lowered connected {m}x128: C1 0 codes 0 scales; "
            "C2 0 codes 0 scales").encode() in read("spike.log")
    assert physical["shape_mnk"] == [m, 128, 128]
    assert (physical["first_site"], physical["second_site"]) == (
        "functional:matmul", "functional:matmul_1")
    abi = {entry["slot"]: entry for entry in obj["buffer_abi"]}
    assert set(abi) == set(INPUTS) | set(OUTPUTS)
    assert all(entry["name"] == slot and entry["alignment_bytes"] == 64
               for slot, entry in abi.items())
    assert abi["a1_activation"]["minimum_bytes"] == m * 128
    assert abi["a1_scales"]["minimum_bytes"] == m * 4
    assert abi["c1_tiled_observed"]["minimum_bytes"] == m * 128
    assert abi["c2_tiled"]["minimum_bytes"] == m * 128
    assert abi["c1_scales"]["minimum_bytes"] == m * 4
    assert abi["c2_scales"]["minimum_bytes"] == m * 4
    if m == 96:
        assert read("object_manifest_repro.json") == read("object_manifest.json")
        assert read("qualification_manifest_repro.json") == read("qualification_manifest.json")


def test_published_checkout_rebuilds_same_resident_pair_object_and_spike_result() -> None:
    index = json.loads((FRESH / "index.json").read_text())
    old = json.loads((EVIDENCE / "m96/object_manifest.json").read_text())
    new = json.loads((FRESH / "object_manifest.json").read_text())
    old_run = json.loads((EVIDENCE / "m96/qualification_manifest.json").read_text())
    new_run = json.loads((FRESH / "qualification_manifest.json").read_text())
    assert index["schema"] == "mx_gemmini.resident_pair_object_fresh_reproduction.v1"
    assert index["compiler_revision"] == new["compiler_revision"] == (
        "5172afafd0a099a010358d4c81038bccf40ea602")
    assert index["baseline_compiler_revision"] == old["compiler_revision"]
    for key in index["object_fields_equal_to_baseline"]:
        assert new[key] == old[key], key
    for key in index["qualification_fields_equal_to_baseline"]:
        assert new_run[key] == old_run[key], key
    assert new_run["status"] == index["status"] == (
        "source_connected_object_matched_on_pinned_spike")
    assert new_run["spike_exit_code"] == 0
    for name, digest in index["files_sha256"].items():
        path = FRESH / name
        data = path.read_bytes() if path.is_file() else gzip.decompress(
            (FRESH / f"{name}.gz").read_bytes())
        assert _sha(data) == digest, name
