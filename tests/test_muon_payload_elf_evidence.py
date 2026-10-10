"""Audit the data-bound Muon ELF and preserve the failed Cyclotron scope."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.quant_reference import exact_bf16_x2
from mx_gemmini_support.source_payload import load_bundle
from tools.link_mx_muon_payload import _checked_payload, _tile_major_reference


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/fp8_vpu_muon_payload_266c593"
OBJECT = ROOT / "docs/evidence/fp8_vpu_muon_mmio_object_266c593"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_muon_payload_elf_is_bound_to_source_bytes_and_full_bf16_reference():
    index = json.loads((EVIDENCE / "index.json").read_text())
    assert index["schema"] == "mx_gemmini.fp8_vpu_muon_payload_archive.v1"
    for name, digest in index["files_sha256"].items():
        assert _sha((EVIDENCE / name).read_bytes()) == digest
    receipt = json.loads((EVIDENCE / "payload_manifest.json").read_text())
    assert receipt == json.loads((EVIDENCE / "payload_manifest_repro.json").read_text())
    assert receipt["schema"] == "mx_gemmini.muon_payload_elf.v1"
    assert receipt["status"] == "source_payload_bound_muon_elf_unexecuted"
    assert receipt["qualification"] == "linked_payload_and_verifier_only"
    assert receipt["precision"] == "FP8"
    assert receipt["bf16_output_layout"] == "row_major_bf16"
    assert receipt["payload_linker_sha256"] == _sha((
        ROOT / "tools/link_mx_muon_payload.py").read_bytes())
    assert receipt["object_manifest_sha256"] == _sha((OBJECT / "object_manifest.json").read_bytes())
    assert receipt["physical_program_sha256"] == _sha(gzip.decompress((
        OBJECT / "physical_program.json.gz").read_bytes()))
    bundle = ROOT / index["source_bundle"]
    assert receipt["bundle_manifest_sha256"] == _sha((bundle / "manifest.json").read_bytes())
    _, resources = load_bundle(bundle)
    expected = exact_bf16_x2(resources["golden_bf16"])
    assert receipt["shape_mnk"] == [256, 256, 256]
    assert receipt["bf16_elements_to_compare"] == 65536
    assert receipt["expected_bf16_sha256"] == _sha(expected)
    assert receipt["files_sha256"]["mx_expected_bf16.bin"] == _sha(expected)
    for name in ("activation", "activation_scales", "weight", "weight_scales"):
        assert receipt["files_sha256"][f"{name}.bin"] == _sha(resources[name])
    for name in ("mx_kernel.cpp", "mx_payload.S", "mx_kernel.elf"):
        assert receipt["files_sha256"][name] == _sha((EVIDENCE / name).read_bytes())
    source = (EVIDENCE / "mx_kernel.cpp").read_text()
    assert "mx_issue(activation, activation_scales, output_bf16, " in source
    assert "scratch_output_scales, weight, weight_scales, UINT32_C(0x00084000))" in source
    assert "i < 65536" in source and "mx_mismatch_count = mismatches" in source
    assert "mx_completed = 1" in source
    assembly = (EVIDENCE / "mx_payload.S").read_text()
    for name in ("activation", "activation_scales", "weight", "weight_scales",
                 "mx_expected_bf16"):
        assert f'.incbin "{name}.bin"' in assembly
        assert f".size {name}, .-{name}" in assembly
    elf = (EVIDENCE / "mx_kernel.elf").read_bytes()
    assert elf[:4] == b"\x7fELF" and elf[4] == 1
    assert int.from_bytes(elf[18:20], "little") == 243


def test_linker_rejects_a_different_source_bundle(tmp_path):
    for name in ("mx_issue.o", "mx_issue.h", "object_manifest.json"):
        (tmp_path / name).write_bytes((OBJECT / name).read_bytes())
    (tmp_path / "physical_program.json").write_bytes(gzip.decompress((
        OBJECT / "physical_program.json.gz").read_bytes()))
    receipt, _, buffers, expected = _checked_payload(
        tmp_path, ROOT / "docs/evidence/radiance_tilewise_vpu_x2_266c593/bundle")
    assert receipt["shape_mnk"] == [256, 256, 256]
    assert len(buffers) == 6 and len(expected) == 131072
    with pytest.raises(ValueError, match="differ"):
        _checked_payload(
            tmp_path, ROOT / "docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593/bundle")


def test_tile_major_reference_follows_the_physical_output_plan():
    row_major = b"".join(i.to_bytes(2, "little") for i in range(16))
    plan = {"tile": [2, 2, 2], "output_tiles": [
        {"index": i, "m_start": row, "n_start": col}
        for i, (row, col) in enumerate(((0, 0), (0, 2), (2, 0), (2, 2)))]}
    actual = _tile_major_reference(row_major, [4, 4, 4], plan)
    assert [int.from_bytes(actual[i:i+2], "little") for i in range(0, 32, 2)] == [
        0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    with pytest.raises(ValueError, match="overlap"):
        _tile_major_reference(row_major, [4, 4, 4], {**plan, "output_tiles": [
            *plan["output_tiles"][:3], {"index": 3, "m_start": 0, "n_start": 0}]})


def test_current_cyclotron_result_is_explicitly_a_numerical_failure():
    diagnostic = json.loads((EVIDENCE / "cyclotron_diagnostic.json").read_text())
    assert diagnostic["schema"] == "mx_gemmini.muon_payload_cyclotron_diagnostic.v1"
    assert diagnostic["status"] == "executed_but_numerical_parity_failed_on_current_cyclotron"
    assert diagnostic["linked_elf_sha256"] == _sha((EVIDENCE / "mx_kernel.elf").read_bytes())
    assert diagnostic["cyclotron_log_sha256"] == _sha((EVIDENCE / "cyclotron.log").read_bytes())
    assert diagnostic["completed"] == 1
    assert diagnostic["reported_bf16_mismatches"] == 65517
    assert diagnostic["independently_counted_bf16_mismatches"] == 65517
    assert diagnostic["nonzero_output_bytes"] == 0
    assert b"simulation finished" in (EVIDENCE / "cyclotron.log").read_bytes()


def test_isolated_patched_cyclotron_matches_all_compiler_bf16_outputs():
    first = json.loads((EVIDENCE / "patched_cyclotron_qualification.json").read_text())
    repro = json.loads((EVIDENCE / "patched_cyclotron_qualification_repro.json").read_text())
    assert first["schema"] == repro["schema"] == (
        "mx_gemmini.muon_payload_patched_cyclotron.v1")
    assert first["status"] == repro["status"] == (
        "all_65536_bf16_outputs_matched_on_isolated_patched_cyclotron")
    assert first["qualification"] == repro["qualification"] == "experimental_functional_model_only"
    assert first["scope"] == repro["scope"]
    assert first["qualifier_sha256"] == repro["qualifier_sha256"] == _sha((
        ROOT / "tools/qualify_mx_muon_cyclotron.py").read_bytes())
    assert first["patch_sha256"] == repro["patch_sha256"] == _sha((
        EVIDENCE / "cyclotron_compiler_stream.patch").read_bytes())
    assert first["compiler_elf_sha256"] == repro["compiler_elf_sha256"] == _sha((
        EVIDENCE / "mx_kernel.elf").read_bytes())
    assert first["payload_manifest_sha256"] == repro["payload_manifest_sha256"] == _sha((
        EVIDENCE / "payload_manifest.json").read_bytes())
    assert first["cyclotron_binary_sha256"] == repro["cyclotron_binary_sha256"]
    assert first["patched_model_sha256"] == repro["patched_model_sha256"]
    assert first["cyclotron_mx_tests_passed"] == repro["cyclotron_mx_tests_passed"] == 35
    assert first["cyclotron_mx_tests_sha256"] == _sha((
        EVIDENCE / "mxgemmini_tests.log").read_bytes())
    assert first["bf16_outputs_checked"] == repro["bf16_outputs_checked"] == 65536
    _, resources = load_bundle(ROOT / "docs/evidence/radiance_tilewise_vpu_x2_266c593/bundle")
    expected = exact_bf16_x2(resources["golden_bf16"])
    dump = gzip.decompress((EVIDENCE / "patched_cyclotron_gmem.bin.gz").read_bytes())
    base = min(first["output_address"], first["mismatch_address"], first["completed_address"])
    output = dump[first["output_address"]-base:first["output_address"]-base+len(expected)]
    mismatch = dump[first["mismatch_address"]-base:first["mismatch_address"]-base+4]
    completed = dump[first["completed_address"]-base:first["completed_address"]-base+4]
    assert output == expected
    assert int.from_bytes(mismatch, "little") == 0
    assert int.from_bytes(completed, "little") == 1
    for qualification in (first, repro):
        assert qualification["expected_bf16_sha256"] == _sha(expected)
        assert len(qualification["runs"]) == 2
        for run in qualification["runs"]:
            assert run["gmem_sha256"] == _sha(dump)
            assert run["output_sha256"] == _sha(expected)
            assert run["completed"] == 1
            assert run["reported_bf16_mismatches"] == 0
            assert run["independent_bf16_mismatches"] == 0
    assert first["runs"][0]["cyclotron_log_sha256"] == _sha((
        EVIDENCE / "first.cyclotron.log").read_bytes())
    assert first["runs"][1]["cyclotron_log_sha256"] == _sha((
        EVIDENCE / "repro.cyclotron.log").read_bytes())
