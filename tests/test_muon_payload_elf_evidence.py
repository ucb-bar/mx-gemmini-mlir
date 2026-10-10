"""Audit the data-bound Muon ELF and preserve the failed Cyclotron scope."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.quant_reference import exact_bf16_x2
from mx_gemmini_support.source_payload import load_bundle
from tools.link_mx_muon_payload import _checked_payload


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
