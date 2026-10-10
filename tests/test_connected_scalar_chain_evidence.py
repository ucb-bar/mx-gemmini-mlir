"""Audit the source-bound two-operation MX/VPU Spike replay."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.quant_reference import bf16_add_scalar
from tools.qualify_narrow_vpu_object import _append_zero_adds


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_connected_scalar_chain_266c593"
SOURCE = (ROOT / "docs/evidence/"
          "nicolas_narrow_vpu_pair_64x64x64_64x32x64_9cd918c")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_two_ordered_vpu_commands_match_every_source_identity_reference():
    index = _read(EVIDENCE / "index.json")
    obj = _read(EVIDENCE / "object_manifest.json")
    compiled = _read(EVIDENCE / "compile_manifest.json")
    qualified = _read(EVIDENCE / "qualification_manifest.json")
    physical = json.loads(gzip.decompress(
        (EVIDENCE / "physical_program.json.gz").read_bytes()))
    source_bound = gzip.decompress(
        (SOURCE / "bound/connected.mlir.gz").read_bytes()).decode()
    derived = (EVIDENCE / "connected_adds.mlir").read_text()
    source_c1 = gzip.decompress((SOURCE / "bound/c1_bf16.bin.gz").read_bytes())
    assert derived == _append_zero_adds(source_bound)
    assert bf16_add_scalar(source_c1, 0) == source_c1
    assert index["schema"] == "mx_gemmini.connected_scalar_chain_spike.v1"
    assert index["status"] == "two_ordered_vpu_ops_matched_source_identity_on_pinned_spike"
    assert index["scope"] == "Nicolas source MM1/MM2 plus derived ADDS +0; identity case only"
    assert index["compiler_revision"].startswith("00d0b8e")
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["baseline_index_sha256"] == _sha((SOURCE / "index.json").read_bytes())
    assert index["bound_mlir_sha256"] == obj["bound_mlir_sha256"] == _sha(derived.encode())
    assert index["object_manifest_sha256"] == _sha((EVIDENCE / "object_manifest.json").read_bytes())
    assert index["qualification_manifest_sha256"] == _sha(
        (EVIDENCE / "qualification_manifest.json").read_bytes())
    assert index["object_sha256"] == obj["object_sha256"] == compiled["object_sha256"] == (
        _sha(gzip.decompress((EVIDENCE / "mx_issue.o.gz").read_bytes())))
    assert index["elf_sha256"] == qualified["elf_sha256"] == _sha(
        gzip.decompress((EVIDENCE / "mx_program.elf.gz").read_bytes()))
    assert index["spike_log_sha256"] == qualified["spike_log_sha256"] == _sha(
        (EVIDENCE / "spike.log").read_bytes())
    assert obj["issuer_c_sha256"] == _sha(gzip.decompress(
        (EVIDENCE / "mx_issue.c.gz").read_bytes()))
    assert obj["physical_program_sha256"] == _sha(gzip.decompress(
        (EVIDENCE / "physical_program.json.gz").read_bytes()))
    assert compiled["lowering_family"] == "resident_vpu_pair"
    assert compiled["native_verifier_sha256"] is not None
    assert (obj["allocated_data_section_bytes"], obj["embedded_operand_bytes"],
            obj["embedded_golden_bytes"]) == (0, 0, 0)
    assert qualified["status"] == "derived_zero_adds_vpu_chain_matched_on_pinned_spike"
    assert qualified["reference_kind"] == "source_chain_plus_bf16_adds_zero_identity"
    assert qualified["spike_exit_code"] == 0
    assert tuple(index[key] for key in (
        "compared_c1_bf16_values", "compared_c1_fp8_codes",
        "compared_c1_e8m0_scales", "compared_c2_fp8_codes",
        "compared_c2_e8m0_scales")) == (4096, 4096, 128, 2048, 64)
    assert "lowered narrow MX/VPU: C1 BF16 0, C1 0 codes 0 scales, " \
           "C2 0 codes 0 scales" in (EVIDENCE / "spike.log").read_text()
    commands = physical["commands"]
    vector_indices = [i for i, item in enumerate(commands)
                      if item["kind"] == "command" and item["funct"] == 33]
    assert len(vector_indices) == 2
    assert [(commands[i]["rs2"]["immediate"] & 0xf,
             commands[i]["rs2"]["immediate"] >> 16)
            for i in vector_indices] == [(4, 0x4000), (3, 0)]
    assert commands[vector_indices[0] + 1]["kind"] == "fence"
    assert commands[vector_indices[1] + 1]["kind"] == "fence"
    assert next(i for i, item in enumerate(commands)
                if item["kind"] == "command" and item["funct"] == 34) > vector_indices[1]
