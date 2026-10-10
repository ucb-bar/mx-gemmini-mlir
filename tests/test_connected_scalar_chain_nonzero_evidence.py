"""Audit the published-checkout nonidentity MX/VPU Spike replay."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.quant_reference import (
    bf16_add_scalar, exact_bf16_x2, quantize_bf16_fp8_output)
from tools.qualify_narrow_vpu_object import _append_scalar_adds


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "docs/evidence/"
          "nicolas_narrow_vpu_pair_64x64x64_64x32x64_9cd918c")
EVIDENCE = ROOT / "docs/evidence/nicolas_connected_scalar_chain_nonzero_266c593"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_nonzero_scalar_chain_preserves_source_binding_and_changes_all_goldens():
    index = _read(EVIDENCE / "index.json")
    receipt = _read(EVIDENCE / "qualification_manifest.json")
    obj = _read(EVIDENCE / "object_manifest.json")
    compiled = _read(EVIDENCE / "compile_manifest.json")
    baseline = _read(SOURCE / "index.json")
    bound = gzip.decompress((SOURCE / "bound/connected.mlir.gz").read_bytes()).decode()
    derived = (EVIDENCE / "connected_adds.mlir").read_text()
    source_bf16 = gzip.decompress((SOURCE / "bound/c1_bf16.bin.gz").read_bytes())
    assert derived == _append_scalar_adds(bound, 0x3fc0)
    assert index["compiler_revision"].startswith("f492142")
    assert index["rtl_revision"] == baseline["rtl_revision"]
    assert index["baseline_index_sha256"] == _sha((SOURCE / "index.json").read_bytes())
    assert index["model_sha256"] == receipt["model_sha256"] == (
        "0750e78eadeaef36ed94857e92168056dd6196e72dafcd09c20b6db287452071")
    assert index["status"] == "two_ordered_vpu_ops_matched_derived_nonzero_on_pinned_spike"
    assert receipt["status"] == "derived_nonzero_adds_vpu_chain_matched_on_pinned_spike"
    assert receipt["reference_kind"] == "source_chain_plus_bf16_adds_nonzero_model_derived"
    assert index["derived_adds_bf16"] == receipt["derived_adds_bf16"] == 0x3fc0
    assert receipt["spike_exit_code"] == 0
    assert index["reference_resources_sha256"] == receipt["reference_resources_sha256"]
    assert index["source_to_derived_byte_differences"] == {
        "c1_codes_ref": 4082, "c1_scales_ref": 128,
        "c2_codes_ref": 2015, "c2_scales_ref": 64}
    assert index["source_to_derived_byte_differences"] == receipt[
        "source_to_derived_byte_differences"]
    derived_bf16 = bf16_add_scalar(exact_bf16_x2(source_bf16), 0x3fc0)
    c1_codes, c1_scales = quantize_bf16_fp8_output(derived_bf16, 64, 64)
    assert c1_codes == gzip.decompress((EVIDENCE / "c1_codes_ref.bin.gz").read_bytes())
    assert c1_scales == gzip.decompress((EVIDENCE / "c1_scales_ref.bin.gz").read_bytes())
    assert receipt["resources_sha256"]["c1_bf16"] == _sha(source_bf16)
    for name in ("c1_codes_ref", "c1_scales_ref", "c2_codes_ref", "c2_scales_ref"):
        data = gzip.decompress((EVIDENCE / f"{name}.bin.gz").read_bytes())
        old = gzip.decompress((SOURCE / "bound" / f"{name}.bin.gz").read_bytes())
        assert _sha(data) == index["reference_resources_sha256"][name]
        assert sum(a != b for a, b in zip(data, old)) == index[
            "source_to_derived_byte_differences"][name]
    assert index["bound_mlir_sha256"] == obj["bound_mlir_sha256"] == _sha(derived.encode())
    assert index["object_manifest_sha256"] == _sha((EVIDENCE / "object_manifest.json").read_bytes())
    assert index["qualification_manifest_sha256"] == _sha(
        (EVIDENCE / "qualification_manifest.json").read_bytes())
    assert index["object_sha256"] == obj["object_sha256"] == compiled["object_sha256"] == _sha(
        gzip.decompress((EVIDENCE / "mx_issue.o.gz").read_bytes()))
    assert index["elf_sha256"] == receipt["elf_sha256"] == _sha(
        gzip.decompress((EVIDENCE / "mx_program.elf.gz").read_bytes()))
    assert index["spike_log_sha256"] == receipt["spike_log_sha256"] == _sha(
        (EVIDENCE / "spike.log").read_bytes())
    assert obj["issuer_c_sha256"] == _sha(gzip.decompress(
        (EVIDENCE / "mx_issue.c.gz").read_bytes()))
    assert obj["physical_program_sha256"] == _sha(gzip.decompress(
        (EVIDENCE / "physical_program.json.gz").read_bytes()))
    assert compiled["native_verifier_sha256"] is not None
    assert (obj["allocated_data_section_bytes"], obj["embedded_operand_bytes"],
            obj["embedded_golden_bytes"]) == (0, 0, 0)
    assert (index["compared_c1_bf16_values"], index["compared_c1_fp8_codes"],
            index["compared_c1_e8m0_scales"], index["compared_c2_fp8_codes"],
            index["compared_c2_e8m0_scales"]) == (4096, 4096, 128, 2048, 64)
    assert "lowered narrow MX/VPU: C1 BF16 0, C1 0 codes 0 scales, " \
           "C2 0 codes 0 scales" in (EVIDENCE / "spike.log").read_text()
    commands = json.loads(gzip.decompress(
        (EVIDENCE / "physical_program.json.gz").read_bytes()))["commands"]
    vpus = [i for i, command in enumerate(commands)
            if command["kind"] == "command" and command["funct"] == 33]
    assert [(commands[i]["rs2"]["immediate"] & 0xf,
             commands[i]["rs2"]["immediate"] >> 16) for i in vpus] == [
                 (4, 0x4000), (3, 0x3fc0)]
    assert all(commands[i + 1]["kind"] == "fence" for i in vpus)
    assert next(i for i, command in enumerate(commands)
                if command["kind"] == "command" and command["funct"] == 34) > vpus[-1]
