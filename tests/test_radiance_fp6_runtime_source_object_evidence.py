"""Audit the two-wave FP6 object with runtime LUT banks on pinned Spike."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.command_ir import emit_c
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile
from tools.emit_mx_object import _referenced_buffers


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/radiance_fp6_fullout_266c593/fp6_128x128x1024"
EVIDENCE = ROOT / "docs/evidence/radiance_fp6_runtime_source_object_266c593"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archived(name: str) -> bytes:
    path = EVIDENCE / name
    return path.read_bytes() if path.is_file() else gzip.decompress(
        (EVIDENCE / f"{name}.gz").read_bytes())


def test_generated_fp6_source_object_rebinds_packed_codes_scales_and_luts():
    index = json.loads(_archived("index.json"))
    assert index == json.loads(_archived("index_repro.json"))
    assert index["schema"] == "mx_gemmini.runtime_fp6_source_object_spike.v1"
    assert index["status"] == "two_runtime_fp6_source_payloads_matched_on_pinned_spike"
    assert index["spike_exit_code"] == 0
    assert index["compared_bf16_outputs"] == 2 * 128 * 128
    assert index["permutation"] == "swap_M_64_rows_and_N_64_columns_with_pair_LUT_banks_v1"
    assert "serial two-wave schedule" in index["scope"]
    assert "no Muon/RTL/FPGA claim" in index["scope"]
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["qualifier_sha256"] == _sha(
        (ROOT / "tools/qualify_runtime_fp6_source_object.py").read_bytes())
    for digest, artifact in (("object_sha256", "mx_issue.o"),
                             ("object_manifest_sha256", "object_manifest.json"),
                             ("elf_sha256", "mx_runtime_fp6_source.elf"),
                             ("spike_log_sha256", "spike.log")):
        assert index[digest] == _sha(_archived(artifact))
    assert b"runtime source FP6: 0/32768 BF16 mismatches" in _archived("spike.log")
    elf = _archived("mx_runtime_fp6_source.elf")
    assert elf[:4] == b"\x7fELF" and int.from_bytes(elf[18:20], "little") == 243

    receipt = json.loads(_archived("object_manifest.json"))
    assert receipt == json.loads(_archived("object_manifest_repro.json"))
    assert receipt["object_sha256"] == index["object_sha256"]
    assert receipt["issuer_c_sha256"] == _sha(_archived("mx_issue.c"))
    assert receipt["issuer_h_sha256"] == _sha(_archived("mx_issue.h"))
    assert len(receipt["object_emitter_sha256"]) == 64
    assert receipt["bound_mlir_sha256"] == _sha((SOURCE / "bound.mlir").read_bytes())
    assert receipt["physical_program_sha256"] == _sha(_archived("physical_program.json"))
    assert receipt["allocated_data_section_bytes"] == 0
    assert receipt["embedded_operand_bytes"] == receipt["embedded_golden_bytes"] == 0
    assert [item["name"] for item in receipt["buffer_abi"]] == [
        "activation", "activation_lut", "activation_scales", "output_bf16",
        "output_lut", "scratch_output_scales", "weight", "weight_lut",
        "weight_scales"]
    assert receipt["buffer_abi"][3]["layout"] == "row_major_bf16"
    assert receipt["source_bundle_manifest_sha256"] == index[
        "source_bundle_manifest_sha256"] == _sha((SOURCE / "bundle/manifest.json").read_bytes())

    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE3M2OnlyGemminiRocketConfig.json")
    manifest, first = load_bundle(SOURCE / "bundle")
    program = lower_bound_source((SOURCE / "bound.mlir").read_text(),
                                 profile, manifest, first)
    assert _referenced_buffers(program, manifest) == receipt["buffer_abi"]
    assert program.receipt() == json.loads(_archived("physical_program.json"))
    assert emit_c([step.command for step in program.steps], transport="rocket_rocc",
                  buffers=tuple(entry["name"] for entry in receipt["buffer_abi"])) == (
                      _archived("mx_issue.c").decode())
    assert program.mode == "spike_serial" and len(program.plan["waves"]) == 2
    assert sum(step.phase == "upload_lut" and
               getattr(step.command, "funct", None) == 29
               for step in program.steps) == 3

    first_case, second_case = index["payloads"]
    assert [case["name"] for case in index["payloads"]] == ["first", "second"]
    names = ("activation", "activation_lut", "activation_scales", "output_lut",
             "weight", "weight_lut", "weight_scales", "golden_bf16")
    for name in names:
        assert _archived(f"data/first_{name}.bin") == first[name]
        assert first_case[f"{name}_sha256"] == _sha(first[name])
        assert second_case[f"{name}_sha256"] == _sha(
            _archived(f"data/second_{name}.bin"))
        assert first_case[f"{name}_sha256"] != second_case[f"{name}_sha256"]

    # Rebuild the second expected matrix by logical coordinates, separate
    # from the packed-byte permutation in the qualification tool.
    second_golden = _archived("data/second_golden_bf16.bin")
    original = first["golden_bf16"]
    for row in range(128):
        for col in range(128):
            dst = (row * 128 + col) * 2
            src = (((row + 64) % 128) * 128 + (col + 64) % 128) * 2
            assert second_golden[dst:dst + 2] == original[src:src + 2]
    assert _archived("data/second_activation.bin") == (
        first["activation"][32768:] + first["activation"][:32768])
    for name in ("activation_lut", "weight_lut", "output_lut"):
        assert _archived(f"data/second_{name}.bin") == (
            first[name][384:] + first[name][:384])
    for name, rows, width in (("activation_scales", 32, 128),
                              ("weight", 1024, 64),
                              ("weight_scales", 32, 128)):
        old = first[name]
        new = _archived(f"data/second_{name}.bin")
        for row in range(rows):
            half = width // 2
            assert new[row * width:(row + 1) * width] == (
                old[row * width + half:(row + 1) * width] +
                old[row * width:row * width + half])
    driver = _archived("mx_runtime_driver.c")
    assert driver.count(b"mx_issue(") == 2
    assert b"first_after_second" in driver
