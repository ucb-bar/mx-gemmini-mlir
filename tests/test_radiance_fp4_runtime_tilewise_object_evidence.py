"""Check the archived two-payload Spike run and the four-tile pointer ABI."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.quant_reference import exact_bf16_x2
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile
from tools.emit_mx_object import _referenced_buffers


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_fp4_runtime_tilewise_object_266c593"
SOURCE = ROOT / "docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archived(name: str) -> bytes:
    path = EVIDENCE / name
    if path.is_file():
        return path.read_bytes()
    return gzip.decompress((EVIDENCE / f"{name}.gz").read_bytes())


def test_four_tile_runtime_object_rebinds_two_distinct_source_payloads():
    index = json.loads(_archived("index.json"))
    assert index == json.loads(_archived("index_repro.json"))
    assert index["schema"] == "mx_gemmini.runtime_fp4_tilewise_object_spike.v1"
    assert index["status"] == "two_runtime_fp4_tilewise_payloads_matched_on_pinned_spike"
    assert index["spike_exit_code"] == 0
    assert index["compared_bf16_outputs"] == 2 * 256 * 256
    assert index["permutation"] == "swap_M_128_rows_and_N_128_columns_v1"
    assert "no Muon/FPGA claim" in index["scope"]
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    for digest, artifact in (("object_sha256", "mx_issue.o"),
                             ("object_manifest_sha256", "object_manifest.json"),
                             ("elf_sha256", "mx_runtime_fp4_tilewise.elf"),
                             ("spike_log_sha256", "spike.log")):
        assert index[digest] == _sha(_archived(artifact))
    assert len(index["extension_sha256"]) == 64
    assert index["qualifier_sha256"] == _sha(
        (ROOT / "tools/qualify_runtime_fp4_tilewise_object.py").read_bytes())
    assert b"runtime FP4 tilewise: 0/131072 BF16 mismatches" in _archived("spike.log")
    elf = _archived("mx_runtime_fp4_tilewise.elf")
    assert elf[:4] == b"\x7fELF" and int.from_bytes(elf[18:20], "little") == 243

    receipt = json.loads(_archived("object_manifest.json"))
    assert receipt["schema"] == "mx_gemmini.linkable_object.v1"
    assert receipt["object_sha256"] == index["object_sha256"]
    assert receipt["issuer_c_sha256"] == _sha(_archived("mx_issue.c"))
    assert receipt["issuer_h_sha256"] == _sha(_archived("mx_issue.h"))
    assert receipt["allocated_data_section_bytes"] == 0
    assert receipt["embedded_operand_bytes"] == receipt["embedded_golden_bytes"] == 0
    assert receipt["buffer_abi"][2]["name"] == "output_bf16"
    assert receipt["buffer_abi"][2]["layout"] == "output_tile_major_bf16"
    assert receipt["buffer_abi"][2]["minimum_bytes"] == 256 * 256 * 2
    assert receipt["physical_program_sha256"] == _sha(_archived("physical_program.json"))
    assert receipt["object_emitter_sha256"] == _sha(
        (ROOT / "tools/emit_mx_object.py").read_bytes())
    program_receipt = json.loads(_archived("physical_program.json"))
    assert len(program_receipt["plan"]["output_tiles"]) == 4
    assert program_receipt["plan"]["vector_tile_policy"] == (
        "bf16_muls_x2_each_output_tile_v1")
    assert sum(step["phase"] == "vpu" for step in program_receipt["steps"]) == 4

    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    manifest, first = load_bundle(SOURCE / "bundle")
    program = lower_bound_source((SOURCE / "tilewise_bound.mlir").read_text(),
                                 profile, manifest, first)
    assert _referenced_buffers(program, manifest) == receipt["buffer_abi"]
    assert program.receipt() == program_receipt
    assert receipt["source_bundle_manifest_sha256"] == index[
        "source_bundle_manifest_sha256"] == _sha((SOURCE / "bundle/manifest.json").read_bytes())

    first_case, second_case = index["payloads"]
    assert [case["name"] for case in index["payloads"]] == ["first", "second"]
    for name in ("activation", "activation_scales", "weight", "weight_scales"):
        assert _archived(f"data/first_{name}.bin") == first[name]
        assert first_case[f"{name}_sha256"] == _sha(first[name])
        assert second_case[f"{name}_sha256"] == _sha(
            _archived(f"data/second_{name}.bin"))
        assert first_case[f"{name}_sha256"] != second_case[f"{name}_sha256"]
    assert first_case["golden_bf16_sha256"] == _sha(first["golden_bf16"])
    assert _archived("data/first_expected_bf16.bin") == exact_bf16_x2(
        first["golden_bf16"])

    # Independently map the fixture's logical output quadrants and its packed
    # M/N input axes. This checks that the second call does arithmetic on a
    # different valid payload, rather than only changing a pointer address.
    second_golden = bytearray(256 * 256 * 2)
    source_golden = first["golden_bf16"]
    for row in range(256):
        source_row = (row + 128) % 256
        for col in range(256):
            source_col = (col + 128) % 256
            dst = (row * 256 + col) * 2
            src = (source_row * 256 + source_col) * 2
            second_golden[dst:dst + 2] = source_golden[src:src + 2]
    assert second_case["golden_bf16_sha256"] == _sha(second_golden)
    assert _archived("data/second_expected_bf16.bin") == exact_bf16_x2(
        bytes(second_golden))
    assert second_case["expected_x2_bf16_sha256"] == _sha(
        _archived("data/second_expected_bf16.bin"))
    assert _archived("data/second_activation.bin") == (
        first["activation"][16384:] + first["activation"][:16384])
    for name, rows, width in (("activation_scales", 8, 256),
                              ("weight", 256, 128),
                              ("weight_scales", 8, 256)):
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
    assert b"tile * 16384 + local" in driver
    assert b".incbin" in _archived("mx_runtime_data.S")
