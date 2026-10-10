"""Audit committed Radiance FP8 source rebinding on Nicolas's pinned Spike."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile
from tools.emit_mx_object import _referenced_buffers


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/radiance_fp8_512_tk256_latest_266c593"
EVIDENCE = ROOT / "docs/evidence/radiance_fp8_runtime_source_object_266c593"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archived(name: str) -> bytes:
    path = EVIDENCE / name
    return path.read_bytes() if path.is_file() else gzip.decompress(
        (EVIDENCE / f"{name}.gz").read_bytes())


def test_committed_fp8_source_object_rebinds_two_payloads():
    index = json.loads(_archived("index.json"))
    assert index == json.loads(_archived("index_repro.json"))
    assert index["schema"] == "mx_gemmini.runtime_fp8_source_object_spike.v1"
    assert index["status"] == "two_runtime_fp8_source_payloads_matched_on_pinned_spike"
    assert index["spike_exit_code"] == 0
    assert index["compared_bf16_outputs"] == 2 * 128 * 128
    assert index["permutation"] == "swap_M_64_rows_and_N_64_columns_v1"
    assert "no Muon/RTL/FPGA claim" in index["scope"]
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["qualifier_sha256"] == _sha(
        (ROOT / "tools/qualify_runtime_fp8_source_object.py").read_bytes())
    for digest, artifact in (("object_sha256", "mx_issue.o"),
                             ("object_manifest_sha256", "object_manifest.json"),
                             ("elf_sha256", "mx_runtime_fp8_source.elf"),
                             ("spike_log_sha256", "spike.log")):
        assert index[digest] == _sha(_archived(artifact))
    assert b"runtime source FP8: 0/32768 BF16 mismatches" in _archived("spike.log")
    elf = _archived("mx_runtime_fp8_source.elf")
    assert elf[:4] == b"\x7fELF" and int.from_bytes(elf[18:20], "little") == 243

    receipt = json.loads(_archived("object_manifest.json"))
    assert receipt["object_sha256"] == index["object_sha256"]
    assert receipt["issuer_c_sha256"] == _sha(_archived("mx_issue.c"))
    assert receipt["issuer_h_sha256"] == _sha(_archived("mx_issue.h"))
    assert receipt["object_emitter_sha256"] == _sha(
        (ROOT / "tools/emit_mx_object.py").read_bytes())
    assert receipt["bound_mlir_sha256"] == _sha((SOURCE / "bound.mlir").read_bytes())
    assert receipt["physical_program_sha256"] == _sha(_archived("physical_program.json"))
    assert receipt["allocated_data_section_bytes"] == 0
    assert receipt["embedded_operand_bytes"] == receipt["embedded_golden_bytes"] == 0
    assert receipt["buffer_abi"][2]["layout"] == "row_major_bf16"
    assert receipt["buffer_abi"][2]["minimum_bytes"] == 128 * 128 * 2
    assert receipt["source_bundle_manifest_sha256"] == index[
        "source_bundle_manifest_sha256"] == _sha((SOURCE / "bundle/manifest.json").read_bytes())

    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    manifest, first = load_bundle(SOURCE / "bundle")
    program = lower_bound_source((SOURCE / "bound.mlir").read_text(),
                                 profile, manifest, first)
    assert _referenced_buffers(program, manifest) == receipt["buffer_abi"]
    assert program.receipt() == json.loads(_archived("physical_program.json"))
    assert manifest["shape_mnk"][:2] == manifest["tile_mnk"][:2]

    first_case, second_case = index["payloads"]
    assert [case["name"] for case in index["payloads"]] == ["first", "second"]
    for name in ("activation", "activation_scales", "weight",
                 "weight_scales", "golden_bf16"):
        assert _archived(f"data/first_{name}.bin") == first[name]
        assert first_case[f"{name}_sha256"] == _sha(first[name])
        assert second_case[f"{name}_sha256"] == _sha(
            _archived(f"data/second_{name}.bin"))
        assert first_case[f"{name}_sha256"] != second_case[f"{name}_sha256"]

    # Compute the second matrix directly from logical source coordinates;
    # this is independent of the tool's packed-byte permutation procedure.
    second_golden = _archived("data/second_golden_bf16.bin")
    source_golden = first["golden_bf16"]
    for row in range(128):
        for col in range(128):
            dst = (row * 128 + col) * 2
            src = (((row + 64) % 128) * 128 + (col + 64) % 128) * 2
            assert second_golden[dst:dst + 2] == source_golden[src:src + 2]
    assert _archived("data/second_activation.bin") == (
        first["activation"][32768:] + first["activation"][:32768])
    for name, rows in (("activation_scales", 16),
                       ("weight", 512), ("weight_scales", 16)):
        old = first[name]
        new = _archived(f"data/second_{name}.bin")
        for row in range(rows):
            assert new[row * 128:(row + 1) * 128] == (
                old[row * 128 + 64:(row + 1) * 128] +
                old[row * 128:row * 128 + 64])
    driver = _archived("mx_runtime_driver.c")
    assert driver.count(b"mx_issue(") == 2
    assert b"first_after_second" in driver
