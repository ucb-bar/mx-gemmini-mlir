"""Audit clean-checkout 128-cubed model2MLIR → MX/VPU object → Spike runs."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.quant_reference import exact_bf16_x2, quantize_bf16_fp8_output
from mx_gemmini_support.resident_pair_graph import INPUTS
from mx_gemmini_support.resident_vpu_graph import OUTPUTS, lower_connected_fp8_vpu_pair
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.emit_resident_vpu_object import _readout_commands


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_square_128_vpu_pair_266c593_cc4859c"
PLAIN = ROOT / "docs/evidence/nicolas_connected_plain_chain_128_266c593"
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"
COMPILER = "cc4859c98f47fdad07de3dab01f8b900ac437287"
MODEL2MLIR = "e9ded36eb85abf2d9097ac4dc11457c825853388"
RTL = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"


def _read(root: Path, name: str) -> bytes:
    raw = (root / name).read_bytes()
    return gzip.decompress(raw) if name.endswith(".gz") else raw


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("case,profile_name", (
    ("fp4_vpu", "MxE4M3Fp4VpuGemminiRocketConfig"),
    ("e4m3_vpu", "MxE4M3VpuGemminiRocketConfig"),
))
def test_square_128_vpu_source_derived_spike_parity(case: str, profile_name: str):
    archive = json.loads((EVIDENCE / "archive_manifest.json").read_text())
    assert archive["schema"] == "mx_gemmini.derived_square_128_vpu_pair_archive.v1"
    assert archive["compiler_revision"] == COMPILER
    saved = archive["cases"][case]
    assert saved["profile"] == profile_name
    assert len(saved["files"]) == 31
    root = EVIDENCE / case
    for name, descriptor in saved["files"].items():
        content = _read(root, name)
        assert (len(content), _sha(content)) == (
            descriptor["bytes"], descriptor["sha256"]), name

    index = json.loads(_read(root, "index.json"))
    capture = json.loads(_read(root, "capture/receipt.json"))
    obj = json.loads(_read(root, "object/object_manifest.json"))
    dispatch = json.loads(_read(root, "object/compile_manifest.json"))
    facts = json.loads(_read(root, "resources/facts.json"))
    physical = json.loads(_read(root, "object/physical_program.json.gz"))
    profile = load_profile(PROFILES / f"{profile_name}.json")
    assert index["schema"] == "mx_gemmini.derived_square_128_vpu_pair_spike.v1"
    assert index["status"] == "source_derived_square_128_vpu_pair_matched_on_pinned_spike"
    assert index["source_scope"] == facts["source_scope"]
    assert index["compiler_revision"] == capture["compiler_revision"] == (
        obj["compiler_revision"]) == COMPILER
    assert index["model2mlir_revision"] == capture["model2mlir_revision"] == MODEL2MLIR
    assert index["rtl_revision"] == obj["rtl_revision"] == RTL
    assert index["profile_sha256"] == obj["profile_sha256"] == profile_sha256(profile)
    assert index["first_shape_mnk"] == index["second_shape_mnk"] == (
        obj["shape_mnk"]) == physical["shape_mnk"] == [128, 128, 128]
    assert [site["shape"] for site in capture["sites"]] == [
        [128, 128, 128], [128, 128, 128]]
    assert capture["opaque_calls"] == {}
    assert dispatch["lowering_family"] == "resident_vpu_pair"
    assert dispatch["native_verifier_sha256"] is not None
    assert obj["schema"] == "mx_gemmini.resident_vpu_linkable_object.v1"
    assert obj["transport"] == "rocket_rocc"
    assert (obj["allocated_data_section_bytes"], obj["embedded_operand_bytes"],
            obj["embedded_golden_bytes"]) == (0, 0, 0)
    assert (index["compared_c1_bf16_values"], index["compared_c1_fp8_codes"],
            index["compared_c1_e8m0_scales"], index["compared_c2_fp8_codes"],
            index["compared_c2_e8m0_scales"]) == (
                16384, 16384, 512, 16384, 512)
    assert index["spike_exit_code"] == 0
    assert (b"lowered 128 MX/VPU: C1 BF16 0, C1 0 codes 0 scales, "
            b"C2 0 codes 0 scales") in _read(root, "build/spike.log")
    for artifact, digest in (
        ("connected.mlir.gz", index["bound_mlir_sha256"]),
        ("object/mx_issue.o.gz", index["object_sha256"]),
        ("build/mx_program.elf.gz", index["elf_sha256"]),
        ("build/spike.log", index["spike_log_sha256"]),
    ):
        assert _sha(_read(root, artifact)) == digest
    assert obj["issuer_c_sha256"] == _sha(_read(root, "object/mx_issue.c.gz"))
    assert obj["physical_program_sha256"] == _sha(
        _read(root, "object/physical_program.json.gz"))

    source = json.loads((PLAIN / "index.json").read_text())
    assert index["source_sha256"] == source["source_sha256"]
    assert index["header_sha256"] == source["header_sha256"]
    for slot in INPUTS:
        assert index["resources_sha256"][slot] == (
            source["files_sha256"][f"{slot}.bin"])
        assert obj["input_sha256"][slot] == index["resources_sha256"][slot]
    original_bf16 = _read(root, "resources/c1_bf16.bin.gz")
    assert quantize_bf16_fp8_output(original_bf16, 128, 128) == (
        _read(PLAIN, "c1_codes_ref.bin.gz"),
        _read(PLAIN, "c1_scales_ref.bin.gz"))
    assert quantize_bf16_fp8_output(exact_bf16_x2(original_bf16), 128, 128) == (
        _read(root, "resources/c1_codes_ref.bin.gz"),
        _read(root, "resources/c1_scales_ref.bin.gz"))


@pytest.mark.parametrize("case,profile_name", (
    ("fp4_vpu", "MxE4M3Fp4VpuGemminiRocketConfig"),
    ("e4m3_vpu", "MxE4M3VpuGemminiRocketConfig"),
))
def test_square_128_vpu_relowers_typed_graph_and_rejects_live_overlap(
        case: str, profile_name: str):
    root = EVIDENCE / case
    resources = {name: _read(root, f"resources/{name}.bin.gz") for name in INPUTS}
    graph = _read(root, "connected.mlir.gz").decode()
    profile = load_profile(PROFILES / f"{profile_name}.json")
    buffers = {name: name for name in INPUTS}
    outputs = {name: name for name in OUTPUTS}
    pair = lower_connected_fp8_vpu_pair(
        graph, profile, resources, buffers=buffers, outputs=outputs)
    assert (pair.first_width, pair.second_width, pair.c1_row, pair.c2_row) == (
        128, 128, 2048, 8192)
    commands = _readout_commands(pair, outputs)
    names = tuple(sorted({operand.buffer for command in commands
                          if isinstance(command, Command)
                          for operand in (command.rs1, command.rs2)
                          if operand.buffer is not None}))
    assert emit_c(commands, transport="rocket_rocc", buffers=names).encode() == (
        _read(root, "object/mx_issue.c.gz"))
    changed = dict(resources)
    changed["b2_weight"] = bytes([resources["b2_weight"][0] ^ 1]) + (
        resources["b2_weight"][1:])
    with pytest.raises(ValueError, match="input digest differs"):
        lower_connected_fp8_vpu_pair(
            graph, profile, changed, buffers=buffers, outputs=outputs)
    with pytest.raises(ValueError, match="placement or ABI differs"):
        lower_connected_fp8_vpu_pair(
            graph.replace("output_row = 8192 : i32", "output_row = 4096 : i32"),
            profile, resources, buffers=buffers, outputs=outputs)


def test_square_128_vpu_profiles_keep_separate_receipts():
    a = json.loads((EVIDENCE / "fp4_vpu/index.json").read_text())
    b = json.loads((EVIDENCE / "e4m3_vpu/index.json").read_text())
    assert a["profile_sha256"] != b["profile_sha256"]
    for key in ("bound_mlir_sha256", "object_sha256", "elf_sha256",
                "spike_log_sha256", "resources_sha256"):
        if key == "bound_mlir_sha256":
            assert a[key] != b[key]
        else:
            assert a[key] == b[key]
