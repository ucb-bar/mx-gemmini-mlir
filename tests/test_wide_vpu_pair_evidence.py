"""Audit published clean-checkout FP8 MX/VPU width expansion."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.resident_pair_graph import INPUTS
from mx_gemmini_support.resident_vpu_graph import OUTPUTS, lower_connected_fp8_vpu_pair
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.emit_resident_vpu_object import _readout_commands


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_wide_vpu_pair_266c593_a1c0b4d"
SOURCE = ROOT / "docs/evidence/nicolas_chain_pipelined_full_266c593/object_manifest.json"
PROFILE_DIR = ROOT / "profiles/gemmini-mx-cleanup-266c593"
COMPILER = "a1c0b4ded11a9fa2bed26d9088da443cc07e849a"
RTL = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
MODEL2MLIR = "e9ded36eb85abf2d9097ac4dc11457c825853388"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(case: Path, name: str) -> bytes:
    raw = (case / name).read_bytes()
    return gzip.decompress(raw) if name.endswith(".gz") else raw


@pytest.mark.parametrize("width,profile_name", (
    (96, "MxE4M3Fp4VpuGemminiRocketConfig"),
    (128, "MxE4M3VpuGemminiRocketConfig"),
))
def test_wide_vpu_clean_checkout_replays_every_output(width: int, profile_name: str):
    archive = json.loads((EVIDENCE / "archive_manifest.json").read_text())
    assert archive["schema"] == "mx_gemmini.derived_wide_vpu_pair_archive.v1"
    assert archive["compiler_revision"] == COMPILER
    case = EVIDENCE / f"n{width}"
    saved = archive["cases"][f"n{width}"]
    assert saved["profile"] == profile_name
    assert len(saved["files"]) == 31
    for name, descriptor in saved["files"].items():
        content = _read(case, name)
        assert (len(content), _sha(content)) == (
            descriptor["bytes"], descriptor["sha256"]), name

    index = json.loads(_read(case, "index.json"))
    capture = json.loads(_read(case, "capture/receipt.json"))
    obj = json.loads(_read(case, "object/object_manifest.json"))
    dispatch = json.loads(_read(case, "object/compile_manifest.json"))
    physical = json.loads(_read(case, "object/physical_program.json.gz"))
    facts = json.loads(_read(case, "resources/facts.json"))
    profile = load_profile(PROFILE_DIR / f"{profile_name}.json")
    assert index["schema"] == "mx_gemmini.derived_wide_vpu_pair_spike.v1"
    assert index["status"] == "source_derived_wide_vpu_pair_matched_on_pinned_spike"
    assert index["source_scope"] == facts["source_scope"]
    assert index["compiler_revision"] == capture["compiler_revision"] == (
        obj["compiler_revision"]) == COMPILER
    assert index["rtl_revision"] == obj["rtl_revision"] == RTL
    assert index["model2mlir_revision"] == capture["model2mlir_revision"] == MODEL2MLIR
    assert index["model_sha256"] == facts["model_sha256"]
    assert index["profile_sha256"] == obj["profile_sha256"] == profile_sha256(profile)
    assert index["second_shape_mnk"] == obj["shape_mnk"] == (
        physical["shape_mnk"]) == [64, width, 64]
    assert [site["shape"] for site in capture["sites"]] == [
        [64, 64, 64], [64, width, 64]]
    assert capture["opaque_calls"] == {}
    assert dispatch["lowering_family"] == "resident_vpu_pair"
    assert obj["schema"] == "mx_gemmini.resident_vpu_linkable_object.v1"
    assert obj["transport"] == "rocket_rocc"
    assert (obj["allocated_data_section_bytes"], obj["embedded_operand_bytes"],
            obj["embedded_golden_bytes"]) == (0, 0, 0)
    assert (index["compared_c1_bf16_values"], index["compared_c1_fp8_codes"],
            index["compared_c1_e8m0_scales"], index["compared_c2_fp8_codes"],
            index["compared_c2_e8m0_scales"]) == (
                4096, 4096, 128, 64 * width, 2 * width)
    assert index["spike_exit_code"] == 0
    assert (b"C1 BF16 0, C1 0 codes 0 scales, C2 0 codes 0 scales" in
            _read(case, "build/spike.log"))
    for artifact, digest in (
        ("connected.mlir.gz", index["bound_mlir_sha256"]),
        ("object/mx_issue.o.gz", index["object_sha256"]),
        ("build/mx_program.elf.gz", index["elf_sha256"]),
        ("build/spike.log", index["spike_log_sha256"]),
    ):
        assert _sha(_read(case, artifact)) == digest
    assert obj["issuer_c_sha256"] == _sha(_read(case, "object/mx_issue.c.gz"))
    assert obj["physical_program_sha256"] == _sha(
        _read(case, "object/physical_program.json.gz"))

    source = json.loads(SOURCE.read_text())["source_resource_sha256"]
    for slot in ("a1_activation", "a1_scales", "b1_weight", "b1_scales",
                 "c1_bf16"):
        assert index["resources_sha256"][slot] == source[slot]
    for slot in INPUTS:
        assert obj["input_sha256"][slot] == index["resources_sha256"][slot]
    # The extra columns are an explicit transformation of Nicolas's B2 bytes.
    weights = _read(case, "resources/b2_weight.bin.gz")
    scales = _read(case, "resources/b2_scales.bin.gz")
    original_weights = b"".join(
        weights[row * width:row * width + 64] for row in range(64))
    original_scales = b"".join(
        scales[group * width:group * width + 64] for group in range(2))
    assert _sha(original_weights) == source["b2_weight"]
    assert _sha(original_scales) == source["b2_scales"]
    for row in range(64):
        base = weights[row * width:row * width + 64]
        extra = weights[row * width + 64:(row + 1) * width]
        assert extra == bytes(code ^ 0x80 for code in base[:width - 64])
    for group in range(2):
        base = scales[group * width:group * width + 64]
        assert scales[group * width + 64:(group + 1) * width] == base[:width - 64]


@pytest.mark.parametrize("width,profile_name", (
    (96, "MxE4M3Fp4VpuGemminiRocketConfig"),
    (128, "MxE4M3VpuGemminiRocketConfig"),
))
def test_wide_vpu_archived_typed_graph_relowers_and_rejects_payload_change(
        width: int, profile_name: str):
    case = EVIDENCE / f"n{width}"
    resources = {name: _read(case, f"resources/{name}.bin.gz") for name in INPUTS}
    graph = _read(case, "connected.mlir.gz").decode()
    profile = load_profile(PROFILE_DIR / f"{profile_name}.json")
    buffers = {name: name for name in INPUTS}
    outputs = {name: name for name in OUTPUTS}
    pair = lower_connected_fp8_vpu_pair(
        graph, profile, resources, buffers=buffers, outputs=outputs)
    assert pair.second_width == width
    commands = _readout_commands(pair, outputs)
    names = tuple(sorted({operand.buffer for command in commands
                          if isinstance(command, Command)
                          for operand in (command.rs1, command.rs2)
                          if operand.buffer is not None}))
    assert emit_c(commands, transport="rocket_rocc", buffers=names).encode() == (
        _read(case, "object/mx_issue.c.gz"))
    changed = dict(resources)
    changed["b2_weight"] = bytes([resources["b2_weight"][0] ^ 1]) + (
        resources["b2_weight"][1:])
    with pytest.raises(ValueError, match="input digest differs"):
        lower_connected_fp8_vpu_pair(
            graph, profile, changed, buffers=buffers, outputs=outputs)
