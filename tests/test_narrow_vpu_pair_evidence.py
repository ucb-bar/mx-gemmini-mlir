"""Fresh model2MLIR → MX/VPU object → Nicolas Spike numerical evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.resident_pair_graph import INPUTS
from mx_gemmini_support.resident_vpu_graph import OUTPUTS, lower_connected_fp8_vpu_pair
from mx_gemmini_support.target_profile import load_profile
from tools.emit_resident_vpu_object import _readout_commands


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_narrow_vpu_pair_64x64x64_64x32x64_9cd918c"
PROFILE = (ROOT / "profiles/gemmini-mx-cleanup-266c593"
           / "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _read(name: str) -> bytes:
    data = (EVIDENCE / name).read_bytes()
    return gzip.decompress(data) if name.endswith(".gz") else data


def test_narrow_mx_vpu_source_parity_on_fresh_spike() -> None:
    index = json.loads(_read("index.json"))
    assert index["schema"] == "mx_gemmini.nicolas_narrow_vpu_pair_fresh_spike.v1"
    assert index["status"] == "source_mx_vpu_and_narrow_mm2_matched_on_pinned_spike"
    assert index["first_shape_mnk"] == [64, 64, 64]
    assert index["second_shape_mnk"] == [64, 32, 64]
    assert index["compared"] == {
        "c1_bf16_values": 4096, "c1_fp8_codes": 4096,
        "c1_e8m0_scales": 128, "c2_fp8_codes": 2048,
        "c2_e8m0_scales": 64}
    for name, descriptor in index["files"].items():
        data = _read(name)
        assert len(data) == descriptor["bytes"], name
        assert hashlib.sha256(data).hexdigest() == descriptor["sha256"], name
    capture = json.loads(_read("capture/receipt.json"))
    binding = json.loads(_read("bound/binding_manifest.json"))
    obj = json.loads(_read("object/object_manifest.json"))
    spike = json.loads(_read("spike/qualification_manifest.json"))
    assert capture["model2mlir_revision"] == index["model2mlir_revision"]
    assert [site["shape"] for site in capture["sites"]] == [
        [64, 64, 64], [64, 32, 64]]
    assert binding["bound_mlir_sha256"] == obj["bound_mlir_sha256"] == (
        spike["bound_mlir_sha256"])
    assert obj["shape_mnk"] == [64, 32, 64]
    assert (obj["allocated_data_section_bytes"], obj["embedded_operand_bytes"],
            obj["embedded_golden_bytes"]) == (0, 0, 0)
    abi = {entry["slot"]: entry["minimum_bytes"] for entry in obj["buffer_abi"]}
    assert (abi["b1_weight"], abi["b2_weight"], abi["c2_tiled"],
            abi["c2_scales"]) == (4096, 2048, 2048, 64)
    assert spike["object_sha256"] == obj["object_sha256"]
    assert spike["spike_exit_code"] == 0
    assert (b"lowered narrow MX/VPU: C1 BF16 0, C1 0 codes 0 scales, "
            b"C2 0 codes 0 scales") in _read("spike/spike.log")


def test_narrow_mx_vpu_relowers_typed_graph_and_rejects_changed_payload() -> None:
    mlir = _read("bound/connected.mlir.gz").decode()
    resources = {slot: _read(f"bound/{slot}.bin.gz") for slot in INPUTS}
    buffers = {name: name for name in INPUTS}
    outputs = {name: name for name in OUTPUTS}
    profile = load_profile(PROFILE)
    pair = lower_connected_fp8_vpu_pair(
        mlir, profile, resources, buffers=buffers, outputs=outputs)
    assert pair.second_width == 32
    commands = _readout_commands(pair, outputs)
    names = tuple(sorted({operand.buffer for command in commands
                          if isinstance(command, Command)
                          for operand in (command.rs1, command.rs2)
                          if operand.buffer is not None}))
    assert emit_c(commands, transport="rocket_rocc", buffers=names).encode() == (
        _read("object/mx_issue.c.gz"))
    changed = dict(resources)
    changed["b2_weight"] = bytes([changed["b2_weight"][0] ^ 1]) + changed["b2_weight"][1:]
    with pytest.raises(ValueError, match="input digest differs"):
        lower_connected_fp8_vpu_pair(
            mlir, profile, changed, buffers=buffers, outputs=outputs)
