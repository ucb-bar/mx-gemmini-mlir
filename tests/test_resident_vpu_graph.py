"""Self-contained structural checks for the typed MX+VPU object path."""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.resident_pair_graph import input_digest
from mx_gemmini_support.resident_vpu_graph import (INPUTS, OUTPUTS,
                                                   lower_connected_fp8_vpu_pair)
from mx_gemmini_support.target_profile import load_profile
from tools.emit_resident_vpu_object import _buffer_abi, _readout_commands


ROOT = Path(__file__).resolve().parents[1]
MLIR = (ROOT / "docs/evidence/nicolas_connected_chain_upstream_e9ded36_20261010"
        / "connected_bound.mlir")
PROFILE = (ROOT / "profiles/gemmini-mx-cleanup-266c593"
           / "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _fixture():
    resources = {
        slot: bytes((index * 13 + number) & 255 for index in range(length))
        for number, (slot, length) in enumerate((
            ("a1_activation", 4096), ("a1_scales", 128),
            ("b1_weight", 4096), ("b1_scales", 128),
            ("b2_weight", 4096), ("b2_scales", 128)))
    }
    buffers = {slot: slot for slot in INPUTS}
    outputs = {slot: slot for slot in OUTPUTS}
    digest = input_digest(resources, buffers)
    mlir = re.sub(r'(mx.runtime_resources_sha256 = ")[0-9a-f]{64}(")',
                  lambda match: match[1] + digest + match[2],
                  MLIR.read_text(), count=1)
    return mlir, load_profile(PROFILE), resources, buffers, outputs


def test_typed_vpu_pair_emits_complete_data_free_abi():
    mlir, profile, resources, buffers, outputs = _fixture()
    pair = lower_connected_fp8_vpu_pair(
        mlir, profile, resources, buffers=buffers, outputs=outputs)
    commands = _readout_commands(pair, outputs)
    abi = _buffer_abi(commands, buffers, outputs)
    assert (pair.first_site, pair.second_site) == (
        "functional:matmul", "functional:matmul_1")
    assert [command.funct for command in commands if isinstance(command, Command)].count(33) == 1
    assert [command.funct for command in commands if isinstance(command, Command)].count(34) == 1
    assert len(abi) == 11
    assert {entry["slot"] for entry in abi} == set(INPUTS) | set(OUTPUTS)
    assert {entry["minimum_bytes"] for entry in abi
            if entry["slot"] == "c1_bf16_observed"} == {8192}
    assert all(entry["alignment_bytes"] == 64 for entry in abi)


def test_vpu_graph_binds_runtime_symbols_and_site_ids_without_source_names():
    mlir, profile, resources, buffers, outputs = _fixture()
    renamed_inputs = {slot: f"runtime_{slot}" for slot in INPUTS}
    renamed_outputs = {slot: f"runtime_{slot}" for slot in OUTPUTS}
    renamed_resources = {renamed_inputs[slot]: resources[slot] for slot in INPUTS}
    mlir = mlir.replace("functional:matmul_1", "capture:second")
    mlir = mlir.replace("functional:matmul", "capture:first")
    for old, new in (("b2_weight", renamed_inputs["b2_weight"]),
                     ("b2_scales", renamed_inputs["b2_scales"]),
                     ("c1_scales", renamed_outputs["c1_scales"]),
                     ("c2_scales", renamed_outputs["c2_scales"])):
        mlir = mlir.replace(f'= "{old}"', f'= "{new}"')
    pair = lower_connected_fp8_vpu_pair(
        mlir, profile, renamed_resources,
        buffers=renamed_inputs, outputs=renamed_outputs)
    assert (pair.first_site, pair.second_site) == ("capture:first", "capture:second")
    abi = _buffer_abi(_readout_commands(pair, renamed_outputs),
                      renamed_inputs, renamed_outputs)
    assert {entry["name"] for entry in abi} == (
        set(renamed_inputs.values()) | set(renamed_outputs.values()))


def test_vpu_graph_rejects_wrong_edge_payload_and_vpu_mode():
    mlir, profile, resources, buffers, outputs = _fixture()
    lower = lambda text, data=resources: lower_connected_fp8_vpu_pair(
        text, profile, data, buffers=buffers, outputs=outputs)
    wrong_edge = mlir.replace(
        '"mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s)',
        '"mx_gemmini.resident_contract"(%c1, %c1s, %b2, %a1s)')
    with pytest.raises(ValueError, match="SSA tensor edges"):
        lower(wrong_edge)
    with pytest.raises(ValueError, match="input digest differs"):
        lower(mlir, resources | {"b2_weight": b"x" * 4096})
    with pytest.raises(ValueError, match="no BF16 immediate|in-place scalar operation"):
        lower(mlir.replace('kind = "muls"', 'kind = "mul"'))
    with pytest.raises(ValueError, match="placement or ABI differs"):
        lower(mlir.replace('scale_buffer = "c1_scales"',
                           'scale_buffer = "b2_scales"'))
