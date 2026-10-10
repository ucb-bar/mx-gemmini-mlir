"""Check source-bound FP6 resident edges, LUT reuse, and ABI sizes."""

from __future__ import annotations

import hashlib
import gzip
import json
import os
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.fp6_plain_chain import (
    lower_fp6_plain_chain, render_fp6_plain_chain, source_resources)
from mx_gemmini_support.resident_pair_graph import FP6_INPUTS, lower_connected_pair
from mx_gemmini_support.resident_pair_plan import plan_fp6_resident_pair
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.emit_resident_pair_object import _buffer_abi
from tools.qualify_nicolas_fp6_resident_chain import compiler_driver


ROOT = Path(__file__).resolve().parents[1]
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"
OUTPUTS = ("c1_scales", "c1_tiled_observed", "c2_scales", "c2_tiled")


def fixture(dimension: int = 64):
    software = RTL / "software/gemmini-rocc-tests"
    source_path = software / f"bareMetalC/matmul_tiled_fp6_{dimension}x{dimension}_chain.c"
    header_path = software / f"include/matmul_fp6_{dimension}x{dimension}_chain.h"
    if not source_path.is_file() or not header_path.is_file():
        pytest.skip("requires Nicolas's pinned FP6 resident source")
    profile = load_profile(PROFILE, rtl_root=RTL)
    source = source_path.read_text()
    resources = source_resources(source, header_path.read_text(), dimension=dimension)
    mlir = render_fp6_plain_chain(
        profile, resources,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        header_sha256=hashlib.sha256(header_path.read_bytes()).hexdigest(),
        dimension=dimension)
    return profile, resources, mlir, source


@pytest.mark.parametrize("dimension", (64, 128))
def test_fp6_source_bound_resident_pair(dimension: int):
    profile, resources, mlir, source = fixture(dimension)
    commands = lower_fp6_plain_chain(mlir, profile, resources, dimension=dimension)
    plan = plan_fp6_resident_pair(profile, shape=(dimension,) * 3,
                                  a_row=0, c1_row=2048, c2_row=4096)
    rows = dimension * dimension // 32
    assert (plan.a_rows, plan.b_rows, plan.c_rows, plan.b_row) == (
        rows, rows, rows, 16384 - rows)
    issued = [command for command in commands if isinstance(command, Command)]
    assert [command.funct for command in issued].count(8) == 2
    assert [command.funct for command in issued].count(26) == 2
    assert [command.funct for command in issued].count(27) == 3
    assert [command.funct for command in issued].count(29) == 6
    assert [command.rs1.buffer for command in issued if command.funct == 29] == [
        "b1_lut", "a1_lut", "c1_lut", "b2_lut", "c1_lut", "c2_lut"]
    assert [command.funct for command in issued[:3]] == [7, 0, 0]
    assert issued[-1].funct == 3
    pair = lower_connected_pair(
        mlir, profile, resources, buffers={name: name for name in FP6_INPUTS},
        c1_scales="c1_scales", c1_tiled_observed="c1_tiled_observed",
        c2_tiled="c2_tiled", precision="fp6_e3m2")
    abi = _buffer_abi(pair, {name: name for name in FP6_INPUTS},
                      {name: name for name in OUTPUTS}, precision="fp6_e3m2")
    sizes = {entry["slot"]: entry["minimum_bytes"] for entry in abi}
    assert {sizes[name] for name in FP6_INPUTS[6:]} == {dimension // 2 * 12}
    assert sizes["a1_activation"] == dimension * dimension // 2
    assert sizes["c1_tiled_observed"] == dimension * dimension // 2
    driver = compiler_driver(source, dimension=dimension)
    assert driver.count("mx_issue(") == 2
    assert driver.count('check_nibbles("C1"') == 1
    assert driver.count('check_nibbles("C2"') == 1


@pytest.mark.parametrize("dimension", (64, 128))
def test_fp6_resident_pair_rejects_lut_tampering(dimension: int):
    profile, resources, mlir, _ = fixture(dimension)
    with pytest.raises(ValueError):
        lower_fp6_plain_chain(
            mlir, profile, {**resources, "c1_lut": resources["c1_lut"][:-1]},
            dimension=dimension)
    with pytest.raises(ValueError):
        lower_fp6_plain_chain(
            mlir.replace('activation_lut_buffer = "c1_lut"',
                         'activation_lut_buffer = "a1_lut"'),
            profile, resources, dimension=dimension)
    with pytest.raises(ValueError):
        lower_fp6_plain_chain(
            mlir.replace('runtime_buffer = "c1_lut"',
                         'runtime_buffer = "c2_lut"', 1),
            profile, resources, dimension=dimension)
    with pytest.raises(ValueError):
        lower_fp6_plain_chain(
            mlir.replace('lut_groups = 32 : i32', 'lut_groups = 31 : i32') if dimension == 64
            else mlir.replace('lut_groups = 64 : i32', 'lut_groups = 63 : i32'),
            profile, resources, dimension=dimension)


@pytest.mark.parametrize("dimension", (64, 128))
def test_fp6_archived_spike_evidence(dimension: int):
    archive = (ROOT / "docs/evidence" /
               f"nicolas_fp6_connected_resident_{dimension}_266c593")
    index = json.loads((archive / "index.json").read_text())
    assert index["status"] == "source_and_compiler_matched_on_pinned_spike"
    assert index["fresh_remote_checkout_reproduced"] is True
    assert index["compared_fp6_codes_each_output"] == dimension * dimension
    assert index["compared_e8m0_scales_each_output"] == dimension * dimension // 32
    for filename, descriptor in index["files"].items():
        raw = (archive / filename).read_bytes()
        data = gzip.decompress(raw) if filename.endswith(".gz") else raw
        assert len(data) == descriptor["bytes"]
        assert hashlib.sha256(data).hexdigest() == descriptor["sha256"]
    receipt = json.loads((archive / "chain/receipt.json").read_text())
    obj = json.loads((archive / "object/object_manifest.json").read_text())
    replay = json.loads((archive / "fresh_checkout.json").read_text())
    assert receipt["source_spike"]["matched"]
    assert receipt["compiler_spike"]["matched"]
    assert receipt["compiler_revision"] == replay["compiler_revision"]
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["embedded_operand_bytes"] == 0
    assert obj["embedded_golden_bytes"] == 0
    assert obj["object_sha256"] == index["files"]["object/mx_issue.o.gz"]["sha256"]
    report = verify_ir((archive / "chain/connected.mlir").read_text(),
                       load_profile(PROFILE))
    assert (report["contracts"], report["resident_contracts"],
            report["runtime_luts"]) == (1, 1, 6)
