"""Check physical row-major BF16 writes across multi-output MX tiles."""

from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path

import pytest

from mx_gemmini_support.bind_payload import bind_payload, select_bf16_output_layout
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.emit_mx_object import _referenced_buffers


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / ("profiles/gemmini-mx-cleanup-266c593/"
                  "MxE4M3Fp4VpuGemminiRocketConfig.json")
FP4 = ROOT / "docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593"
FP8 = ROOT / "docs/evidence/radiance_fp8_512_tk256_latest_266c593"


def _check_full_row_major_coverage(program) -> None:
    m, n, _ = program.shape
    coverage: Counter[int] = Counter()
    output = [step.command for step in program.steps
              if step.phase == "readout" and
              getattr(getattr(step.command, "rs1", None), "buffer", None) ==
              "output_bf16"]
    assert output
    for command in output:
        rows = (command.rs2.immediate >> 48) & 0xffff
        cols = (command.rs2.immediate >> 32) & 0xffff
        assert 1 <= rows <= 16 and cols == 16
        for physical_row in range(rows):
            for byte in range(cols):
                coverage[command.rs1.byte_offset + physical_row * cols + byte] += 1
    assert len(coverage) == m * n * 2
    assert set(coverage.values()) == {1}
    assert set(coverage) == set(range(m * n * 2))


def test_fp4_four_output_tiles_lower_to_logical_row_major(tmp_path):
    profile = load_profile(PROFILE)
    manifest, resources = load_bundle(FP4 / "bundle")
    bound = (FP4 / "tilewise_bound.mlir").read_text()
    default = lower_bound_source(bound, profile, manifest, resources)
    archived = json.loads(gzip.decompress(
        (FP4 / "build/physical_program.json.gz").read_bytes()))
    assert default.receipt() == archived
    selected_tile_major = select_bf16_output_layout(
        bound, profile, manifest, layout="output_tile_major_bf16")
    assert lower_bound_source(selected_tile_major, profile, manifest,
                              resources).receipt() == default.receipt()

    selected = select_bf16_output_layout(bound, profile, manifest)
    assert verify_ir(selected, profile)["contracts"] == 1
    program = lower_bound_source(selected, profile, manifest, resources)
    assert program.plan["bf16_output_layout"] == "row_major_bf16"
    assert len(program.plan["output_tiles"]) == 4
    assert sum(step.phase == "vpu" for step in program.steps) == 4
    _check_full_row_major_coverage(program)
    assert _referenced_buffers(program, manifest)[2]["layout"] == "row_major_bf16"
    readouts = [step.command for step in program.steps
                if step.phase == "readout" and
                getattr(getattr(step.command, "rs1", None), "buffer", None) ==
                "output_bf16"]
    assert len(readouts) == 512
    assert [readouts[index].rs1.byte_offset for index in (0, 1, 128, 129)] == [
        0, 512, 256, 768]
    write_standalone_sources(tmp_path / "standalone", program, resources)
    driver = (tmp_path / "standalone/mx_driver.c").read_text()
    assert "if (got[i] != expected[i])" in driver
    assert "uint32_t tile =" not in driver

    with pytest.raises(ValueError, match="unselected"):
        select_bf16_output_layout(selected, profile, manifest)
    with pytest.raises(ValueError, match="unknown"):
        select_bf16_output_layout(bound, profile, manifest, layout="blocked")
    with pytest.raises(ValueError, match="memory layout"):
        verify_ir(selected.replace('memory_layout = "row_major_bf16"',
                                   'memory_layout = "interleaved"'), profile)


def test_fp8_retiling_to_64_columns_uses_partial_readout_transfers():
    profile = load_profile(PROFILE)
    original, resources = load_bundle(FP8 / "bundle")
    manifest = {**original, "tile_mnk": [64, 64, 256]}
    bound = bind_payload((FP8 / "profile_bound.mlir").read_text(), profile,
                         manifest)
    selected = select_bf16_output_layout(bound, profile, manifest)
    program = lower_bound_source(selected, profile, manifest, resources)
    assert len(program.plan["output_tiles"]) == 4
    _check_full_row_major_coverage(program)
    readouts = [step.command for step in program.steps
                if step.phase == "readout" and
                getattr(getattr(step.command, "rs1", None), "buffer", None) ==
                "output_bf16"]
    assert len(readouts) == 256
    assert {(command.rs2.immediate >> 48) & 0xffff for command in readouts} == {8}
    assert [readouts[index].rs1.byte_offset for index in (0, 1, 64, 65)] == [
        0, 256, 128, 384]
