"""Check compiler accumulator commands against the pinned DRAMMvout C paths."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from mx_gemmini_support.accumulator_readback import bind_accumulator_dram_readout
from mx_gemmini_support.bind_payload import bind_payload, select_bf16_output_layout
from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"
EVIDENCE = ROOT / "docs/evidence/nicolas_dram_mvout_spike_fallback_public_6ed3fcb_266c593"
RTL = Path(os.environ["MX_RTL_ROOT"]) if os.environ.get("MX_RTL_ROOT") else None
CASES = (
    ("fp8_64x64x64_dram_mvout_spike", "matmul_tiled_fp8_64x64_DRAMMvout.c",
     128, (0, 2048, 4096, 6144), 1),
    ("fp4_64x64x64_dram_mvout_spike", "matmul_tiled_fp4_64x64_DRAMMvout.c",
     256, (0, 64, 4096, 4160), 1),
    ("fp8_128x128x256_dram_mvout_spike", "matmul_tiled_fp8_128x128x256_DRAMMvout.c",
     256, tuple(i * 4096 + j * 128 for i in range(8) for j in range(2)), 2),
)


def _bound(case: str):
    directory = EVIDENCE / case
    profile = load_profile(PROFILE)
    manifest, resources = load_bundle(directory / "bundle")
    text = bind_handoff((directory / "handoff.mlir").read_text(), profile)
    text = bind_payload(text, profile, manifest)
    text = select_bf16_output_layout(text, profile, manifest)
    return bind_accumulator_dram_readout(text, profile, manifest), profile, manifest, resources


@pytest.mark.parametrize("case,source_name,stride,offsets,waves", CASES)
def test_accumulator_commands_follow_pinned_source_geometry(
        case: str, source_name: str, stride: int,
        offsets: tuple[int, ...], waves: int) -> None:
    selected, profile, manifest, resources = _bound(case)
    if RTL is not None:
        source = RTL / "software/gemmini-rocc-tests/bareMetalC" / source_name
        source_text = source.read_text()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_driver_sha256"]
        assert "uint32_t c_dest = acc_addr;" in source_text
        assert "uint32_t c_flag = 0xb8;" in source_text
        assert "gemmini_mvout((void *) dram_ptr, acc_tile_addr);" in source_text
        assert "sf_mem[i] = 0x7f7f7f7f7f7f7f7f;" in source_text
    assert verify_ir(selected, profile)["contracts"] == 1
    program = lower_bound_source(selected, profile, manifest, resources)
    assert program.mode == "rtl_accumulator"
    assert program.source_golden_preserving is False
    assert program.plan["acc_to_gmem"] is True
    assert program.plan["hardware_numerical_qualification"] == "unqualified"
    assert program.plan["scale_loading"] == "source_header_rocc_2d_not_nicolas_mmio_constant"
    store = [s.command for s in program.steps if s.phase == "configure" and
             getattr(s.command, "funct", None) == 0 and s.command.rs1.immediate == 2]
    assert len(store) == 1 and store[0].rs2.immediate == stride
    compute = [s.command for s in program.steps if s.phase == "compute" and
               getattr(s.command, "funct", None) == 8]
    assert len(compute) == waves
    assert {c.rs2.immediate for c in compute} == {(0x80000000 << 32) | 0x2b8}
    moves = [s.command for s in program.steps if s.phase == "readout" and
             getattr(s.command, "funct", None) == 3]
    assert tuple(c.rs1.byte_offset for c in moves) == offsets
    assert tuple(c.rs2.immediate & 0xffffffff for c in moves) == tuple(
        0x80000000 + i * 16 for i in range(len(offsets)))
    assert {(c.rs2.immediate >> 32) & 0xffff for c in moves} == {16}
    assert {(c.rs2.immediate >> 48) & 0xffff for c in moves} == {16}


def test_accumulator_route_rejects_unbound_or_wrong_source():
    selected, profile, manifest, resources = _bound(CASES[0][0])
    with pytest.raises(ValueError, match="unselected"):
        bind_accumulator_dram_readout(selected, profile, manifest)
    with pytest.raises(ValueError, match="pinned DRAMMvout"):
        lower_bound_source(selected, profile,
                           {**manifest, "source_driver_sha256": "0" * 64}, resources)
    with pytest.raises(ValueError, match="source memory"):
        lower_bound_source(selected.replace(', mx.accumulator_dram_readout = "fp8_64"', ""),
                           profile, manifest, resources)
    with pytest.raises(ValueError, match="unsupported source memory"):
        verify_ir(selected.replace('source_memory = "accumulator"',
                                   'source_memory = "host"'), profile)
    with pytest.raises(ValueError, match="row_major_bf16"):
        verify_ir(selected.replace('memory_layout = "row_major_bf16"',
                                   'memory_layout = "output_tile_major_bf16"'), profile)
