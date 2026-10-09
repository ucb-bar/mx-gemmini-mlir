"""RTL funct 33/34 packing and selected VPU/SPAD_REQUANT gates."""

from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import spad_requant_command, vpu_command
from mx_gemmini_support.target_profile import load_profile


PROFILES = Path(__file__).resolve().parents[1] / "profiles/gemmini-mx-cleanup-266c593"


def _profile(name):
    return load_profile(PROFILES / f"{name}.json")


def test_vpu_expsum_uses_selected_fused_gate_and_rtl_fields():
    profile = _profile("MxE4M3Fp4VpuGemminiRocketConfig")
    command = vpu_command(profile, kind="expsum", src1_row=2048, src2_row=4096,
                          dst_row=6144, rows=128, reduction_length=8,
                          broadcast=True, second_dst_row=7168)
    assert command.funct == 33
    assert command.rs1.immediate == (2048 | (4096 << 14) | (6144 << 28) | (128 << 42))
    assert command.rs2.immediate == (13 | (1 << 4) | (8 << 5) | (7168 << 16))
    assert profile["resources"]["vpu_config"] == {"units": 2, "exp_sub": True, "exp_sum": True}


def test_vector_ops_refuse_disabled_config_and_oob_rows():
    baseline = _profile("MxGemminiRocketConfig")
    with pytest.raises(ValueError, match="no VPU"):
        vpu_command(baseline, kind="add", src1_row=0, src2_row=1, dst_row=2, rows=1)
    vpu = _profile("MxE4M3VpuGemminiRocketConfig")
    with pytest.raises(ValueError, match="scratchpad"):
        vpu_command(vpu, kind="add", src1_row=16383, src2_row=0, dst_row=1, rows=2)
    with pytest.raises(ValueError, match="divide"):
        vpu_command(vpu, kind="expsum", src1_row=0, src2_row=1024, dst_row=2048,
                    rows=7, reduction_length=4, broadcast=True, second_dst_row=3072)


def test_spad_requant_packs_radiance_p_feed_fields():
    profile = _profile("MxE4M3Fp4VpuGemminiRocketConfig")
    command = spad_requant_command(profile, source_row=4096, destination_row=3072,
                                   m=64, n=128, output_format="fp8_e4m3", tiled=True,
                                   resident=True, scale_dram_address=0x10000000)
    assert command.funct == 34
    assert command.rs1.immediate == (4096 | (3072 << 14) | (1 << 28) | (1 << 29) |
                                     (0x10000000 << 30))
    assert command.rs2.immediate == (64 | (128 << 16))
    fp4 = spad_requant_command(profile, source_row=4096, destination_row=3072,
                               m=64, n=128, output_format="fp4_e2m1", tiled=True,
                               resident=True, scale_dram_address=0x10000000)
    assert fp4.rs2.immediate == command.rs2.immediate | (1 << 32)
    with pytest.raises(ValueError, match="multiple of 8"):
        spad_requant_command(profile, source_row=4096, destination_row=3072,
                             m=7, n=128, output_format="fp8_e4m3", tiled=True,
                             resident=True, scale_dram_address=0x10000000)
