"""The two dense attention matrix stages use the selected source packet ABI."""

from pathlib import Path

import pytest

from mx_gemmini_support.nicolas_attention_matrix import lower_dense_attention_matrix
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "profiles/gemmini-mx-cleanup-266c593"


def test_qk_and_pv_matrix_packets_match_source_geometry() -> None:
    profile = load_profile(PROFILE_DIR / "MxE4M3Fp4VpuGemminiRocketConfig.json")
    qk = lower_dense_attention_matrix(profile, "qk")
    pv = lower_dense_attention_matrix(profile, "pv")
    assert [packet.funct for packet in qk] == [0, 0, 26, 9, 24, 8]
    assert [packet.funct for packet in pv] == [0, 0, 26, 9, 24, 8]
    assert qk[0].rs1.immediate == pv[0].rs1.immediate == 0x1c004
    assert qk[1].rs2.immediate == pv[1].rs2.immediate == 128
    assert qk[2].rs1.buffer == pv[2].rs1.buffer == "scale_sink"
    assert qk[2].rs1.or_bits >> 61 == 0
    assert pv[2].rs1.or_bits >> 61 == 1
    assert qk[3].rs2.immediate == pv[3].rs2.immediate == (4 << 32) | (4 << 16) | 2
    assert (qk[4].rs1.immediate, qk[4].rs2.immediate) == (0, 384)
    assert (pv[4].rs1.immediate, pv[4].rs2.immediate) == (640, 1024)
    assert qk[5].rs2.immediate == (384 << 32) | 0x238
    assert pv[5].rs2.immediate == (1024 << 32) | 0x238


def test_attention_matrix_rejects_other_stage_and_plain_profile() -> None:
    vpu = load_profile(PROFILE_DIR / "MxE4M3Fp4VpuGemminiRocketConfig.json")
    plain = load_profile(PROFILE_DIR / "MxGemminiRocketConfig.json")
    with pytest.raises(ValueError, match="qk or pv"):
        lower_dense_attention_matrix(vpu, "mlp")
    with pytest.raises(ValueError, match=r"MX\+VPU"):
        lower_dense_attention_matrix(plain, "qk")
