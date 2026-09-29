"""Selected RTL LUT indexing and operand nibble-layout boundary vectors."""

import pytest

from mx_gemmini_support.fp6 import pack_fp6_indexed_contraction
from mx_gemmini_support.layout import lut_load_rs2


def test_exact_fp6_projection_and_operand_packing():
    m = n = k = 32
    a_lut = [[(group * 3 + i) % 64 for i in range(16)] for group in range(m // 2)]
    b_lut = [[(group * 5 + i) % 64 for i in range(16)] for group in range(n // 2)]
    a_codes = [[a_lut[row // 2][(row + inner) % 16] for inner in range(k)] for row in range(m)]
    b_codes = [[b_lut[col // 2][(inner + col) % 16] for col in range(n)] for inner in range(k)]

    result = pack_fp6_indexed_contraction(
        a_codes, b_codes, activation_lut=a_lut, weight_lut=b_lut, granularity_shift=1
    )
    assert (result.m, result.k, result.n) == (32, 32, 32)
    assert len(result.activation_bytes) == len(result.weight_bytes) == 512
    # A: [M/2][K], byte low/even M and high/odd M at the same K.
    assert result.activation_bytes[0] == 0x10
    assert result.activation_bytes[1] == 0x21
    assert result.activation_bytes[32] == 0x32
    # B: [K][N/2], byte low/even N and high/odd N at the same K.
    assert result.weight_bytes[0] == 0x10
    assert result.weight_bytes[1] == 0x32
    assert result.weight_bytes[16] == 0x21
    assert len(result.activation_lut_bytes) == len(result.weight_lut_bytes) == 192
    assert lut_load_rs2(result.activation_lut_lines, operand="activation") == (6 << 34) | (1 << 32) | 16


def test_selected_spatial_boundary_and_line_capacity():
    lines = [[i] * 16 for i in range(64)]
    a_codes = [[row // 2] * 32 for row in range(128)]
    b_codes = [[col // 2 for col in range(128)] for _ in range(32)]
    result = pack_fp6_indexed_contraction(
        a_codes, b_codes, activation_lut=lines, weight_lut=lines
    )
    assert result.activation_lut_lines == result.weight_lut_lines == 64
    assert result.activation_bytes[-1] == 0
    assert result.weight_bytes[-1] == 0
    with pytest.raises(ValueError, match="within 128"):
        pack_fp6_indexed_contraction(
            a_codes + [[0] * 32] * 32, b_codes, activation_lut=lines, weight_lut=lines
        )
    with pytest.raises(ValueError, match="at most 64"):
        pack_fp6_indexed_contraction(
            a_codes, b_codes, activation_lut=lines, weight_lut=lines, granularity_shift=0
        )


def test_missing_code_is_rejected_instead_of_requantized():
    lines = [[0] * 16 for _ in range(16)]
    a_codes = [[0] * 32 for _ in range(32)]
    b_codes = [[0] * 32 for _ in range(32)]
    a_codes[1][0] = 1
    with pytest.raises(ValueError, match="activation code 1 missing from LUT line 0"):
        pack_fp6_indexed_contraction(a_codes, b_codes, activation_lut=lines, weight_lut=lines)
    a_codes[1][0] = 0
    b_codes[0][3] = 1
    with pytest.raises(ValueError, match="weight code 1 missing from LUT line 1"):
        pack_fp6_indexed_contraction(a_codes, b_codes, activation_lut=lines, weight_lut=lines)
