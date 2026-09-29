"""Source-layout vectors for direct MXFP8 and MXFP4 transfers."""

import pytest

from mx_gemmini_support.operands import pack_direct_operands


def test_fp8_direct_bytes_are_logical_row_major():
    activation = [[(row * 32 + inner) % 256 for inner in range(32)] for row in range(16)]
    weight = [[(inner + 7 * col) % 256 for col in range(16)] for inner in range(32)]
    packed = pack_direct_operands("mxfp8", activation, weight)
    assert (packed.m, packed.k, packed.n) == (16, 32, 16)
    assert len(packed.activation_bytes) == len(packed.weight_bytes) == 512
    assert packed.activation_bytes[32] == activation[1][0]
    assert packed.activation_bytes[31] == activation[0][31]
    assert packed.weight_bytes[16] == weight[1][0]
    assert packed.weight_bytes[1] == weight[0][1]


def test_fp4_pairs_activation_rows_and_weight_columns():
    activation = [[row % 16 for _ in range(32)] for row in range(32)]
    weight = [[col % 16 for col in range(32)] for _ in range(32)]
    packed = pack_direct_operands("mxfp4", activation, weight)
    assert (packed.m, packed.k, packed.n) == (32, 32, 32)
    assert len(packed.activation_bytes) == len(packed.weight_bytes) == 512
    assert packed.activation_bytes[0] == 0x10
    assert packed.activation_bytes[32] == 0x32
    assert packed.weight_bytes[:3] == bytes.fromhex("10 32 54")
    assert packed.weight_bytes[16] == 0x10


def test_direct_packer_rejects_wrong_format_shape_and_code_width():
    a = [[0] * 32 for _ in range(32)]
    b = [[0] * 32 for _ in range(32)]
    with pytest.raises(ValueError, match="selected MXFP8 and MXFP4"):
        pack_direct_operands("mxfp6", a, b)
    with pytest.raises(ValueError, match="4-bit"):
        pack_direct_operands("mxfp4", [[16] * 32 for _ in range(32)], b)
    with pytest.raises(ValueError, match="4-bit"):
        pack_direct_operands("mxfp4", [[True] * 32 for _ in range(32)], b)
    with pytest.raises(ValueError, match="K must agree"):
        pack_direct_operands("mxfp4", a, b[:-1])
    with pytest.raises(ValueError, match="multiples of 32"):
        pack_direct_operands("mxfp4", a[:-16], b)
