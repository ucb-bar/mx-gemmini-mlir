"""Independent boundary vectors for the selected RTL's layout equations."""

import pytest

from mx_gemmini_support.layout import (
    config_format_code,
    lut_load_rs2,
    pack_fp6_lut,
    pack_scale_rows,
    scale_load_rs2,
    scale_physical_location,
    scale_row_index,
)


def test_config_codes_and_k_block_row_order():
    assert [config_format_code(fmt) for fmt in ("mxfp8", "mxfp6", "mxfp4")] == [0, 1, 2]
    assert [scale_row_index(tile=2, k_tile=k, tiles_axis=4) for k in range(4)] == [2, 2, 6, 6]
    with pytest.raises(ValueError):
        scale_row_index(tile=4, k_tile=0, tiles_axis=4)


def test_scale_bank_crossings_match_selected_address_bits():
    assert scale_physical_location("mxfp8", 127, 15).bank == 0
    fp8_next = scale_physical_location("mxfp8", 128, 0)
    assert (fp8_next.bank, fp8_next.row, fp8_next.upload_offset) == (1, 0, 2048)
    assert scale_physical_location("mxfp8", 0, 0, buffer=1).bank == 2

    quad_first = scale_physical_location("mxfp6", 0, 16)
    assert (quad_first.bank, quad_first.row, quad_first.lane, quad_first.upload_offset) == (1, 0, 0, 16)
    quad_next = scale_physical_location("mxfp4", 1, 0)
    assert (quad_next.bank, quad_next.row, quad_next.upload_offset) == (0, 1, 32)
    assert scale_physical_location("mxfp6", 0, 16, buffer=1).bank == 3


def test_scale_upload_rejects_overflow_and_encodes_operand_selector():
    rows = [[127] * 16, [104] * 16]
    payload = pack_scale_rows("mxfp8", rows)
    assert payload == bytes([127] * 16 + [104] * 16)
    assert scale_load_rs2(len(payload), operand="activation") == 32
    assert scale_load_rs2(len(payload), operand="weight") == (1 << 32) | 32
    with pytest.raises(ValueError, match="4 KiB"):
        pack_scale_rows("mxfp6", [[127] * 32] * 129)
    with pytest.raises(ValueError, match="8-byte"):
        scale_load_rs2(12, operand="activation")


def test_fp6_lut_wire_order_and_dma_padding():
    line = [1, 2, 3, 4] + [0] * 12
    payload = pack_fp6_lut([line])
    assert payload[:3] == bytes.fromhex("81 30 10")
    assert len(payload) == 16
    assert payload[12:] == bytes(4)
    assert len(pack_fp6_lut([line, line])) == 24
    assert lut_load_rs2(1, operand="activation") == (6 << 34) | (1 << 32) | 1
    assert lut_load_rs2(1, operand="weight") == (6 << 34) | 1
    with pytest.raises(ValueError, match="64"):
        pack_fp6_lut([line] * 65)
    with pytest.raises(ValueError, match="6-bit"):
        pack_fp6_lut([[64] + [0] * 15])
