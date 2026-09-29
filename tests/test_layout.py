"""Independent boundary vectors for the selected RTL's layout equations."""

import pytest

from mx_gemmini_support.layout import (
    config_scale_mem_rs1,
    config_format_code,
    lut_load_rs2,
    pack_fp6_lut,
    pack_scale_rows,
    pack_wave_scales,
    plan_scale_waves,
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


def test_fp8_k_wave_plan_obeys_loop_field_before_scale_capacity():
    one = plan_scale_waves("mxfp8", tiles_i=1, tiles_j=1, k_tiles=510)
    assert [(w.k_tile_start, w.k_tiles, w.activation_rows, w.weight_rows) for w in one] == [
        (0, 510, 255, 255)
    ]
    split = plan_scale_waves("mxfp8", tiles_i=1, tiles_j=1, k_tiles=512)
    assert [(w.k_tile_start, w.k_tiles) for w in split] == [(0, 510), (510, 2)]
    assert all(w.activation_rows * 16 <= 4096 and w.weight_rows * 16 <= 4096 for w in split)
    with pytest.raises(ValueError, match="whole 32-element"):
        plan_scale_waves("mxfp8", tiles_i=1, tiles_j=1, k_tiles=3)


def test_quad_format_wave_payloads_restart_at_local_row_zero():
    waves = plan_scale_waves("mxfp6", tiles_i=4, tiles_j=2, k_tiles=66)
    assert [(w.k_tile_start, w.k_tiles, w.activation_rows, w.weight_rows) for w in waves] == [
        (0, 64, 128, 64), (64, 2, 4, 2)
    ]
    activation = [[block] * 32 for block in range(33) for _ in range(4)]
    weight = [[block + 50] * 32 for block in range(33) for _ in range(2)]
    act, wgt = pack_wave_scales(waves[1], activation, weight)
    assert len(act) == 128 and len(wgt) == 64
    assert act == bytes([32] * 128) and wgt == bytes([82] * 64)
    assert scale_physical_location("mxfp6", 0, 0).upload_offset == 0
    with pytest.raises(ValueError, match="full scale plan"):
        pack_wave_scales(waves[1], activation[:-1], weight)


def test_scale_wave_rejects_axis_that_cannot_fit_one_k_block():
    with pytest.raises(ValueError, match="active scale window"):
        plan_scale_waves("mxfp4", tiles_i=129, tiles_j=1, k_tiles=2)
    with pytest.raises(ValueError, match="9-bit"):
        plan_scale_waves("mxfp8", tiles_i=512, tiles_j=1, k_tiles=2)


def test_scale_mem_config_bitfields_follow_execute_controller():
    rs1 = config_scale_mem_rs1(
        tiles_i=3, tiles_j=5, k_tiles=8,
        activation_buffer=1, weight_buffer=0,
        reset_requantizer=True, resident=True,
        output_scale_address=0x1234,
    )
    assert rs1 & ((1 << 33) - 1) == 0x1234
    assert (rs1 >> 33) & 511 == 3
    assert (rs1 >> 42) & 511 == 5
    assert (rs1 >> 51) & 511 == 8
    assert [(rs1 >> bit) & 1 for bit in (60, 61, 62, 63)] == [1, 0, 1, 1]
    with pytest.raises(ValueError, match="9-bit"):
        config_scale_mem_rs1(tiles_i=1, tiles_j=1, k_tiles=512)
