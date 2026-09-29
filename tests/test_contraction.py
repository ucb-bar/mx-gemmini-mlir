"""Checks that operand K slices and scale waves stay aligned."""

import pytest

from mx_gemmini_support.contraction import plan_mx_contraction_payload


def test_direct_fp4_payload_slices_operands_and_scales_together():
    m, n, blocks = 64, 32, 65
    k = blocks * 32
    activation = [[(row + inner) % 16 for inner in range(k)] for row in range(m)]
    weight = [[(inner + col) % 16 for col in range(n)] for inner in range(k)]
    a_scales = [[block + row for block in range(blocks)] for row in range(m)]
    b_scales = [[100 + block + col for block in range(blocks)] for col in range(n)]
    payload = plan_mx_contraction_payload("mxfp4", activation, weight, a_scales, b_scales)

    assert (payload.m, payload.k, payload.n) == (m, k, n)
    assert [(p.wave.block_start, p.wave.block_stop) for p in payload.waves] == [(0, 64), (64, 65)]
    first, last = payload.waves
    assert len(first.activation_bytes) == (m // 2) * (64 * 32)
    assert len(first.weight_bytes) == (64 * 32) * (n // 2)
    assert len(last.activation_bytes) == (m // 2) * 32
    assert len(last.weight_bytes) == 32 * (n // 2)
    assert last.activation_bytes[0] == 0x10  # k=2048, rows 0/1
    assert last.weight_bytes[0] == 0x10  # k=2048, columns 0/1
    assert last.activation_scale_bytes[:32] == bytes(64 + row for row in range(32))
    assert last.weight_scale_bytes == bytes(164 + col for col in range(32))


def test_fp6_payload_requires_exact_codebooks_and_keeps_one_lut_upload():
    a_codes = [[0] * 64 for _ in range(32)]
    b_codes = [[0] * 32 for _ in range(64)]
    a_scales = [[104, 104] for _ in range(32)]
    b_scales = [[104] * 32 for _ in range(2)]  # [G][N] functional contraction
    lines = [[0] * 16 for _ in range(16)]
    payload = plan_mx_contraction_payload(
        "mxfp6", a_codes, b_codes, a_scales, b_scales,
        weight_scale_layout="gn", activation_lut=lines, weight_lut=lines,
    )
    assert len(payload.waves) == 1
    assert payload.lut_lines_per_operand == (16, 16)
    assert len(payload.activation_lut_bytes) == len(payload.weight_lut_bytes) == 192
    assert payload.waves[0].activation_bytes == bytes(1024)
    assert payload.waves[0].weight_bytes == bytes(1024)
    with pytest.raises(ValueError, match="requires activation and weight codebooks"):
        plan_mx_contraction_payload("mxfp6", a_codes, b_codes, a_scales, b_scales, weight_scale_layout="gn")


def test_payload_rejects_mismatched_scale_shape():
    a_codes = [[0] * 64 for _ in range(16)]
    b_codes = [[0] * 16 for _ in range(64)]
    a_scales = [[104] for _ in range(16)]  # one block, while operands have two
    b_scales = [[104] for _ in range(16)]
    with pytest.raises(ValueError, match="shapes disagree"):
        plan_mx_contraction_payload("mxfp8", a_codes, b_codes, a_scales, b_scales)
