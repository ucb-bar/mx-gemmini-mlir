"""The bringup emitter must reject unsupported programs and select RTL controls."""

import pytest

from mx_gemmini_support.contraction import plan_mx_contraction_payload
from mx_gemmini_support.diagnostic_program import (
    emit_single_window_baremetal_c,
    emit_two_wave_baremetal_c,
)


def _payload(fmt):
    code = {"mxfp8": 0x38, "mxfp6": 0x0c, "mxfp4": 0x02}[fmt]
    activation = [[code] * 32 for _ in range(32)]
    weight = [[code] * 32 for _ in range(32)]
    scales = [[127] for _ in range(32)]
    lut = [[0, code] + [0] * 14 for _ in range(16)] if fmt == "mxfp6" else None
    return plan_mx_contraction_payload(
        fmt, activation, weight, scales, scales,
        activation_lut=lut, weight_lut=lut,
    )


@pytest.mark.parametrize("fmt,code", [("mxfp8", 0), ("mxfp6", 1), ("mxfp4", 2)])
def test_emitter_selects_format_and_required_lut_uploads(fmt, code):
    source = emit_single_window_baremetal_c(_payload(fmt), [[0x4200] * 32 for _ in range(32)])
    assert f"false, {code}, {code}, 3, {int(fmt == 'mxfp6')});" in source
    assert "gemmini_mx_load_scales((uint64_t)A_scales" in source
    assert "gemmini_mx_load_scales((uint64_t)B_scales" in source
    assert "gemmini_loop_ws_spad(" in source
    if fmt == "mxfp6":
        assert "gemmini_mx_load_lut_dt((uint64_t)B_lut, 16, 0, 6);" in source
        assert "gemmini_mx_load_lut_dt((uint64_t)A_lut, 16, 1, 6);" in source
        assert "gemmini_mx_lut_disable" not in source
    else:
        assert "gemmini_mx_lut_disable();" in source
        assert "gemmini_mx_load_lut_dt" not in source


def test_emitter_fails_closed_for_unqualified_shape_and_oracle():
    payload = _payload("mxfp8")
    with pytest.raises(ValueError, match="expected matrix"):
        emit_single_window_baremetal_c(payload, [[0x4200] * 32])

    activation = [[0x38] * 64 for _ in range(32)]
    weight = [[0x38] * 32 for _ in range(64)]
    scales = [[127, 127] for _ in range(32)]
    larger = plan_mx_contraction_payload("mxfp8", activation, weight, scales, scales)
    with pytest.raises(ValueError, match="one square 32 or 64 scale window"):
        emit_single_window_baremetal_c(larger, [[0x4200] * 32 for _ in range(32)])


def test_emitter_64_loads_all_fp6_lut_lines():
    size = 64
    codes = [[0x0c] * size for _ in range(size)]
    scales = [[127, 127] for _ in range(size)]
    lut = [[0, 0x0c] + [0] * 14 for _ in range(size // 2)]
    payload = plan_mx_contraction_payload(
        "mxfp6", codes, codes, scales, scales,
        activation_lut=lut, weight_lut=lut,
    )
    source = emit_single_window_baremetal_c(payload, [[0x4280] * size for _ in range(size)])
    assert "#define M 64" in source
    assert "gemmini_mx_load_lut_dt((uint64_t)B_lut, 32, 0, 6);" in source
    assert "gemmini_mx_load_lut_dt((uint64_t)A_lut, 32, 1, 6);" in source


@pytest.mark.parametrize("fmt,one,two,four", [
    ("mxfp8", 0x38, 0x40, 0x48),
    ("mxfp6", 0x0c, 0x10, 0x14),
    ("mxfp4", 0x02, 0x04, 0x06),
])
def test_two_wave_emitter_reloads_scales_and_accumulates(fmt, one, two, four):
    activation = [
        [one if row < 16 else two] * 32 + [two if row < 16 else four] * 32
        for row in range(32)
    ]
    weight = [[one if col < 16 else two for col in range(32)] for _ in range(64)]
    scales = [[127, 127] for _ in range(32)]
    lut = [[0, one, two, four] + [0] * 12 for _ in range(16)] if fmt == "mxfp6" else None
    payload = plan_mx_contraction_payload(
        fmt, activation, weight, scales, scales,
        activation_lut=lut, weight_lut=lut, max_blocks_per_wave=1,
    )
    source = emit_two_wave_baremetal_c(payload, [[0x42c0] * 32 for _ in range(32)])
    assert source.count("gemmini_loop_ws_spad(") == 2
    assert source.count("gemmini_mx_load_scales((uint64_t)AS") == 2
    assert "false, false, false, false, true, NO_ACTIVATION" in source
    assert "two-wave" in source
    if fmt == "mxfp6":
        assert source.count("gemmini_mx_load_lut_dt(") == 2  # one A and one B upload
    with pytest.raises(ValueError, match="two-wave diagnostic requires"):
        emit_two_wave_baremetal_c(
            plan_mx_contraction_payload(
                fmt, activation, weight, scales, scales,
                activation_lut=lut, weight_lut=lut,
            ),
            [[0x42c0] * 32 for _ in range(32)],
        )
