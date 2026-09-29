"""The bringup emitter must reject unsupported programs and select RTL controls."""

import pytest

from mx_gemmini_support.contraction import plan_mx_contraction_payload
from mx_gemmini_support.diagnostic_program import emit_single_window_baremetal_c


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
    with pytest.raises(ValueError, match="one 32x32x32 scale window"):
        emit_single_window_baremetal_c(larger, [[0x4200] * 32 for _ in range(32)])
