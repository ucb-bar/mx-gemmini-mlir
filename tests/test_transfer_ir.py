from __future__ import annotations

import pytest

from mx_gemmini_support.command_ir import emit_c
from mx_gemmini_support.contraction import plan_mx_contraction_payload
from mx_gemmini_support.transfer_ir import plan_uploads


def _payload(fmt: str):
    code = {"mxfp8": 0x38, "mxfp6": 0x0c, "mxfp4": 0x02}[fmt]
    lut = [[0, code] + [0] * 14 for _ in range(16)] if fmt == "mxfp6" else None
    return plan_mx_contraction_payload(
        fmt, [[code] * 64 for _ in range(32)],
        [[code] * 32 for _ in range(64)],
        [[127, 127] for _ in range(32)], [[127, 127] for _ in range(32)],
        activation_lut=lut, weight_lut=lut, max_blocks_per_wave=1,
    )


def test_profile_bound_wave_uploads_share_fields_across_transports():
    uploads = plan_uploads(_payload("mxfp8"), {
        "schema": "radiance.soc_profile.v1", "mx": {"formats": ["mxfp8"], "lut": False}
    })
    assert len(uploads) == 2
    assert [len(w.commands) for w in uploads] == [2, 2]
    assert uploads[1].commands[1].rs2.immediate == (1 << 32) | 32
    rocket = emit_c(list(uploads[1].commands), transport="rocket_rocc",
                    buffers=tuple(uploads[1].buffers))
    muon = emit_c(list(uploads[1].commands), transport="muon_mmio",
                  buffers=tuple(uploads[1].buffers))
    assert "a_scales_1" in rocket and "a_scales_1" in muon
    assert "b_scales_1" in rocket and "b_scales_1" in muon
    assert "MX Rocket commands require RV64" in rocket


def test_profile_rejects_absent_format_and_fp6_loads_lut_once():
    fp6 = _payload("mxfp6")
    with pytest.raises(ValueError, match="absent"):
        plan_uploads(fp6, {"schema": "radiance.soc_profile.v1", "mx": {"formats": ["mxfp8"]}})
    with pytest.raises(ValueError, match="no FP6 LUT"):
        plan_uploads(fp6, {"schema": "radiance.soc_profile.v1", "mx": {"formats": ["mxfp6"], "lut": False}})
    uploads = plan_uploads(fp6, {
        "schema": "mx_gemmini.target_profile.v1", "mx": {"formats": ["mxfp8", "mxfp6", "mxfp4"]}
    })
    assert [command.funct for command in uploads[0].commands] == [29, 29, 27, 27]
    assert [command.funct for command in uploads[1].commands] == [27, 27]
