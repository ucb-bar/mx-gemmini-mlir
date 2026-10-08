"""Checked operand metadata uploads for a packed MX contraction.

Each wave is a separate physical transfer plan. The contraction scheduler must
still issue operand tiles and compute commands between these uploads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .command_ir import Command, Operand
from .contraction import MxContractionPayload
from .layout import lut_load_rs2, scale_load_rs2


@dataclass(frozen=True)
class WaveUploads:
    index: int
    buffers: dict[str, bytes]
    commands: tuple[Command, ...]


def plan_uploads(payload: MxContractionPayload, profile: Mapping) -> tuple[WaveUploads, ...]:
    """Bind scale and optional LUT DMA commands to one selected target profile."""
    if profile.get("schema") not in ("mx_gemmini.target_profile.v1", "radiance.soc_profile.v1"):
        raise ValueError("unsupported MX target profile")
    mx = profile.get("mx")
    if not isinstance(mx, dict) or payload.fmt not in mx.get("formats", []):
        raise ValueError(f"{payload.fmt} is absent from the selected MX profile")
    if payload.fmt == "mxfp6" and profile.get("schema") == "radiance.soc_profile.v1" and not mx.get("lut"):
        raise ValueError("selected MX profile has no FP6 LUT")
    if not payload.waves:
        raise ValueError("MX contraction has no scale waves")
    result = []
    for index, wave in enumerate(payload.waves):
        buffers = {
            f"a_scales_{index}": wave.activation_scale_bytes,
            f"b_scales_{index}": wave.weight_scale_bytes,
        }
        commands = []
        if index == 0 and payload.fmt == "mxfp6":
            a_lines, b_lines = payload.lut_lines_per_operand
            if not payload.activation_lut_bytes or not payload.weight_lut_bytes:
                raise ValueError("FP6 LUT bytes are missing")
            buffers.update(a_lut=payload.activation_lut_bytes,
                           b_lut=payload.weight_lut_bytes)
            commands.extend((
                Command(29, Operand(buffer="b_lut"),
                        Operand(immediate=lut_load_rs2(b_lines, operand="weight"))),
                Command(29, Operand(buffer="a_lut"),
                        Operand(immediate=lut_load_rs2(a_lines, operand="activation"))),
            ))
        commands.extend((
            Command(27, Operand(buffer=f"a_scales_{index}"),
                    Operand(immediate=scale_load_rs2(len(wave.activation_scale_bytes),
                                                     operand="activation"))),
            Command(27, Operand(buffer=f"b_scales_{index}"),
                    Operand(immediate=scale_load_rs2(len(wave.weight_scale_bytes),
                                                     operand="weight"))),
        ))
        result.append(WaveUploads(index, buffers, tuple(commands)))
    return tuple(result)
