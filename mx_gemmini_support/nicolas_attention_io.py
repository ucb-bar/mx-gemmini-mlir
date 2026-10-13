"""Source-equivalent MX scale, operand, and readout packets for dense attention."""

from __future__ import annotations

from .command_ir import Command, Fence
from .memory_phase import (lower_linear_spad_mvout, lower_matrix_mvin,
                           lower_scale_load, lower_strided_scale_dma)
from .nicolas_attention_shape import DenseAttentionShape


ATTENTION_BUFFERS = (
    "Q_SCALES", "KT_SCALES", "V_SCALES", "Q_IN", "KT_IN", "V_IN",
    "scale_sink", "S_hw", "p_scales", "O_hw", "P_hw", "p_codes_hw")


def _profile_rows(profile: dict, shape: DenseAttentionShape) -> int:
    if (profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_rows") != 16 or
            profile.get("geometry", {}).get("mesh_columns") != 16 or
            not profile.get("resources", {}).get("vpu") or
            not profile.get("resources", {}).get("spad_requant")):
        raise ValueError("dense attention I/O requires the DIM16 MX+VPU Rocket profile")
    rows = profile["resources"]["scratchpad_bytes"] // 16
    if rows < shape.end_row:
        raise ValueError("dense attention scratchpad is too small")
    shape.validate(profile)
    return rows


def lower_dense_attention_scales(
        profile: dict, shape: DenseAttentionShape = DenseAttentionShape(32, 64, 64)
        ) -> tuple[Command, ...]:
    """Load Q and Kᵀ into scale half zero and V into weight half one."""
    _profile_rows(profile, shape)
    config = profile["resources"].get("scale_mem_config", {})
    if config.get("size_bytes") != 16384:
        raise ValueError("dense attention scale memory differs from source")
    return (
        *lower_scale_load(profile, buffer="Q_SCALES",
                          payload_bytes=shape.buffer_bytes()["Q_SCALES"],
                          operand="activation").commands,
        *lower_scale_load(profile, buffer="KT_SCALES",
                          payload_bytes=shape.buffer_bytes()["KT_SCALES"],
                          operand="weight").commands,
        *lower_strided_scale_dma(profile, buffer="V_SCALES", row_bytes=shape.d,
                                 rows=shape.sk // 32, pitch_bytes=shape.d,
                                 destination_byte=4096, selector=1).commands,
    )


def lower_dense_attention_operands(
        profile: dict, shape: DenseAttentionShape = DenseAttentionShape(32, 64, 64)
        ) -> tuple[Command, ...]:
    """Tile Q, Kᵀ, and V with the source's DIM16 byte strides."""
    _profile_rows(profile, shape)
    return (
        *lower_matrix_mvin(profile, buffer="Q_IN", matrix_rows=shape.sq,
                           matrix_cols=shape.d, burst_cols=16, spad_row=0).commands,
        *lower_matrix_mvin(profile, buffer="KT_IN", matrix_rows=shape.d,
                           matrix_cols=shape.sk, burst_cols=16,
                           spad_row=shape.kt_row).commands,
        *lower_matrix_mvin(profile, buffer="V_IN", matrix_rows=shape.sk,
                           matrix_cols=shape.d, burst_cols=16,
                           spad_row=shape.v_row).commands,
    )


def lower_dense_attention_readout(
        profile: dict, stage: str,
        shape: DenseAttentionShape = DenseAttentionShape(32, 64, 64)
        ) -> tuple[Command, ...]:
    """Read one source scratchpad result as 16-byte DMA rows."""
    _profile_rows(profile, shape)
    outputs = {
        "score": ("S_hw", shape.score_row, shape.sq * shape.sk * 2 // 16),
        "probability": ("P_hw", shape.score_row, shape.sq * shape.sk * 2 // 16),
        "probability_codes": ("p_codes_hw", shape.probability_row,
                              shape.sq * shape.sk // 16),
        "output": ("O_hw", shape.output_row, shape.sq * shape.d * 2 // 16),
    }
    if stage not in outputs:
        raise ValueError("dense attention readout stage is unsupported")
    name, start, rows = outputs[stage]
    return lower_linear_spad_mvout(
        profile, buffer=name, total_bytes=rows * 16,
        tile_cols=16, spad_row=start).commands


def lower_dense_attention_program(mlir_text: str, profile: dict,
                                  binding: dict, *,
                                  phase: str = "whole") -> tuple[Command | Fence, ...]:
    """Assemble the source-bound matrix/I/O packets around typed VPU commands."""
    from .nicolas_attention_matrix import lower_dense_attention_matrix
    from .nicolas_attention_vector import lower_dense_attention_vector

    vector = lower_dense_attention_vector(mlir_text, profile, binding)
    shape = DenseAttentionShape.from_plan(binding["plan"])
    if phase not in {"whole", "setup", "check", "timed"}:
        raise ValueError("dense attention needs a selected issue phase")
    if phase == "setup":
        return lower_dense_attention_scales(profile, shape)
    program = (
        *((*lower_dense_attention_scales(profile, shape), Fence())
          if phase == "whole" else ()),
        *lower_dense_attention_operands(profile, shape),
        *lower_dense_attention_matrix(profile, "qk", shape=shape),
        *(lower_dense_attention_readout(profile, "score", shape)
          if phase in {"whole", "check"} else ()),
        *vector,
        *lower_dense_attention_matrix(profile, "pv", shape=shape),
        *lower_dense_attention_readout(profile, "output", shape),
        *(lower_dense_attention_readout(profile, "probability", shape)
          if phase in {"whole", "check"} else ()),
        *(lower_dense_attention_readout(profile, "probability_codes", shape)
          if phase in {"whole", "check"} else ()),
    )
    return program
