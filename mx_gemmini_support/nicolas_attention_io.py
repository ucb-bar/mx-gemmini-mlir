"""Source-equivalent MX scale, operand, and readout packets for dense attention."""

from __future__ import annotations

from .command_ir import Command, Fence
from .memory_phase import (lower_linear_spad_mvout, lower_matrix_mvin,
                           lower_scale_load, lower_strided_scale_dma)


ATTENTION_BUFFERS = (
    "Q_SCALES", "KT_SCALES", "V_SCALES", "Q_IN", "KT_IN", "V_IN",
    "scale_sink", "S_hw", "p_scales", "O_hw", "P_hw", "p_codes_hw")


def _profile_rows(profile: dict) -> int:
    if (profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_rows") != 16 or
            profile.get("geometry", {}).get("mesh_columns") != 16 or
            not profile.get("resources", {}).get("vpu") or
            not profile.get("resources", {}).get("spad_requant")):
        raise ValueError("dense attention I/O requires the DIM16 MX+VPU Rocket profile")
    rows = profile["resources"]["scratchpad_bytes"] // 16
    if rows < 1280:
        raise ValueError("dense attention scratchpad is too small")
    return rows


def lower_dense_attention_scales(profile: dict) -> tuple[Command, ...]:
    """Load Q and Kᵀ into scale half zero and V into weight half one."""
    _profile_rows(profile)
    config = profile["resources"].get("scale_mem_config", {})
    if config.get("size_bytes") != 16384:
        raise ValueError("dense attention scale memory differs from source")
    return (
        *lower_scale_load(profile, buffer="Q_SCALES", payload_bytes=64,
                          operand="activation").commands,
        *lower_scale_load(profile, buffer="KT_SCALES", payload_bytes=128,
                          operand="weight").commands,
        *lower_strided_scale_dma(profile, buffer="V_SCALES", row_bytes=64,
                                 rows=2, pitch_bytes=64,
                                 destination_byte=4096, selector=1).commands,
    )


def lower_dense_attention_operands(profile: dict) -> tuple[Command, ...]:
    """Tile Q, Kᵀ, and V with the source's DIM16 byte strides."""
    _profile_rows(profile)
    return (
        *lower_matrix_mvin(profile, buffer="Q_IN", matrix_rows=32,
                           matrix_cols=64, burst_cols=16, spad_row=0).commands,
        *lower_matrix_mvin(profile, buffer="KT_IN", matrix_rows=64,
                           matrix_cols=64, burst_cols=16, spad_row=128).commands,
        *lower_matrix_mvin(profile, buffer="V_IN", matrix_rows=64,
                           matrix_cols=64, burst_cols=16, spad_row=768).commands,
    )


def lower_dense_attention_readout(profile: dict, stage: str) -> tuple[Command, ...]:
    """Read one source scratchpad result as 16-byte DMA rows."""
    _profile_rows(profile)
    outputs = {
        "score": ("S_hw", 384, 256),
        "probability": ("P_hw", 384, 256),
        "probability_codes": ("p_codes_hw", 640, 128),
        "output": ("O_hw", 1024, 256),
    }
    if stage not in outputs:
        raise ValueError("dense attention readout stage is unsupported")
    name, start, rows = outputs[stage]
    return lower_linear_spad_mvout(
        profile, buffer=name, total_bytes=rows * 16,
        tile_cols=16, spad_row=start).commands


def lower_dense_attention_program(mlir_text: str, profile: dict,
                                  binding: dict) -> tuple[Command | Fence, ...]:
    """Assemble the source-bound matrix/I/O packets around typed VPU commands."""
    from .nicolas_attention_matrix import lower_dense_attention_matrix
    from .nicolas_attention_vector import lower_dense_attention_vector

    vector = lower_dense_attention_vector(mlir_text, profile, binding)
    program = (
        *lower_dense_attention_scales(profile), Fence(),
        *lower_dense_attention_operands(profile),
        *lower_dense_attention_matrix(profile, "qk"),
        *lower_dense_attention_readout(profile, "score"),
        *vector,
        *lower_dense_attention_matrix(profile, "pv"),
        *lower_dense_attention_readout(profile, "output"),
        *lower_dense_attention_readout(profile, "probability"),
        *lower_dense_attention_readout(profile, "probability_codes"),
    )
    if len(program) != 127:
        raise ValueError("dense attention physical program length differs")
    return program
