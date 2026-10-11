"""Source-equivalent MX scale, operand, and readout packets for dense attention."""

from __future__ import annotations

from .command_ir import Command, Fence, Operand
from .physical_program import _config_ld, _config_st, _transfer


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
        Command(27, Operand(buffer="Q_SCALES"), Operand(immediate=64)),
        Command(27, Operand(buffer="KT_SCALES"),
                Operand(immediate=(1 << 32) | 128)),
        Command(27, Operand(buffer="V_SCALES", address_mask=(1 << 40) - 1,
                            or_bits=64 << 40),
                Operand(immediate=(2 << 46) | (4096 << 33) |
                                  (1 << 32) | 64)),
    )


def lower_dense_attention_operands(profile: dict) -> tuple[Command, ...]:
    """Tile Q, Kᵀ, and V with the source's DIM16 byte strides."""
    _profile_rows(profile)
    commands = [_config_ld(64)]
    for i in range(2):
        for k in range(4):
            commands.append(_transfer(2, "Q_IN", i * 16 * 64 + k * 16,
                                      (i * 4 + k) * 16))
    commands.append(_config_ld(64))
    for name, start in (("KT_IN", 128), ("V_IN", 768)):
        for k in range(4):
            for j in range(4):
                commands.append(_transfer(2, name, k * 16 * 64 + j * 16,
                                          start + (k * 4 + j) * 16))
        if name == "KT_IN":
            commands.append(_config_ld(64))
    return tuple(commands)


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
    return (_config_st(16), *(_transfer(3, name, r * 16, start + r)
                              for r in range(0, rows, 16)))


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
