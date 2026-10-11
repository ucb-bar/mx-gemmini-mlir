"""Physical QK and PV matrix packets for Nicolas's dense MX+VPU attention.

This is a bounded source-equivalent matrix issuer. The typed connected graph
will call it after checking the two model2MLIR contraction sites and their
scratchpad handoff.
"""

from __future__ import annotations

import re

from .command_ir import Command, Operand
from .nicolas_attention_shape import DenseAttentionShape
from .target_profile import require_compute


STAGES = {
    "qk": {"m": 32, "n": 64, "k": 64, "a_row": 0,
           "b_end_row": 384, "output_row": 384, "weight_half": 0},
    "pv": {"m": 32, "n": 64, "k": 64, "a_row": 640,
           "b_end_row": 1024, "output_row": 1024, "weight_half": 1},
}


def lower_dense_attention_matrix(profile: dict, stage: str,
                                 *, scale_sink: str = "scale_sink",
                                 shape: DenseAttentionShape = DenseAttentionShape(32, 64, 64)
                                 ) -> tuple[Command, ...]:
    """Encode source `mm`'s config, scale, and LOOP_WS_SPAD packets."""
    if stage not in STAGES:
        raise ValueError("dense attention matrix stage must be qk or pv")
    if (profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_rows") != 16 or
            profile.get("geometry", {}).get("mesh_columns") != 16 or
            not profile.get("resources", {}).get("vpu") or
            not profile.get("resources", {}).get("spad_requant")):
        raise ValueError("dense attention matrix requires the DIM16 MX+VPU Rocket profile")
    require_compute(profile, "fp8_e4m3", "fp8_e4m3", pe_mode=8,
                    activation_projection="direct", weight_projection="direct")
    shape.validate(profile)
    spec = ({"m": shape.sq, "n": shape.sk, "k": shape.d,
             "a_row": 0, "b_end_row": shape.score_row,
             "output_row": shape.score_row, "weight_half": 0}
            if stage == "qk" else
            {"m": shape.sq, "n": shape.d, "k": shape.sk,
             "a_row": shape.probability_row, "b_end_row": shape.output_row,
             "output_row": shape.output_row, "weight_half": 1})
    return lower_spad_matmul(
        profile, m=spec["m"], n=spec["n"], k=spec["k"],
        a_row=spec["a_row"], b_end_row=spec["b_end_row"],
        output_row=spec["output_row"], weight_half=spec["weight_half"],
        scale_sink=scale_sink)


def lower_spad_matmul(profile: dict, *, m: int, n: int, k: int,
                      a_row: int, b_end_row: int, output_row: int,
                      weight_half: int, scale_sink: str) -> tuple[Command, ...]:
    """Lower one complete DIM16 E4M3 scratchpad matmul with BF16 output."""
    if (profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_rows") != 16 or
            profile.get("geometry", {}).get("mesh_columns") != 16):
        raise ValueError("MX scratchpad matmul requires a DIM16 Rocket profile")
    if (any(type(value) is not int for value in
            (m, n, k, a_row, b_end_row, output_row, weight_half)) or
            any(value <= 0 or value % 16 for value in (m, n, k)) or
            max(m, n, k) >= 1 << 16 or weight_half not in (0, 1)):
        raise ValueError("MX scratchpad matmul geometry or scale half is unsupported")
    require_compute(profile, "fp8_e4m3", "fp8_e4m3", pe_mode=8,
                    activation_projection="direct", weight_projection="direct")
    a, b_end, c = a_row, b_end_row, output_row
    b_start = b_end - k * n // 16
    rows = profile["resources"]["scratchpad_bytes"] // 16
    if (not 0 <= a < b_start or a + m * k // 16 > b_start or
            b_start < 0 or b_end > c or c + m * n * 2 // 16 > rows or
            max(a, b_end, c) >= 1 << 14):
        raise ValueError("MX scratchpad matmul placement exceeds selected profile")
    i, j, kk = m // 16, n // 16, k // 16
    wsel = weight_half
    def command(funct: int, rs1: int | Operand, rs2: int) -> Command:
        return Command(funct, rs1 if isinstance(rs1, Operand) else Operand(immediate=rs1),
                       Operand(immediate=rs2))
    return (
        # gemmini_extended3_config_ex(WS, ..., OUT_BF16=3).
        command(0, (1 << 16) | (3 << 14) | (1 << 2), 1 << 48),
        command(0, 2, n * 2),
        # gemmini_mxquant_config_mvout(scale_sink, I, J, K, 0, wsel, 1).
        command(26, Operand(buffer=scale_sink, address_mask=(1 << 33) - 1,
                            or_bits=(wsel << 61) | (kk << 51) |
                                    (j << 42) | (i << 33)), 1),
        command(9, 0, (kk << 32) | (j << 16) | i),
        command(24, a, b_end),
        # SPAD_STORE=0x38 skips loop DMA and drains BF16 into scratchpad.
        command(8, 0, (c << 32) | 0x200 | 0x38),
    )


def lower_managed_spad_matmul(profile: dict, *, m: int, n: int, k: int,
                              a_row: int, b_end_row: int, output_row: int,
                              weight_half: int, activation_half: int,
                              scale_sink: str) -> tuple[Command, ...]:
    """Issue one managed-scale loop after its gated weight-scale load.

    The caller owns the schedule: it must place a gated load for the same
    weight half before this config and may prefetch the next half afterward.
    The selected RTL's managed config waits for the load to land and releases
    the other half; the loop alternates accumulator halves with INC_ACC.
    """
    if (profile.get("source", {}).get("gemmini_revision") !=
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52" or
            profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_rows") != 16 or
            profile.get("geometry", {}).get("mesh_columns") != 16 or
            not profile.get("resources", {}).get("vpu") or
            not profile.get("resources", {}).get("spad_requant")):
        raise ValueError("managed MX scratchpad matmul needs the pinned DIM16 MX+VPU RTL")
    require_compute(profile, "fp8_e4m3", "fp8_e4m3", pe_mode=8,
                    activation_projection="direct", weight_projection="direct")
    if (any(type(value) is not int for value in
            (m, n, k, a_row, b_end_row, output_row, weight_half, activation_half)) or
            any(value <= 0 or value % 16 for value in (m, n, k)) or
            max(m, n, k) >= 1 << 16 or
            weight_half not in (0, 1) or activation_half not in (0, 1) or
            not isinstance(scale_sink, str) or
            re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", scale_sink) is None):
        raise ValueError("managed MX scratchpad matmul geometry or scale half is unsupported")
    b_start = b_end_row - k * n // 16
    limit = profile["resources"]["scratchpad_bytes"] // 16
    spans = ((a_row, a_row + m * k // 16),
             (b_start, b_end_row),
             (output_row, output_row + m * n * 2 // 16))
    if (any(start < 0 or start >= end or end > limit for start, end in spans) or
            any(left[0] < right[1] and right[0] < left[1]
                for i, left in enumerate(spans) for right in spans[i + 1:]) or
            max(a_row, b_end_row, output_row) >= 1 << 14):
        raise ValueError("managed MX scratchpad matmul placement overlaps or exceeds profile")
    i, j, kk = m // 16, n // 16, k // 16
    return (
        Command(26, Operand(buffer=scale_sink, address_mask=(1 << 33) - 1,
                            or_bits=(weight_half << 61) | (activation_half << 60) |
                                    (kk << 51) | (j << 42) | (i << 33)),
                Operand(immediate=(1 << 18) | (1 << 17) | 1)),
        Command(9, Operand(immediate=0),
                Operand(immediate=(kk << 32) | (j << 16) | i)),
        Command(24, Operand(immediate=a_row), Operand(immediate=b_end_row)),
        Command(8, Operand(immediate=0),
                Operand(immediate=(output_row << 32) | 0x200 | 0x100 | 0x38)),
    )


def lower_flash_setup(profile: dict, *, store_stride_bytes: int) -> tuple[Command, ...]:
    """Configure Nicolas's E4M3 mesh and BF16 scratchpad drain once per pass."""
    if (profile.get("source", {}).get("gemmini_revision") !=
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52" or
            profile.get("transport") != "rocket_rocc" or
            not profile.get("resources", {}).get("vpu") or
            not profile.get("resources", {}).get("spad_requant")):
        raise ValueError("flash setup needs the pinned MX+VPU Rocket profile")
    require_compute(profile, "fp8_e4m3", "fp8_e4m3", pe_mode=8,
                    activation_projection="direct", weight_projection="direct")
    if (type(store_stride_bytes) is not int or store_stride_bytes <= 0 or
            store_stride_bytes % 16 or store_stride_bytes >= 1 << 16):
        raise ValueError("flash BF16 store stride is unsupported")
    return (
        Command(0, Operand(immediate=(1 << 16) | (3 << 14) | (1 << 2)),
                Operand(immediate=1 << 48)),
        Command(0, Operand(immediate=2), Operand(immediate=store_stride_bytes)),
    )
