"""Derive a live MX matrix command stream from profile, shape, and tile.

This path has no captured operand or golden. It shares the command scheduler
with payload-bound source qualification, then checks every addressed buffer
extent. Numerical qualification remains a separate compiler acceptance gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .command_ir import Command, emit_c
from .physical_program import _regular_schedule_steps
from .source_gemm import plan_mx_gemm
from .target_profile import profile_sha256


@dataclass(frozen=True)
class PlannedRuntimeIssuer:
    source: str
    shape_mnk: tuple[int, int, int]
    tile_mnk: tuple[int, int, int]
    precision: str
    profile_sha256: str
    buffer_bytes: Mapping[str, int]
    plan: dict


def emit_planned_runtime_issuer(profile: dict, shape_mnk: tuple[int, int, int],
                                tile_mnk: tuple[int, int, int], *, precision: str,
                                symbol: str = "mx_issue_encoded") -> PlannedRuntimeIssuer:
    """Emit serial BF16 readout commands for one legal direct FP4/FP8 GEMM."""
    if precision not in {"FP4", "FP8"}:
        raise ValueError("live MX issuer supports direct FP4 and FP8 only")
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("live MX issuer requires selected Rocket RoCC transport")
    if (len(shape_mnk) != 3 or len(tile_mnk) != 3 or
            any(type(value) is not int or value <= 0 for value in
                (*shape_mnk, *tile_mnk))):
        raise ValueError("live MX issuer needs positive static shape and tile")
    if (not symbol or not (symbol[0].isalpha() or symbol[0] == "_") or
            any(not (char.isalnum() or char == "_") for char in symbol)):
        raise ValueError("live MX issuer needs a C symbol")
    m, n, k = shape_mnk
    tm, tn, tk = tile_mnk
    dim = profile["geometry"]["mesh_columns"]
    if (m % tm or n % tn or tk % dim or
            profile["resources"]["scale_block_elements"] != 32 or
            profile["resources"]["scale_mem_config"]["size_bytes"] // 4 != 4096):
        raise ValueError("live MX issuer shape or scale geometry is unsupported")
    plan = plan_mx_gemm(
        shape=shape_mnk, tile=tile_mnk, datatype=precision, quant_output=False,
        acc_to_gmem=False, scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
        profile=profile)
    if len(plan.get("output_tiles", ())) > 1:
        if tn * 2 % dim or not 1 <= tn * 2 // dim <= 16:
            raise ValueError("live MX row-major BF16 readout has unsupported tile width")
        plan = {**plan, "bf16_output_layout": "row_major_bf16"}
    bits = 4 if precision == "FP4" else 8
    lengths = {
        "activation": m * k * bits // 8,
        "activation_scales": m * k // 32,
        "weight": k * n * bits // 8,
        "weight_scales": k * n // 32,
        "output_bf16": m * n * 2,
        "scratch_output_scales": 2048,
    }
    if (any(value <= 0 or value > 0x7fffffff for value in lengths.values()) or
            m * k % 32 or k * n % 32):
        raise ValueError("live MX issuer buffer extents overflow or lack scale groups")
    steps = _regular_schedule_steps(
        profile, precision, shape_mnk, tile_mnk, plan, lengths,
        quant_output=False, matrix_quant_output=False, vector_requant=False,
        vector_ops=[], mode="spike_serial")
    commands = [step.command for step in steps]
    referenced = {operand.buffer for command in commands if isinstance(command, Command)
                  for operand in (command.rs1, command.rs2) if operand.buffer is not None}
    if (not {"activation", "activation_scales", "weight", "weight_scales",
             "output_bf16"}.issubset(referenced) or referenced - set(lengths)):
        raise ValueError("live MX schedule uses an unbound or omitted buffer")
    source = emit_c(commands, transport="rocket_rocc", buffers=tuple(lengths),
                    symbol=symbol)
    return PlannedRuntimeIssuer(source, shape_mnk, tile_mnk, precision,
                                profile_sha256(profile), lengths, plan)
