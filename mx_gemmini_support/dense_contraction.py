"""Profile-bound dense contraction recipe for serial MX FP4/FP8 kernels."""

from __future__ import annotations

from dataclasses import dataclass

from .dense_kernel import (DenseBatchKernel, DenseKernel, emit_dense_f32_batch_kernel,
                           emit_dense_f32_kernel)
from .compact_runtime_issuer import emit_compact_runtime_issuer
from .planned_runtime_issuer import PlannedRuntimeIssuer, emit_planned_runtime_issuer
from .source_gemm import plan_mx_gemm


MAX_UNROLLED_COMMANDS = 4_096


@dataclass(frozen=True)
class DenseShapePlan:
    logical_shape_mnk: tuple[int, int, int]
    padded_shape_mnk: tuple[int, int, int]
    tile_mnk: tuple[int, int, int]
    precision: str
    output_tiles: int
    k_waves: int
    estimated_commands: int


@dataclass(frozen=True)
class DenseContraction:
    shape_plan: DenseShapePlan
    issuer: PlannedRuntimeIssuer
    kernel: DenseKernel
    issuer_mode: str


@dataclass(frozen=True)
class DenseBatchContraction:
    shape_plan: DenseShapePlan
    batch: int
    issuer: PlannedRuntimeIssuer
    kernel: DenseBatchKernel
    issuer_mode: str


def _round_up(value: int, group: int) -> int:
    return (value + group - 1) // group * group


def plan_dense_shape(profile: dict, logical_shape_mnk: tuple[int, int, int], *,
                     precision: str) -> DenseShapePlan:
    """Choose the smallest mesh tile and a legal K wave without source data.

    This is a correctness-first serial recipe, not a performance schedule.
    It is deliberately conservative about output tiles and scale memory.
    """
    if (precision not in {"FP4", "FP8"} or len(logical_shape_mnk) != 3 or
            any(type(value) is not int or value <= 0 for value in logical_shape_mnk)):
        raise ValueError("MX dense shape needs positive M,N,K and direct FP4/FP8")
    dim = profile["geometry"]["mesh_columns"]
    pe = dim * (2 if precision == "FP4" else 1)
    m, n, k = logical_shape_mnk
    tm = tn = pe
    mp, np = _round_up(m, tm), _round_up(n, tn)
    failures = []
    for tk in sorted((32, 64, 128), key=lambda wave: (_round_up(k, wave) - k,
                                                      _round_up(k, wave) // wave)):
        kp = _round_up(k, tk)
        try:
            plan = plan_mx_gemm(
                shape=(mp, np, kp), tile=(tm, tn, tk), datatype=precision,
                quant_output=False, acc_to_gmem=False,
                scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
                profile=profile)
            scale_half = profile["resources"]["scale_mem_config"]["size_bytes"] // 4
            if (profile["transport"] != "rocket_rocc" or
                    profile["resources"]["scale_block_elements"] != 32 or
                    scale_half != 4096 or
                    max(tm, tn) * (tk // 32) > scale_half):
                raise ValueError("MX scale or transport geometry is unsupported")
            if np > tn and (tn * 2 % dim or not 1 <= tn * 2 // dim <= 16):
                raise ValueError("MX row-major BF16 transfer is unsupported")
        except (KeyError, ValueError) as error:
            failures.append(str(error))
            continue
        tiles = len(plan.get("output_tiles", ())) or 1
        waves = len(plan["waves"])
        values_per_byte = 2 if precision == "FP4" else 1
        transfers_per_wave = (tm // (dim * values_per_byte) +
                              tn // (dim * values_per_byte)) * (tk // dim)
        readout_transfers = (tm if tiles > 1 else plan["c_rows"] // dim)
        commands = 6 + tiles * (waves * (11 + transfers_per_wave) +
                                2 + readout_transfers)
        return DenseShapePlan(
            logical_shape_mnk, (mp, np, kp), (tm, tn, tk), precision,
            tiles, waves, commands)
    raise ValueError(f"MX profile has no serial dense tile: {failures[0] if failures else 'unknown'}")


def emit_dense_contraction(profile: dict, logical_shape_mnk: tuple[int, int, int], *,
                           precision: str, symbol: str,
                           issuer_symbol: str,
                           shared_workspace_symbol: str | None = None) -> DenseContraction:
    """Emit both halves of one source-independent dense MX catalog candidate."""
    shape = plan_dense_shape(profile, logical_shape_mnk, precision=precision)
    compact = shape.estimated_commands > MAX_UNROLLED_COMMANDS
    issuer = (emit_compact_runtime_issuer if compact else emit_planned_runtime_issuer)(
        profile, shape.padded_shape_mnk, shape.tile_mnk, precision=precision,
        symbol=issuer_symbol)
    kernel = emit_dense_f32_kernel(
        issuer, logical_shape_mnk, symbol=symbol, issuer_symbol=issuer_symbol,
        shared_workspace_symbol=shared_workspace_symbol)
    return DenseContraction(shape, issuer, kernel, "compact_loop" if compact else "unrolled")


def emit_dense_batch_contraction(profile: dict,
                                 logical_shape_bmnk: tuple[int, int, int, int], *,
                                 precision: str, symbol: str,
                                 issuer_symbol: str,
                                 shared_workspace_symbol: str | None = None) -> DenseBatchContraction:
    """Emit a serial whole-batch entry over a shared per-slice MX workspace."""
    if (len(logical_shape_bmnk) != 4 or
            any(type(value) is not int or value <= 0 for value in logical_shape_bmnk)):
        raise ValueError("MX batched contraction needs positive B,M,N,K")
    batch, m, n, k = logical_shape_bmnk
    shape = plan_dense_shape(profile, (m, n, k), precision=precision)
    compact = shape.estimated_commands > MAX_UNROLLED_COMMANDS
    issuer = (emit_compact_runtime_issuer if compact else emit_planned_runtime_issuer)(
        profile, shape.padded_shape_mnk, shape.tile_mnk, precision=precision,
        symbol=issuer_symbol)
    kernel = emit_dense_f32_batch_kernel(
        issuer, logical_shape_bmnk, symbol=symbol, issuer_symbol=issuer_symbol,
        shared_workspace_symbol=shared_workspace_symbol)
    return DenseBatchContraction(shape, batch, issuer, kernel,
                                 "compact_loop" if compact else "unrolled")
