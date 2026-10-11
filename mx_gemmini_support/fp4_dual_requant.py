"""Compatibility entry points for Nicolas's source-bound dual-layout FP4 case."""

from __future__ import annotations

from .dual_spad_requant import (BUFFERS, FP4_SPEC, DualRequantSpec,
                                lower_dual_requant, render_dual_requant)


M, N = FP4_SPEC.m, FP4_SPEC.n


def render_fp4_dual_requant(profile: dict, *, source_sha256: str,
                           header_sha256: str) -> str:
    return render_dual_requant(profile, source_sha256=source_sha256,
                               header_sha256=header_sha256, spec=FP4_SPEC)


def lower_fp4_dual_requant(mlir_text: str, profile: dict):
    return lower_dual_requant(mlir_text, profile, FP4_SPEC)
