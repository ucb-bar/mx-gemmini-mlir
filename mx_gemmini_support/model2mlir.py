"""Optional operand handoff from model2MLIR's MX Linear capture.

model2MLIR owns TorchAO quantization and logical tensor orientation. This
target support package owns packing, scale waves, LUT indexing, and commands.
Copying captured tensors to host lists here is an authoring-time operation;
it is not an accelerator runtime transfer or numerical certificate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .contraction import MxContractionPayload, plan_mx_contraction_payload


def _uint8_matrix(value: Any, name: str) -> list[list[int]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("model2MLIR operand handoff requires PyTorch") from exc
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.dtype != torch.uint8:
        raise TypeError(f"{name} must be a rank-2 torch.uint8 tensor")
    if value.requires_grad:
        raise ValueError(f"{name} must not require gradients")
    return value.detach().cpu().tolist()


def plan_rank2_operands(
    operands: Any,
    *,
    activation_lut: Sequence[Sequence[int]] | None = None,
    weight_lut: Sequence[Sequence[int]] | None = None,
    max_blocks_per_wave: int | None = None,
) -> MxContractionPayload:
    """Pack one rank-2 Linear or visible functional-matmul handoff.

    The upstream helper presents A[M,K], B[K,N], activation scales [M,K/32],
    and weight scales [N,K/32]. FP6 codebook policy stays with the caller.
    """
    fmt = getattr(operands, "format", None)
    if fmt not in ("mxfp8", "mxfp6", "mxfp4"):
        raise ValueError("model2MLIR handoff needs a selected MX format")
    return plan_mx_contraction_payload(
        fmt,
        _uint8_matrix(getattr(operands, "activation_codes", None), "activation_codes"),
        _uint8_matrix(getattr(operands, "weight_codes", None), "weight_codes"),
        _uint8_matrix(getattr(operands, "activation_scales", None), "activation_scales"),
        _uint8_matrix(getattr(operands, "weight_scales", None), "weight_scales"),
        weight_scale_layout="ng",
        activation_lut=activation_lut,
        weight_lut=weight_lut,
        max_blocks_per_wave=max_blocks_per_wave,
    )


# Preserve the first explicit API name for callers preparing static Linear sites.
plan_linear_operands = plan_rank2_operands
