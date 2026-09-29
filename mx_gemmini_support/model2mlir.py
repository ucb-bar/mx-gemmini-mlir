"""Optional operand handoff from model2MLIR's MX Linear capture.

model2MLIR owns TorchAO quantization and logical tensor orientation. This
target support package owns packing, scale waves, LUT indexing, and commands.
Copying captured tensors to host lists here is an authoring-time operation;
it is not an accelerator runtime transfer or numerical certificate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace
from typing import Any

from .contraction import MxContractionPayload, plan_mx_contraction_payload


def _uint8_tensor(value: Any, name: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("model2MLIR operand handoff requires PyTorch") from exc
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8:
        raise TypeError(f"{name} must be a torch.uint8 tensor")
    if value.requires_grad:
        raise ValueError(f"{name} must not require gradients")
    return value


def _uint8_matrix(value: Any, name: str) -> list[list[int]]:
    value = _uint8_tensor(value, name)
    if value.ndim != 2:
        raise TypeError(f"{name} must be a rank-2 torch.uint8 tensor")
    return value.detach().cpu().tolist()


@dataclass(frozen=True)
class IndexedMxPayload:
    batch_index: tuple[int, ...]
    payload: MxContractionPayload


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


def plan_independent_batches(
    operands: Any,
    *,
    activation_lut: Sequence[Sequence[int]] | None = None,
    weight_lut: Sequence[Sequence[int]] | None = None,
    max_blocks_per_wave: int | None = None,
) -> tuple[IndexedMxPayload, ...]:
    """Pack each independent rank-2 slice of a rank-2 to rank-4 handoff.

    A common exact FP6 codebook is reused across batches. This returns source
    buffers only; the caller still schedules every batch and LUT upload.
    """
    names = ("activation_codes", "weight_codes", "activation_scales", "weight_scales")
    values = {name: _uint8_tensor(getattr(operands, name, None), name) for name in names}
    activation = values["activation_codes"]
    if activation.ndim not in (2, 3, 4):
        raise ValueError("MX independent batches need rank-2 to rank-4 operands")
    batch_shape = tuple(activation.shape[:-2])
    if any(extent < 1 for extent in batch_shape):
        raise ValueError("MX independent batch axes must be nonempty")
    if any(value.ndim != activation.ndim or tuple(value.shape[:-2]) != batch_shape
           for value in values.values()):
        raise ValueError("MX operand and E8M0 batch axes must match")
    indices = product(*(range(extent) for extent in batch_shape)) if batch_shape else [()]
    return tuple(
        IndexedMxPayload(index, plan_rank2_operands(
            SimpleNamespace(format=getattr(operands, "format", None), **{
                name: value[index] for name, value in values.items()
            }),
            activation_lut=activation_lut, weight_lut=weight_lut,
            max_blocks_per_wave=max_blocks_per_wave,
        ))
        for index in indices
    )
