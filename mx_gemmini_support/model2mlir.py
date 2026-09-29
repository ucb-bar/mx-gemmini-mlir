"""Logical MX operand handoff from the out-of-tree TorchAO quantizer.

This support package owns TorchAO quantization, logical tensor orientation,
packing, scale waves, LUT indexing, and diagnostic commands.
Copying captured tensors to host lists here is an authoring-time operation;
it is not an accelerator runtime transfer or numerical certificate.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace
from typing import Any

from .contraction import MxContractionPayload, plan_mx_contraction_payload


def _uint8_tensor(value: Any, name: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("MX operand handoff requires PyTorch") from exc
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


@dataclass(frozen=True)
class IndexedTiledMxPayload:
    batch_index: tuple[int, ...]
    m_start: int
    m_stop: int
    n_start: int
    n_stop: int
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
        raise ValueError("MX handoff needs a selected MX format")
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


def iter_spatial_tiles(
    operands: Any,
    *,
    tile_m: int = 32,
    tile_n: int = 32,
    activation_lut: Sequence[Sequence[int]] | None = None,
    weight_lut: Sequence[Sequence[int]] | None = None,
    max_blocks_per_wave: int | None = None,
) -> Iterator[IndexedTiledMxPayload]:
    """Yield independent spatial tiles without packing the entire contraction.

    Input codes/scales use the quantizer's logical axes. FP6 callers supply one
    exact source codebook per two global M rows and N columns; each tile takes
    only its own lines. No command order, output assembly, or host work is
    inferred from this buffer plan.
    """
    fmt = getattr(operands, "format", None)
    unit = {"mxfp8": 16, "mxfp6": 32, "mxfp4": 32}.get(fmt)
    if unit is None:
        raise ValueError("spatial tiling needs a selected MX format")
    if any(type(extent) is not int or extent < unit or extent > 128 or extent % unit
           for extent in (tile_m, tile_n)):
        raise ValueError("MX spatial tile extents need aligned values within 128")
    names = ("activation_codes", "weight_codes", "activation_scales", "weight_scales")
    values = {name: _uint8_tensor(getattr(operands, name, None), name) for name in names}
    activation, weight = values["activation_codes"], values["weight_codes"]
    if activation.ndim not in (2, 3, 4) or weight.ndim != activation.ndim:
        raise ValueError("MX spatial tiling needs rank-2 to rank-4 operands")
    batch_shape = tuple(activation.shape[:-2])
    if any(extent < 1 for extent in batch_shape):
        raise ValueError("MX independent batch axes must be nonempty")
    if any(value.ndim != activation.ndim or tuple(value.shape[:-2]) != batch_shape
           for value in values.values()):
        raise ValueError("MX operand and E8M0 batch axes must match")
    m, k = activation.shape[-2:]
    weight_k, n = weight.shape[-2:]
    if (k != weight_k or k < 32 or k % 32 or m < unit or n < unit
            or m % unit or n % unit):
        raise ValueError("MX spatial tiling needs aligned M/N and matching whole K blocks")
    if tuple(values["activation_scales"].shape[-2:]) != (m, k // 32):
        raise ValueError("activation E8M0 axes disagree with A")
    if tuple(values["weight_scales"].shape[-2:]) != (n, k // 32):
        raise ValueError("weight E8M0 axes disagree with B")
    if fmt == "mxfp6":
        if activation_lut is None or weight_lut is None:
            raise ValueError("MXFP6 spatial tiles require exact codebooks")
        if len(activation_lut) != m // 2 or len(weight_lut) != n // 2:
            raise ValueError("MXFP6 global codebook lines disagree with M/N")
    elif activation_lut is not None or weight_lut is not None:
        raise ValueError("direct MX spatial tiles do not use codebooks")

    indices = product(*(range(extent) for extent in batch_shape)) if batch_shape else [()]
    for index in indices:
        for m_start in range(0, m, tile_m):
            m_stop = min(m, m_start + tile_m)
            for n_start in range(0, n, tile_n):
                n_stop = min(n, n_start + tile_n)
                tile = SimpleNamespace(
                    format=fmt,
                    activation_codes=activation[index][m_start:m_stop],
                    weight_codes=weight[index][:, n_start:n_stop],
                    activation_scales=values["activation_scales"][index][m_start:m_stop],
                    weight_scales=values["weight_scales"][index][n_start:n_stop],
                )
                yield IndexedTiledMxPayload(
                    index, m_start, m_stop, n_start, n_stop,
                    plan_rank2_operands(
                        tile,
                        activation_lut=(activation_lut[m_start // 2:m_stop // 2]
                                        if activation_lut is not None else None),
                        weight_lut=(weight_lut[n_start // 2:n_stop // 2]
                                    if weight_lut is not None else None),
                        max_blocks_per_wave=max_blocks_per_wave,
                    ),
                )
