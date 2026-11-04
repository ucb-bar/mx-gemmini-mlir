"""Losslessly transpose gpt-oss MXFP4 expert weights to Gemmini B layout.

The checkpoint pairs adjacent K values in each output row. Gemmini's direct
FP4 B operand pairs adjacent N columns at each K. Codes and E8M0 scales are
permuted, never decoded or requantized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PackedExpertWeight:
    k: int
    n: int
    weight_bytes: bytes  # [K][N/2], low nibble = even N
    scale_bytes: bytes   # [K/32][N], original E8M0 values


def repack_expert_weight(blocks: np.ndarray, scales: np.ndarray) -> PackedExpertWeight:
    """Map one expert's [N][K/32][16] codes/scales to Gemmini's B buffers."""
    blocks = np.asarray(blocks)
    scales = np.asarray(scales)
    if (blocks.dtype != np.uint8 or scales.dtype != np.uint8 or blocks.ndim != 3
            or scales.ndim != 2 or blocks.shape[-1] != 16
            or blocks.shape[:2] != scales.shape):
        raise ValueError("gpt-oss expert needs U8 blocks [N,K/32,16] and scales [N,K/32]")
    n, groups, _ = blocks.shape
    k = groups * 32
    if n == 0 or groups == 0 or n % 32:
        raise ValueError("direct FP4 B requires nonzero N multiple of 32 and K multiple of 32")
    checkpoint_bytes = blocks.reshape(n, k // 2)
    codes = np.empty((n, k), dtype=np.uint8)
    codes[:, 0::2] = checkpoint_bytes & 15
    codes[:, 1::2] = checkpoint_bytes >> 4
    b = np.ascontiguousarray(codes.T[:, 0::2] | (codes.T[:, 1::2] << 4))
    b_scales = np.ascontiguousarray(scales.T)
    return PackedExpertWeight(k, n, b.tobytes(), b_scales.tobytes())


def repack_expert_tile(blocks: np.ndarray, scales: np.ndarray, *,
                       output_start: int, output_count: int,
                       block_start: int, block_count: int) -> PackedExpertWeight:
    """Select whole MX groups and Gemmini output tiles from one expert.

    The slice is over original checkpoint coordinates: output rows N and
    input K/32 blocks. No code or scale is numerically transformed.
    """
    blocks = np.asarray(blocks)
    scales = np.asarray(scales)
    if (blocks.dtype != np.uint8 or scales.dtype != np.uint8 or blocks.ndim != 3
            or scales.ndim != 2 or blocks.shape[-1] != 16
            or blocks.shape[:2] != scales.shape):
        raise ValueError("gpt-oss expert needs U8 blocks [N,K/32,16] and scales [N,K/32]")
    if (any(type(value) is not int for value in
            (output_start, output_count, block_start, block_count)) or
            output_start < 0 or output_count <= 0 or output_start % 32 or
            output_count % 32 or output_start + output_count > blocks.shape[0] or
            block_start < 0 or block_count <= 0 or
            block_start + block_count > blocks.shape[1]):
        raise ValueError("gpt-oss tile needs aligned N and whole in-range K/32 blocks")
    return repack_expert_weight(
        blocks[output_start:output_start + output_count,
               block_start:block_start + block_count, :],
        scales[output_start:output_start + output_count,
               block_start:block_start + block_count])


def restore_checkpoint_layout(packed: PackedExpertWeight) -> tuple[np.ndarray, np.ndarray]:
    """Inverse mapping for a byte-for-byte provenance check."""
    if packed.k % 32 or packed.n % 32 or packed.k <= 0 or packed.n <= 0:
        raise ValueError("invalid packed Gemmini geometry")
    if (len(packed.weight_bytes) != packed.k * packed.n // 2
            or len(packed.scale_bytes) != packed.k * packed.n // 32):
        raise ValueError("packed Gemmini buffers do not fit geometry")
    b = np.frombuffer(packed.weight_bytes, dtype=np.uint8).reshape(packed.k, packed.n // 2)
    codes = np.empty((packed.k, packed.n), dtype=np.uint8)
    codes[:, 0::2] = b & 15
    codes[:, 1::2] = b >> 4
    rows = codes.T
    checkpoint_bytes = np.ascontiguousarray(rows[:, 0::2] | (rows[:, 1::2] << 4))
    blocks = checkpoint_bytes.reshape(packed.n, packed.k // 32, 16)
    scales = np.ascontiguousarray(np.frombuffer(packed.scale_bytes, dtype=np.uint8)
                                  .reshape(packed.k // 32, packed.n).T)
    return blocks, scales
