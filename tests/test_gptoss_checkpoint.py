"""A checkpoint's MXFP4 bytes survive the Gemmini B layout conversion."""

import numpy as np
import pytest

from mx_gemmini_support.gptoss_checkpoint import (repack_expert_tile,
                                                  repack_expert_weight,
                                                  restore_checkpoint_layout)
from mx_gemmini_support.operands import pack_direct_operands


def test_checkpoint_to_gemmini_b_is_lossless_and_matches_direct_packer():
    rng = np.random.default_rng(17)
    blocks = rng.integers(0, 256, size=(32, 2, 16), dtype=np.uint8)
    scales = rng.integers(1, 255, size=(32, 2), dtype=np.uint8)
    packed = repack_expert_weight(blocks, scales)
    assert (packed.k, packed.n) == (64, 32)
    restored_blocks, restored_scales = restore_checkpoint_layout(packed)
    assert np.array_equal(restored_blocks, blocks)
    assert np.array_equal(restored_scales, scales)

    raw = blocks.reshape(32, 32)
    codes = np.empty((32, 64), dtype=np.uint8)
    codes[:, 0::2] = raw & 15
    codes[:, 1::2] = raw >> 4
    expected = pack_direct_operands("mxfp4", [[0] * 64 for _ in range(32)], codes.T.tolist())
    assert packed.weight_bytes == expected.weight_bytes
    assert packed.scale_bytes == scales.T.tobytes()


def test_reject_malformed_checkpoint_shape():
    with pytest.raises(ValueError, match="blocks"):
        repack_expert_weight(np.zeros((32, 2, 15), dtype=np.uint8),
                             np.zeros((32, 2), dtype=np.uint8))


def test_native_tile_preserves_selected_checkpoint_codes_and_scales():
    rng = np.random.default_rng(23)
    blocks = rng.integers(0, 256, size=(96, 4, 16), dtype=np.uint8)
    scales = rng.integers(1, 255, size=(96, 4), dtype=np.uint8)
    packed = repack_expert_tile(blocks, scales, output_start=32,
                                output_count=32, block_start=1, block_count=2)
    assert (packed.k, packed.n) == (64, 32)
    restored_blocks, restored_scales = restore_checkpoint_layout(packed)
    assert np.array_equal(restored_blocks, blocks[32:64, 1:3])
    assert np.array_equal(restored_scales, scales[32:64, 1:3])
    with pytest.raises(ValueError, match="aligned N"):
        repack_expert_tile(blocks, scales, output_start=1,
                           output_count=32, block_start=0, block_count=1)
    with pytest.raises(ValueError, match="in-range"):
        repack_expert_tile(blocks, scales, output_start=32,
                           output_count=32, block_start=4, block_count=1)
