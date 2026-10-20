"""The layout audit must follow actual source matrix-address arithmetic."""

import numpy as np
import pytest

from tools.audit_radiance_compiler_weights import phase_parameters, unpack_weight


@pytest.mark.parametrize(
    "k,n,tile_k", [(64, 128, 32), (512, 192, 256), (5632, 64, 256)]
)
def test_layout_decoder_matches_scalar_native_source_offsets(k, n, tile_k):
    rng = np.random.default_rng(85)
    codes = rng.integers(0, 256, k * n, dtype=np.uint8)
    scales = rng.integers(104, 128, k * n // 32, dtype=np.uint8)
    actual, actual_scales = unpack_weight(
        codes.tobytes(), scales.tobytes(), k=k, n=n, tile_k=tile_k
    )
    nt, ks = n // 64, k // tile_k
    # CMD10 reads each (K-step,N-tile) block; CMD31 uses (N-tile,K-step).
    for row in range(k):
        for col in range(n):
            offset = (
                ((row // tile_k) * nt + col // 64) * tile_k * 64
                + (row % tile_k) * 64
                + col % 64
            )
            assert actual[row, col] == codes[offset]
    for group in range(k // 32):
        for col in range(n):
            row = group * 32
            offset = (
                ((col // 64) * ks + row // tile_k) * (tile_k // 32) * 64
                + (row % tile_k) // 32 * 64
                + col % 64
            )
            assert actual_scales[group, col] == scales[offset]


@pytest.mark.parametrize(
    "k,n,tile_k", [(65, 128, 32), (64, 96, 32), (64, 64, 16), (64, 64, 0)]
)
def test_incomplete_or_unknown_source_tiles_are_refused(k, n, tile_k):
    with pytest.raises(ValueError):
        unpack_weight(b"", b"", k=k, n=n, tile_k=tile_k)


def test_truncated_codes_or_scales_are_refused():
    for codes, scales in ((bytes(2047), bytes(64)), (bytes(2048), bytes(63))):
        with pytest.raises(ValueError, match="extent"):
            unpack_weight(codes, scales, k=32, n=64, tile_k=32)


def test_combined_projection_order_and_final_head_are_source_bound():
    assert phase_parameters(2, 2) == [
        f"lm.model.layers.2.self_attn.{part}_proj.weight" for part in ("q", "k", "v")
    ]
    assert phase_parameters(21, 10) == [
        f"lm.model.layers.21.mlp.{part}_proj.weight" for part in ("gate", "up")
    ]
    assert phase_parameters(255, 14) == ["lm.lm_head.weight"]
    for layer, phase in ((22, 2), (255, 2), (0, 14), (0, 4)):
        with pytest.raises(ValueError):
            phase_parameters(layer, phase)
