"""The candidate audit counts every bound matrix, including reused batch kernels."""

import pytest

from tools.compare_radiance_rocket_candidate import expected_matrix_tiles


def test_all_bound_calls_and_batch_slices_contribute_to_expected_work():
    catalog = {
        "kernels": [
            {
                "symbol": "dense",
                "precision": "mxfp8",
                "padded_shape_mnk": [160, 256, 2048],
                "dimensions": {"batch": 1},
            },
            {
                "symbol": "attention",
                "precision": "mxfp8",
                "padded_shape_mnk": [160, 160, 64],
                "dimensions": {"batch": 32},
            },
        ],
        "bindings": [
            {"symbol": "dense"},
            {"symbol": "attention"},
            {"symbol": "attention"},
        ],
    }
    assert (
        expected_matrix_tiles(catalog)
        == 160 * 256 * 2048 // 4096 + 2 * 32 * 160 * 160 * 64 // 4096
    )
    catalog["kernels"][1]["precision"] = "mxfp4"
    with pytest.raises(ValueError, match="all-FP8"):
        expected_matrix_tiles(catalog)


def test_absent_or_unaligned_matrix_work_refuses():
    with pytest.raises(ValueError, match="no matrix work"):
        expected_matrix_tiles({"kernels": [], "bindings": []})
    with pytest.raises(ValueError, match="invalid compiled matrix extent"):
        expected_matrix_tiles(
            {
                "kernels": [
                    {
                        "symbol": "bad",
                        "precision": "mxfp8",
                        "padded_shape_mnk": [15, 16, 32],
                        "dimensions": {"batch": 1},
                    }
                ],
                "bindings": [{"symbol": "bad"}],
            }
        )
