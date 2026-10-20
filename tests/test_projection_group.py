"""Projection completion batching retains each original matrix partition."""

from pathlib import Path

import pytest

from mx_gemmini_support.projection_group import emit_projection_group
from mx_gemmini_support.scheduled_native_issuer import emit_scheduled_native_issuer
from mx_gemmini_support.target_profile import load_profile

PROFILE = load_profile(
    Path(__file__).resolve().parents[1]
    / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
)


def parts(columns=(2048, 256, 256)):
    return [
        {
            "shape": (160, n, 2048),
            "tile_mnk": (80, 128, 512),
            "weight_symbols": (f"weight{i}", f"scale{i}"),
        }
        for i, n in enumerate(columns)
    ]


def test_one_completion_precedes_all_readouts_and_destinations_do_not_overlap():
    generated = emit_projection_group(
        PROFILE, parts(), symbol="grouped", workspace_symbol="workspace"
    )
    source = generated.source
    assert generated.workspace_bytes == 160 * 2560 * 2 + 2048
    assert generated.output_shapes == ((160, 2048), (160, 256), (160, 256))
    assert source.count("MX_FENCE();") == 1
    outer = source[source.index("void grouped(") :]
    assert (
        outer.index("grouped_issue_2(")
        < outer.index("MX_FENCE();")
        < outer.index("for (uint32_t i")
    )
    for i, n in enumerate((2048, 256, 256)):
        original = emit_scheduled_native_issuer(
            PROFILE,
            (160, n, 2048),
            precision="FP8",
            symbol=f"grouped_issue_{i}",
            tile_mnk=(80, 128, 512),
            defer_completion=True,
        )
        assert original.digest == generated.schedule_digests[i]
        assert original.issuer.plan["completion"] == "caller_fence_before_readout"
    for offset in (0, 655360, 737280):
        assert f"workspace+{offset}u" in source


def test_invalid_groups_cannot_drop_completion_or_change_a_tile_partition():
    with pytest.raises(ValueError):
        emit_scheduled_native_issuer(
            PROFILE, (32, 128, 512), precision="FP8", defer_completion=1
        )
    for bad in (
        parts()[:1],
        parts((2048, 256, 256))[:1] * 9,
        [dict(p, tile_mnk=(160, 64, 512)) if i else p for i, p in enumerate(parts())],
        [dict(p, shape=(160, 256, 1024)) if i else p for i, p in enumerate(parts())],
        [
            {
                "shape": (16, 16, 32),
                "tile_mnk": (16, 16, 32),
                "weight_symbols": (f"w{i}", f"s{i}"),
            }
            for i in range(2)
        ],
    ):
        with pytest.raises(ValueError):
            emit_projection_group(
                PROFILE, bad, symbol="grouped", workspace_symbol="workspace"
            )
