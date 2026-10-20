"""Catch broken recurrent handoffs even when output logits remain available."""

import numpy as np
import pytest

from tools.audit_recurrent_session_state import check_state, digest
from tools.compare_radiance_rocket_cached import gates, quality


def tiny_rows():
    rows = []
    positions = [(0, 0, 128)] + [(1, s, 129 + s) for s in range(32)]
    for p, s, position in positions:
        for o, size, sha in [
            (0, 128000, digest(bytes([position]))),
            (1, 7208960, digest(b"cache" + bytes([position]))),
            (2, 176, digest(np.full(22, position, dtype="<i8").tobytes())),
        ]:
            rows.append(
                {"program": p, "step": s, "output": o, "bytes": size, "sha256": sha}
            )
    return rows


def smol_rows():
    cache = digest(b"prefix")
    rows = [
        {"program": 0, "step": 0, "output": o, "bytes": n, "sha256": cache}
        for o, n in [(0, 113), (1, 2314240)]
    ]
    timestep = np.float32(1)
    for s in range(10):
        timestep = np.float32(timestep + np.float32(-0.1))
        for o, n, sha in [
            (0, 6400, digest(bytes([s]))),
            (1, 2314240, cache),
            (2, 4, digest(np.asarray([timestep], dtype="<f4").tobytes())),
        ]:
            rows.append(
                {"program": 1, "step": s, "output": o, "bytes": n, "sha256": sha}
            )
    rows.append(
        {
            "program": 2,
            "step": 0,
            "output": 0,
            "bytes": 6400,
            "sha256": digest(bytes([9])),
        }
    )
    return rows


@pytest.mark.parametrize(
    "kind,make", [("tinyllama", tiny_rows), ("smolvla", smol_rows)]
)
def test_complete_recurrent_state(kind, make):
    assert check_state(kind, make())


@pytest.mark.parametrize(
    "kind,make,index",
    [
        ("tinyllama", tiny_rows, 2),
        ("tinyllama", tiny_rows, 98),
        ("smolvla", smol_rows, 3),
        ("smolvla", smol_rows, 31),
        ("smolvla", smol_rows, 32),
    ],
)
def test_corrupt_recurrent_state_refused(kind, make, index):
    rows = make()
    rows[index]["sha256"] = digest(b"wrong state")
    with pytest.raises(ValueError):
        check_state(kind, rows)


def test_frozen_cache_is_not_a_decode_session():
    rows = tiny_rows()
    for row in rows:
        if row["output"] == 1:
            row["sha256"] = digest(b"unchanged cache")
    with pytest.raises(ValueError, match="every append"):
        check_state("tinyllama", rows)


def test_mean_error_cannot_hide_bad_worst_case_or_top_token():
    baseline = {"mean_kl_nats": 0.1, "worst_kl_nats": 0.2, "top_token_matches": 28}
    assert all(gates(baseline, baseline).values())
    for changed in (
        {"worst_kl_nats": 0.22},
        {"top_token_matches": 27},
        {"mean_kl_nats": 0.11},
    ):
        assert not all(gates(baseline | changed, baseline).values())


def test_quality_checks_every_position_and_rejects_nonfinite_logits():
    expected = np.zeros((33, 32000), np.float32)
    actual = expected.copy()
    actual[-1, -1] = 100
    assert quality(actual, expected)["top_token_matches"] == 32
    actual[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        quality(actual, expected)
