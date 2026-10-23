"""Large-vocabulary diagnostics must cover the entire recurrent trajectory."""

import hashlib
import math

import numpy as np
import pytest

from tools.evaluate_autoregressive_session import (
    check_state,
    metrics,
    verify_target_identity,
)


def state_rows(vocabulary, shape):
    rows = []
    steps = [(0, 0, 128)] + [(1, s, 129 + s) for s in range(32)]
    for program, step, position in steps:
        values = [
            (vocabulary * 4, b"logits" + bytes([position])),
            (math.prod(shape) * 4, b"cache" + bytes([position])),
            (shape[1] * 8, np.full(shape[1], position, dtype="<i8").tobytes()),
        ]
        for output, (size, raw) in enumerate(values):
            rows.append(
                {
                    "program": program,
                    "step": step,
                    "output": output,
                    "bytes": size,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    return rows


@pytest.mark.parametrize(
    "vocabulary,shape",
    [
        (151936, [2, 28, 1, 2, 160, 128]),
        (256000, [2, 26, 1, 4, 160, 256]),
    ],
)
def test_complete_decoder_state(vocabulary, shape):
    result = check_state(
        state_rows(vocabulary, shape),
        vocabulary=vocabulary,
        cache_shape=shape,
        prefill_tokens=128,
        decode_steps=32,
    )
    assert result == {
        "position_vectors_checked": 33,
        "distinct_complete_kv_states": 33,
        "final_position_per_layer": 160,
    }


@pytest.mark.parametrize(
    "damage",
    [
        "last_position",
        "frozen_cache",
        "short_logits",
        "short_cache",
        "missing",
        "duplicate",
    ],
)
def test_corrupt_or_missing_state_is_rejected(damage):
    vocabulary, shape = 151936, [2, 28, 1, 2, 160, 128]
    rows = state_rows(vocabulary, shape)
    if damage == "last_position":
        rows[-1]["sha256"] = rows[2]["sha256"]
    elif damage == "frozen_cache":
        for r in rows:
            if r["output"] == 1:
                r["sha256"] = rows[1]["sha256"]
    elif damage == "short_logits":
        rows[-3]["bytes"] -= 4
    elif damage == "short_cache":
        rows[-2]["bytes"] -= 4
    elif damage == "missing":
        rows.pop()
    else:
        rows.append(rows[-1].copy())
    with pytest.raises(ValueError):
        check_state(
            rows,
            vocabulary=vocabulary,
            cache_shape=shape,
            prefill_tokens=128,
            decode_steps=32,
        )


def test_probability_metrics_ignore_uniform_logit_offset():
    reference = np.array([[2, 1, -1], [-2, 1, 3]], np.float32)
    result = metrics(reference + np.float32(8), reference)
    assert result["mean_kl_nats"] == pytest.approx(0, abs=1e-14)
    assert result["top_token_matches"] == 2
    assert result["minimum_logit_cosine"] < 1


def test_last_vocabulary_word_and_last_step_are_checked():
    reference = np.ones((33, 17), np.float32)
    actual = reference.copy()
    actual[-1, -1] = 10
    result = metrics(actual, reference)
    assert result["per_position_kl_nats"][-1] > 1
    assert result["worst_kl_nats"] > result["mean_kl_nats"]
    actual[-1, -1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        metrics(actual, reference)


def test_partial_or_narrowed_reference_is_rejected():
    reference = np.ones((33, 17), np.float32)
    for actual in [reference[:-1], reference[:, :-1], reference.astype(np.float16)]:
        with pytest.raises(ValueError):
            metrics(actual, reference)


@pytest.mark.parametrize("changed", ["profile", "spike", "extension"])
def test_simulator_identity_cannot_drift_from_the_predeclared_target(changed):
    identity = {
        "profile_sha256": "a" * 64,
        "spike": {"sha256": "b" * 64},
        "extension": {"sha256": "c" * 64},
    }
    actual = {"profile": "a" * 64, "spike": "b" * 64, "extension": "c" * 64}
    verify_target_identity(identity, **actual)
    actual[changed] = "d" * 64
    with pytest.raises(ValueError, match="predeclared"):
        verify_target_identity(identity, **actual)
