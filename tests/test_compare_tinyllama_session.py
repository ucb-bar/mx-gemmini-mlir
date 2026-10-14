"""TinyLlama logit diagnostics keep token ranks and shapes explicit."""

from __future__ import annotations

import numpy as np
import pytest

from tools.compare_tinyllama_session import metrics


def test_logit_metrics_report_top_token_and_reference_rank() -> None:
    reference = np.zeros((1, 32000), dtype=np.float32)
    reference[0, 7] = 2.0
    reference[0, 11] = 1.0
    same = metrics(reference.copy(), reference)
    assert same["cosine"] == pytest.approx(1.0)
    assert same["rmse"] == 0.0
    assert same["top_token_equal"] is True
    assert same["reference_top_rank_in_actual"] == 1

    observed = reference.copy()
    observed[0, 11] = 3.0
    changed = metrics(observed, reference)
    assert changed["actual_top_token"] == 11
    assert changed["reference_top_token"] == 7
    assert changed["top_token_equal"] is False
    assert changed["reference_top_rank_in_actual"] == 2


def test_logit_metrics_reject_wrong_shape_and_nonfinite_values() -> None:
    reference = np.zeros((1, 32000), dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        metrics(reference[:, :-1], reference)
    observed = reference.copy()
    observed[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        metrics(observed, reference)
