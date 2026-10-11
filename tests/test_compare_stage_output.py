"""A complete console dump is required before numerical comparisons."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tools.compare_stage_output import compare_output


pytest.importorskip("merlin.runtime.backends.spike_model")


def _console(values: np.ndarray) -> str:
    bits = values.astype("<f4").view("<u4")
    words = " ".join(str(int(word)) for word in bits.ravel())
    digest = hashlib.sha256(values.astype("<f4").tobytes()).hexdigest()
    return (f"OUT {values.size} {words}\n"
            f"OUT_SHA256 f32le {values.size} {values.nbytes} {digest}\nDONE\n")


def test_complete_float_output_reports_diagnostic_metrics() -> None:
    target = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    reference = np.asarray([[1.0, 3.0, 2.0]], dtype=np.float32)
    report = compare_output(_console(target), reference, kind="mx_operand_fake_quant")
    assert report["scope"] == "first_result_only"
    assert report["numerical_qualification"] == "not_established"
    assert report["metrics"]["max_abs"] == 1.0
    assert report["metrics"]["target_argmax"] == [2]
    assert report["metrics"]["reference_argmax"] == [1]
    assert not report["metrics"]["argmax_equal"]


def test_truncated_or_corrupt_output_cannot_be_compared() -> None:
    target = np.asarray([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="incomplete"):
        compare_output(_console(target), np.asarray([1.0, 2.0], dtype=np.float32),
                       kind="unquantized_pytorch")
    corrupt = _console(target).replace(hashlib.sha256(target.tobytes()).hexdigest(), "0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        compare_output(corrupt, target, kind="independent_mx_arithmetic")
