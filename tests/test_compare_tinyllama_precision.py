"""Precision comparisons must use the same captured TinyLlama work."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.compare_tinyllama_precision import compare


def _write_run(path, *, fp8_sites: int, shift: float) -> None:
    path.mkdir()
    baseline = np.zeros((1, 32000), dtype=np.float32)
    baseline[0, :3] = [1, 2, 3]
    proxy = baseline.copy()
    proxy[0, 0] += shift
    np.save(path / "baseline.npy", baseline)
    np.save(path / "proxy.npy", proxy)
    receipt = {
        "schema": "mx_gemmini.tinyllama_operand_fake_quant.v1",
        "status": "diagnostic_not_mesh_oracle",
        "capture_model_sha256": "a" * 64,
        "capture_weights_sha256": "b" * 64,
        "capture_inputs_sha256": "c" * 64,
        "baseline_logits_sha256": hashlib.sha256(baseline.tobytes()).hexdigest(),
        "proxy_logits_sha256": hashlib.sha256(proxy.tobytes()).hexdigest(),
        "baseline_top_token": 2,
        "proxy_top_token": int(proxy.argmax()),
        "stage_kind": "prefill", "layers": 22, "capacity": 160,
        "fp8_attention_calls": 44, "fp8_rotary_bmm_calls": 1,
        "fp4_linear_sites": 155 - fp8_sites,
        "fp8_linear_sites": fp8_sites,
        "precision_overrides_sha256": "d" * 64 if fp8_sites else None,
    }
    (path / "result.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_precision_comparison_checks_shared_capture_and_array_digests(tmp_path) -> None:
    base, promoted = tmp_path / "base", tmp_path / "promoted"
    _write_run(base, fp8_sites=0, shift=0)
    _write_run(promoted, fp8_sites=22, shift=0.5)
    report = compare({"base": base, "promoted": promoted})
    assert report["reference_top_token"] == 2
    assert report["runs"][0]["mae"] == 0
    assert report["runs"][1]["fp8_linear_sites"] == 22
    assert report["runs"][1]["max_abs"] == 0.5
    receipt_path = promoted / "result.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["capture_inputs_sha256"] = "changed"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="different captured work"):
        compare({"base": base, "promoted": promoted})
