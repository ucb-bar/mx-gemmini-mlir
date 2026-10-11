"""Small stage-probe checks that do not need a bare-metal toolchain."""

from __future__ import annotations

import numpy as np
import pytest
import hashlib

from mx_gemmini_support.target_profile import load_profile
from tools.build_merlin_stage import (
    _can_hash_first_output,
    _observe_spike_evidence,
    _raw_selected_regions,
    _require_source_coverage,
)
from test_merlin_bf16_prepare import ROOT, SOURCE


def test_first_output_digest_matches_merlin_supported_dtype(tmp_path) -> None:
    golden = tmp_path / "golden.npy"
    np.save(golden, np.zeros((2,), dtype=np.float32))
    assert _can_hash_first_output(tmp_path)
    np.save(golden, np.zeros((2,), dtype=np.bool_))
    assert _can_hash_first_output(tmp_path)
    np.save(golden, np.zeros((2,), dtype=np.int64))
    assert _can_hash_first_output(tmp_path)
    np.save(golden, np.zeros((2,), dtype=np.float16))
    assert not _can_hash_first_output(tmp_path)
    golden.unlink()
    with pytest.raises(ValueError, match="golden.npy"):
        _can_hash_first_output(tmp_path)


def test_stage_refuses_partial_catalog_after_merlin_preparation(tmp_path, monkeypatch) -> None:
    pytest.importorskip("merlin.system.offload")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(ROOT))
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")
    source = tmp_path / "model.mlir"
    source.write_text(SOURCE, encoding="utf-8")
    expected = _raw_selected_regions(source, profile)
    assert expected == {"matmul_0"}
    with pytest.raises(RuntimeError, match="missing=\\['matmul_0'\\]"):
        _require_source_coverage(expected, {"bindings": [], "unbound_precision_proposals": []})
    _require_source_coverage(expected, {
        "bindings": [{"region": "matmul_0"}], "unbound_precision_proposals": []})


def test_completed_spike_evidence_covers_all_result_shapes(tmp_path, monkeypatch) -> None:
    from merlin.common import mlir_query

    monkeypatch.setattr(mlir_query, "forward_signature",
                        lambda _source: ([], [([1], "f32"), ([2], "bf16")]))
    first = hashlib.sha256(np.asarray([1.0], dtype="<f4").tobytes()).hexdigest()
    state = hashlib.sha256(b"\x00" * 4).hexdigest()
    console = (f"OUT 1 1065353216\nOUT_SHA256 f32le 1 4 {first}\n"
               f"OUT_SHA256_INDEX 0 rawbytes 1 4 {first}\n"
               f"OUT_SHA256_INDEX 1 rawbytes 2 4 {state}\n"
               "METRIC build_hash deadbeef\nMETRIC memref_rank_mismatch 0\nDONE\n")
    result = _observe_spike_evidence(console, tmp_path / "model.mlir", "deadbeef",
                                     first_typed=True)
    assert len(result["indexed"]) == 2
    assert result["numerical_qualification"] == "not_established"
    numeric_hash = console.replace("deadbeef", "000000000001")
    assert _observe_spike_evidence(numeric_hash, tmp_path / "model.mlir", "000000000001",
                                   first_typed=True)["indexed"][0]["sha256"] == first
    with pytest.raises(ValueError, match="linked stage"):
        _observe_spike_evidence(console, tmp_path / "model.mlir", "wrong",
                                first_typed=True)
    with pytest.raises(ValueError, match="count"):
        _observe_spike_evidence(console.replace(
            f"OUT_SHA256_INDEX 1 rawbytes 2 4 {state}\n", ""),
            tmp_path / "model.mlir", "deadbeef", first_typed=True)
