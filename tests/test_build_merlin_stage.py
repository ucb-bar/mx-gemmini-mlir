"""Small stage-probe checks that do not need a bare-metal toolchain."""

from __future__ import annotations

import numpy as np
import pytest

from mx_gemmini_support.target_profile import load_profile
from tools.build_merlin_stage import (
    _can_hash_first_output,
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
