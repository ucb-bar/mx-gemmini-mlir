"""New 256-deep source runs retain their model2MLIR capture provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
SOURCE = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))
FP4_SOURCE = Path(os.environ.get("RADIANCE_FP4_GENERATED_ROOT", "/nonexistent"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("precision,source_root,shape", [
    ("fp8", SOURCE, (128, 128, 256)),
    ("fp4", FP4_SOURCE, (128, 128, 256)),
    ("fp8", SOURCE, (256, 256, 256)),
])
def test_fresh_256_deep_capture_binds_source_and_typed_ir(precision, source_root, shape):
    m, n, k = shape
    stem = f"model2mlir_radiance_mx_{precision}_{m}x{n}x{k}"
    capture = json.loads((EVIDENCE / f"{stem}_capture_receipt.json").read_text())
    files = {"source_mlir_sha256": "source.mlir",
             "handoff_mlir_sha256": "handoff.mlir",
             "quantization_manifest_sha256": "manifest.json"}
    for field, suffix in files.items():
        assert capture[field] == _sha(EVIDENCE / f"{stem}_{suffix}")
    assert capture["target_binding"]["bound_mlir_sha256"] == _sha(
        EVIDENCE / f"{stem}_bound.mlir")
    assert capture["model2mlir_revision"] == "7485a829c0195af0ec42820837d609e62e466564"
    assert capture["source_shape"] == list(shape)
    assert capture["selected_site"]["shape"] == list(shape)
    assert capture["source_data_header_origin"] == "generated_or_untracked"
    assert capture["opaque_calls"] == {}
    assert "physical compiler is qualified separately" in capture["lowering_scope"]
    if m == 256:
        assert capture["source_plan_error"] == "C does not fit beside double-buffered A/B tiles"
        assert capture["source_layout"] is None
        assert capture["target_binding"]["target_layout"]["c_spad_dest"] > 0
    else:
        assert capture["source_plan_error"] is None
    header = source_root / f"kernels/gemm_mxgemmini/mxgemm.data.{precision}.m{m}n{n}k{k}.h"
    if header.is_file():
        assert capture["source_data_header_sha256"] == _sha(header)
