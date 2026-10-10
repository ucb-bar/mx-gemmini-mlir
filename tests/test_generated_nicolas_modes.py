"""Pin the independent data model used for five missing DIM16 MX cells."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_generated_modes_266c593"
MODES = {
    "e2m3_e4m3s", "e3m2_e3m2", "e4m3s_e2m3",
    "e4m3s_e4m3", "e4m3_e4m3s",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generated_headers_have_pinned_model_provenance() -> None:
    manifest = json.loads(
        (EVIDENCE / "mx_gemmini_generated_modes_manifest.json").read_text())
    assert manifest["schema"] == "mx_gemmini.nicolas_generated_asymmetric_headers.v1"
    assert manifest["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert manifest["microxcaling_revision"] == (
        "7bc41952de394f5cc5e782baf132e7c7542eb4e4")
    assert manifest["baseline_sha256"] == (
        "b4c7f87d11a0e096bc8ccfe2234c88c992ac5d2bccdb7b4d09c9aca0d770d8dc")
    assert manifest["wrapper_sha256"] == _sha(
        ROOT / "tools/generate_nicolas_missing_headers.py")
    assert set(manifest["headers_sha256"]) == MODES
    for name, digest in manifest["headers_sha256"].items():
        assert _sha(EVIDENCE / f"matmul_data_asym_{name}.h") == digest
