"""Bind scalar epilogues from digest-checked model2MLIR graphs."""

from __future__ import annotations

from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mx_gemmini_support.capture_epilogue import append_captured_tilewise_vpu_muls
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
CASES = (
    "radiance_tilewise_vpu_x2_266c593",
    "radiance_tilewise_vpu_scalar_266c593",
    "radiance_fp4_generated_tilewise_vpu_266c593",
    "radiance_fp4_generated_tilewise_vpu_scalar_266c593",
)


def _read(root: Path, name: str) -> bytes:
    path = root / name
    return path.read_bytes() if path.exists() else gzip.decompress(
        (root / f"{name}.gz").read_bytes())


def _case(name: str):
    root = EVIDENCE / name
    manifest, _ = load_bundle(root / "bundle")
    capture = SimpleNamespace(
        ok=True,
        capture_trace=json.loads(_read(root, "capture_trace.json")),
        quantization_manifest=json.loads(_read(root, "quantization_manifest.json")),
        mlir_text=_read(root, "frontend.mlir").decode())
    return root, manifest, capture


@pytest.mark.parametrize("name", CASES)
def test_captured_epilogue_recovers_archived_typed_mlir(name: str):
    root, manifest, capture = _case(name)
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    bound = append_captured_tilewise_vpu_muls(
        _read(root, "payload_bound.mlir").decode(), profile, manifest, capture)
    assert bound == _read(root, "tilewise_bound.mlir").decode()


def test_captured_epilogue_rejects_graph_and_frontend_drift():
    root, manifest, capture = _case("radiance_tilewise_vpu_scalar_266c593")
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    payload = _read(root, "payload_bound.mlir").decode()

    changed = deepcopy(capture)
    changed.capture_trace["graphs"]["original"]["nodes"][3]["args"][1] = 2.0
    with pytest.raises(ValueError):
        append_captured_tilewise_vpu_muls(payload, profile, manifest, changed)

    changed = deepcopy(capture)
    changed.capture_trace["graphs"]["original"]["nodes"][4]["args"] = []
    with pytest.raises(ValueError):
        append_captured_tilewise_vpu_muls(payload, profile, manifest, changed)

    changed = deepcopy(capture)
    changed.mlir_text += "\n"
    with pytest.raises(ValueError):
        append_captured_tilewise_vpu_muls(payload, profile, manifest, changed)

    changed = deepcopy(capture)
    changed.quantization_manifest["sites"][0]["format"] = "mxfp4"
    with pytest.raises(ValueError):
        append_captured_tilewise_vpu_muls(payload, profile, manifest, changed)
