"""Pin the DIM8/DIM32 generated headers to Nicolas's model and profile gaps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FORMAT = {
    "fp4": ("fp4_e2m1", "direct"),
    "e2m3": ("fp6_e2m3", "lut"),
    "e3m2": ("fp6_e3m2", "lut"),
    "e4m3": ("fp8_e4m3", "lut"),
    "e4m3s": ("fp8_e4m3", "direct"),
    "e5m2": ("fp8_e5m2", "lut"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("dim,baseline", [
    (8, "0ae643c6be0e288d9d9b3cb9c163ac8c669b0fb8941cee85cd1a7daa35c94b83"),
    (32, "13730bd97ec1306bb93d11f7cb7464dd5ce5b1b0a111f524c5679524c9a507b5"),
])
def test_generated_mesh_headers_cover_exact_missing_source_modes(
        dim: int, baseline: str) -> None:
    folder = ROOT / f"docs/evidence/nicolas_generated_mesh_dim{dim}_266c593"
    manifest = json.loads((folder / "generation_manifest.json").read_text())
    assert manifest["schema"] == "mx_gemmini.nicolas_generated_mesh_headers.v1"
    assert manifest["mesh_dim"] == dim
    assert manifest["rtl_revision"].startswith("266c593")
    assert manifest["software_revision"].startswith("350547f")
    assert manifest["microxcaling_revision"] == (
        "7bc41952de394f5cc5e782baf132e7c7542eb4e4")
    assert manifest["baseline_sha256"] == baseline
    assert manifest["wrapper_sha256"] == _sha(
        ROOT / "tools/generate_nicolas_mesh_headers.py")
    assert len(manifest["headers_sha256"]) == 15
    assert set(manifest["header_origin"]) == set(manifest["headers_sha256"])
    for name, digest in manifest["headers_sha256"].items():
        assert _sha(folder / f"matmul_data_asym_{name}_dim{dim}.h") == digest
        assert manifest["header_origin"][name] in {"checked_in", "generated"}

    profile = json.loads((ROOT / "profiles/gemmini-mx-cleanup-266c593" /
                          f"MxDim{dim}AllAsymGemminiRocketConfig.json").read_text())
    previous = json.loads((ROOT / "docs/evidence" /
                           f"nicolas_asym_matrix_dim{dim}_266c593/matrix_first.json").read_text())
    missing = {json.dumps(row["compute"], sort_keys=True)
               for row in previous["uncovered_legal_compute"]}
    assert len(missing) == 15
    selected = set()
    for name in manifest["headers_sha256"]:
        left, right = name.split("_")
        af, ap = FORMAT[left]
        wf, wp = FORMAT[right]
        cells = [cell for cell in profile["legal_compute"] if
                 (cell["activation_format"], cell["activation_projection"],
                  cell["weight_format"], cell["weight_projection"]) ==
                 (af, ap, wf, wp)]
        assert len(cells) == 1
        selected.add(json.dumps(cells[0], sort_keys=True))
    assert selected == missing
