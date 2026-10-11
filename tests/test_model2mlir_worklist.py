"""Model2MLIR whole-graph import must retain every contraction family."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import re

import pytest

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxGemminiRocketConfig.json")
EXPECTED = {
    "tinyllama": (20, 15, 5, 0, 0),
    "deepseek_r1_distill_qwen_1_5b": (14, 8, 6, 0, 0),
    "gemma2_2b": (20, 15, 5, 0, 0),
    "smolvla": (198, 93, 105, 14, 7),
}


@pytest.mark.parametrize("model", EXPECTED)
def test_captured_model_contractions_have_stable_mx_worklist(model: str) -> None:
    frontend = gzip.decompress((ARCHIVE / f"{model}.mlir.gz").read_bytes()).decode()
    expected_sha = next(row["sha256"] for row in
                        json.loads((ARCHIVE / "index.json").read_text())["rows"]
                        if row["model"] == model)
    result = build_model2mlir_worklist(frontend, PROFILE)
    regions, rank2, other, bf16, k_padding = EXPECTED[model]
    assert result["schema"] == "mx_gemmini.model2mlir_contraction_worklist.v1"
    assert result["status"] == "frontend_import_only_no_executable_placement"
    assert result["source_mlir_sha256"] == expected_sha
    assert result["function"] == "forward"
    assert result["contraction_region_count"] == regions
    assert len(result["rank2_matmuls"]) == rank2
    assert len(result["other_contraction_regions"]) == other
    assert sum(site["element_type"] == "bf16" for site in result["rank2_matmuls"]) == bf16
    assert sum(site["k_requires_32_element_scale_padding"]
               for site in result["rank2_matmuls"]) == k_padding
    assert all(site["status"] == "requires_explicit_mx_quantization_and_program_lowering"
               for site in result["rank2_matmuls"])
    assert any(cell["activation_format"] == "fp8_e4m3"
               for cell in result["legal_mx_compute_modes"])


def test_worklist_rejects_graph_without_portable_level_or_matmul_provenance() -> None:
    frontend = gzip.decompress((ARCHIVE / "tinyllama.mlir.gz").read_bytes()).decode()
    with pytest.raises(ValueError, match="linalg-on-tensors provenance"):
        build_model2mlir_worklist(
            frontend.replace('prov.level = "linalg-on-tensors"', 'prov.level = "torch"'),
            PROFILE)
    changed = re.sub(
        r'(linalg\.matmul \{[^\n]*?)prov\.family = "contraction"',
        r'\1prov.family = "elementwise"', frontend, count=1)
    assert changed != frontend
    with pytest.raises(ValueError, match="matmul lacks contraction provenance"):
        build_model2mlir_worklist(changed, PROFILE)
    with pytest.raises(ValueError, match="valid supported portable MLIR"):
        build_model2mlir_worklist(frontend.replace("tensor.empty()", "unknown.empty()", 1),
                                 PROFILE)
