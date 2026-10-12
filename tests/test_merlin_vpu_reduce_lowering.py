"""Normalization reductions use identity-checked source-bound VPU calls."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from mx_gemmini_support.merlin_vpu_reduce import (
    MXVPUReducePostOffload, emit_memref_reduce_shim,
    selected_row_reduce_sites,
)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.vpu_row_reduce import emit_vpu_row_reduce


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984"
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"
VPU = load_profile(PROFILES / "MxE4M3Fp4VpuGemminiRocketConfig.json")
NO_VPU = load_profile(PROFILES / "MxGemminiRocketConfig.json")


def test_row_reduce_kernel_and_ranked_shim() -> None:
    kernel = emit_vpu_row_reduce(VPU, logical_rows=16, row_width=768,
                                 operation="rsum", symbol="mx_reduce_probe_kernel")
    assert (kernel.padded_width, kernel.rows_per_chunk) == (768, 10)
    assert "case 6:" in kernel.source
    shim = emit_memref_reduce_shim({
        "shape": (1, 16, 768), "output_shape": (1, 16),
        "symbol": "mx_reduce_probe", "kernel_symbol": "mx_reduce_probe_kernel",
    })
    assert "a_st0 != 12288" in shim
    assert "b_st0 != 16" in shim
    assert "_mlir_ciface_mx_reduce_probe" in shim
    with pytest.raises(ValueError, match="DIM16 Rocket MX VPU"):
        emit_vpu_row_reduce(NO_VPU, logical_rows=8, row_width=32,
                            operation="rsum", symbol="unavailable")


@pytest.mark.parametrize("model,expected", (
    ("tinyllama", 5), ("deepseek_r1_distill_qwen_1_5b", 3),
    ("gemma2_2b", 9), ("smolvla", 56),
))
def test_frozen_reductions_exclude_fused_softmax(
        model: str, expected: int) -> None:
    pytest.importorskip("merlin.frontends.linalg_mlir")
    source = gzip.decompress((ARCHIVE / f"{model}.mlir.gz").read_bytes()).decode()
    sites = selected_row_reduce_sites(source, VPU)
    assert len(sites) == expected
    assert len({site["region"] for site in sites}) == expected
    assert {site["operation"] for site in sites} == {"rsum"}
    assert selected_row_reduce_sites(source, NO_VPU) == []


@pytest.mark.parametrize("model,expected", (("tinyllama", 5), ("smolvla", 56)))
def test_frozen_reductions_rewrite_to_external_calls(
        tmp_path: Path, model: str, expected: int) -> None:
    pytest.importorskip("merlin.frontends.linalg_mlir")
    from merlin.frontends.linalg_mlir import parse_mlir_file

    source = gzip.decompress((ARCHIVE / f"{model}.mlir.gz").read_bytes()).decode()
    prepared = tmp_path / "source.mlir"
    prepared.write_text(source)
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text("{}")
    output = MXVPUReducePostOffload(VPU)(prepared, sidecar, tmp_path / "rewrite")
    module = parse_mlir_file(output)
    calls = [op for op in module.walk() if op.name == "func.call" and
             "mx_vpu_rsum_" in str(getattr(op, "callee", ""))]
    assert len(calls) == expected
    receipt = json.loads((output.parent / "mx_vpu_reduce_rewrite.json").read_text())
    assert len(receipt["regions"]) == expected
