"""Captured tensor and bias adds use source-bound VPU calls."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from mx_gemmini_support.merlin_vpu_add import (
    MXVPUAddPostOffload, emit_memref_add_shim, selected_add_sites,
)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.vpu_add_kernel import emit_vpu_add


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984"
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"
VPU = load_profile(PROFILES / "MxE4M3Fp4VpuGemminiRocketConfig.json")
NO_VPU = load_profile(PROFILES / "MxGemminiRocketConfig.json")


def test_add_kernel_chunks_tail_and_bias_abi() -> None:
    kernel = emit_vpu_add(VPU, elements=135 * 113, row_width=113,
                          dtype="bf16", broadcast="last_dimension",
                          symbol="mx_add_probe")
    assert kernel.elements_per_chunk == 8192
    assert "case 7063:" in kernel.source
    assert "(base + index) % 113" in kernel.source
    shim = emit_memref_add_shim({
        "shape": (1, 135, 113), "rhs_shape": (113,),
        "dtype": "bf16", "symbol": "mx_add_probe",
        "kernel_symbol": "mx_add_probe_kernel",
    })
    assert "b_st0 != 1" in shim
    assert "_mlir_ciface_mx_add_probe" in shim
    with pytest.raises(ValueError, match="DIM16 Rocket MX VPU"):
        emit_vpu_add(NO_VPU, elements=128, row_width=8, dtype="bf16",
                     broadcast="none", symbol="unavailable")


@pytest.mark.parametrize("model,expected", (
    ("tinyllama", 13), ("deepseek_r1_distill_qwen_1_5b", 10),
    ("gemma2_2b", 17), ("smolvla", 114),
))
def test_frozen_add_sites_are_source_bound(model: str, expected: int) -> None:
    pytest.importorskip("merlin.frontends.linalg_mlir")
    source = gzip.decompress((ARCHIVE / f"{model}.mlir.gz").read_bytes()).decode()
    sites = selected_add_sites(source, VPU)
    assert len(sites) == expected
    assert len({site["region"] for site in sites}) == expected
    assert selected_add_sites(source, NO_VPU) == []


@pytest.mark.parametrize("model,expected", (("tinyllama", 13), ("smolvla", 114)))
def test_frozen_adds_rewrite_to_external_calls(
        tmp_path: Path, model: str, expected: int) -> None:
    pytest.importorskip("merlin.frontends.linalg_mlir")
    from merlin.frontends.linalg_mlir import parse_mlir_file

    source = gzip.decompress((ARCHIVE / f"{model}.mlir.gz").read_bytes()).decode()
    prepared = tmp_path / "source.mlir"
    prepared.write_text(source)
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text("{}")
    output = MXVPUAddPostOffload(VPU)(prepared, sidecar, tmp_path / "rewrite")
    module = parse_mlir_file(output)
    calls = [op for op in module.walk() if op.name == "func.call" and
             "mx_vpu_add_" in str(getattr(op, "callee", ""))]
    assert len(calls) == expected
    receipt = json.loads((output.parent / "mx_vpu_add_rewrite.json").read_text())
    assert len(receipt["regions"]) == expected
