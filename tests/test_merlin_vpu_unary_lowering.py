"""Captured unary operations have exact source-bound VPU calls."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from mx_gemmini_support.merlin_vpu_unary import (
    MXVPUUnaryPostOffload, emit_memref_unary_shim, selected_unary_sites,
)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.vpu_unary_kernel import emit_vpu_unary


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984"
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"
VPU = load_profile(PROFILES / "MxE4M3Fp4VpuGemminiRocketConfig.json")
NO_VPU = load_profile(PROFILES / "MxGemminiRocketConfig.json")


def test_unary_kernel_and_ranked_shim() -> None:
    kernel = emit_vpu_unary(VPU, shape=(1, 50, 1), dtype="f32",
                            operation="rsqrt", symbol="mx_rsqrt_probe_kernel")
    assert kernel.elements_per_chunk == 50
    assert "case 50:" in kernel.source
    shim = emit_memref_unary_shim({"shape": (1, 50, 1), "dtype": "f32",
                                   "symbol": "mx_rsqrt_probe",
                                   "kernel_symbol": "mx_rsqrt_probe_kernel"})
    assert "a_st0 != 50" in shim
    assert "_mlir_ciface_mx_rsqrt_probe" in shim
    sigmoid = emit_vpu_unary(VPU, shape=(1, 8, 5632), dtype="f32",
                             operation="sigmoid", symbol="mx_sigmoid_probe_kernel")
    assert "case 4096:" in sigmoid.source
    assert sigmoid.source.count('.insn r 0x7b, 3, 33') == 8
    tanh = emit_vpu_unary(VPU, shape=(1, 16), dtype="f32",
                          operation="tanh", symbol="mx_tanh_probe_kernel")
    assert "case 16:" in tanh.source
    assert tanh.source.count('.insn r 0x7b, 3, 33') == 6
    square = emit_vpu_unary(VPU, shape=(1, 8, 2048), dtype="f32",
                            operation="square", symbol="mx_square_probe_kernel")
    assert "case 8192:" in square.source
    neg = emit_vpu_unary(VPU, shape=(1, 8, 64), dtype="f32",
                         operation="neg", symbol="mx_neg_probe_kernel")
    assert "case 512:" in neg.source
    gelu = emit_vpu_unary(VPU, shape=(1, 8, 64), dtype="f32",
                          operation="gelu", symbol="mx_gelu_probe_kernel")
    assert "case 512:" in gelu.source
    power = emit_vpu_unary(VPU, shape=(32,), dtype="f32",
                           operation="pow10000", symbol="mx_power_probe_kernel")
    assert "case 32:" in power.source
    with pytest.raises(ValueError, match="DIM16 Rocket MX VPU"):
        emit_vpu_unary(NO_VPU, shape=(8,), dtype="bf16",
                       operation="rsqrt", symbol="unavailable")


@pytest.mark.parametrize("model,expected", (
    ("tinyllama", 16), ("deepseek_r1_distill_qwen_1_5b", 9),
    ("gemma2_2b", 23), ("smolvla", 44),
))
def test_frozen_unary_sites_are_source_bound(model: str, expected: int) -> None:
    pytest.importorskip("merlin.frontends.linalg_mlir")
    source = gzip.decompress((ARCHIVE / f"{model}.mlir.gz").read_bytes()).decode()
    sites = selected_unary_sites(source, VPU)
    assert len(sites) == expected
    assert len({site["region"] for site in sites}) == expected
    assert selected_unary_sites(source, NO_VPU) == []


@pytest.mark.parametrize("model,expected", (("tinyllama", 16), ("gemma2_2b", 23), ("smolvla", 44)))
def test_frozen_unary_rewrites_to_external_calls(
        tmp_path: Path, model: str, expected: int) -> None:
    pytest.importorskip("merlin.frontends.linalg_mlir")
    from merlin.frontends.linalg_mlir import parse_mlir_file

    source = gzip.decompress((ARCHIVE / f"{model}.mlir.gz").read_bytes()).decode()
    prepared = tmp_path / "source.mlir"
    prepared.write_text(source)
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text("{}")
    output = MXVPUUnaryPostOffload(VPU)(prepared, sidecar, tmp_path / "rewrite")
    module = parse_mlir_file(output)
    calls = [op for op in module.walk() if op.name == "func.call" and
             any(f"mx_vpu_{operation}_" in str(getattr(op, "callee", ""))
                 for operation in ("rsqrt", "sigmoid", "tanh", "square", "neg", "gelu", "pow10000"))]
    assert len(calls) == expected
    receipt = json.loads((output.parent / "mx_vpu_unary_rewrite.json").read_text())
    assert len(receipt["regions"]) == expected
