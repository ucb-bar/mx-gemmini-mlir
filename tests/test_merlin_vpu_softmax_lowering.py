"""Captured row softmax regions must become source-bound VPU calls."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from mx_gemmini_support.merlin_vpu_softmax import (
    MXVPUSoftmaxPostOffload, emit_memref_softmax_shim,
    selected_softmax_sites,
)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.vpu_row_softmax import emit_vpu_row_softmax


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984"
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"
VPU = load_profile(PROFILES / "MxE4M3Fp4VpuGemminiRocketConfig.json")
NO_VPU = load_profile(PROFILES / "MxGemminiRocketConfig.json")


def test_model_sized_softmax_tiles_and_shim() -> None:
    kernel = emit_vpu_row_softmax(VPU, logical_rows=1695, row_width=113,
                                  symbol="mx_vpu_softmax_1_15_113_113_kernel")
    assert (kernel.padded_width, kernel.rows_per_chunk) == (120, 68)
    assert kernel.vpu_steps == ("rmax", "sub", "exp", "rsum", "rcp", "mul")
    assert kernel.source.count('.insn r 0x7b, 3, 33') == 12
    shim = emit_memref_softmax_shim((1, 15, 113, 113))
    assert "a_st0 != 191535" in shim
    assert "b_st3 != 1" in shim
    assert "mx_vpu_softmax_1_15_113_113_kernel" in shim
    with pytest.raises(ValueError, match="DIM16 Rocket MX VPU"):
        emit_vpu_row_softmax(NO_VPU, logical_rows=16, row_width=32,
                             symbol="unavailable")


@pytest.mark.parametrize("model,expected", (
    ("tinyllama", 2), ("deepseek_r1_distill_qwen_1_5b", 1),
    ("gemma2_2b", 2), ("smolvla", 14),
))
def test_frozen_source_softmax_rewrites_to_external_vpu_calls(
        tmp_path: Path, model: str, expected: int) -> None:
    pytest.importorskip("merlin.frontends.linalg_mlir")
    from merlin.frontends.linalg_mlir import parse_mlir_file

    source = gzip.decompress((ARCHIVE / f"{model}.mlir.gz").read_bytes()).decode()
    sites = selected_softmax_sites(source, VPU)
    assert len(sites) == expected
    assert len({site["shape"] for site in sites}) >= 1
    prepared = tmp_path / "source.mlir"
    prepared.write_text(source)
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text("{}")
    output = MXVPUSoftmaxPostOffload(VPU)(prepared, sidecar, tmp_path / "rewrite")
    module = parse_mlir_file(output)
    calls = [op for op in module.walk() if op.name == "func.call" and
             "mx_vpu_softmax_" in str(getattr(op, "callee", ""))]
    assert len(calls) == expected
    assert not any(getattr(op.attributes.get("prov.op"), "data", None) == "softmax"
                   for op in module.walk())
    receipt = json.loads((output.parent / "mx_vpu_softmax_rewrite.json").read_text())
    assert len(receipt["regions"]) == expected
