"""Captured rotary and gating arithmetic has exact VPU source bindings."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import pytest

from mx_gemmini_support.merlin_vpu_elementwise import (
    MXVPUElementwisePostOffload, selected_elementwise_sites,
)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.vpu_add_kernel import emit_vpu_elementwise


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984"
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"
VPU = load_profile(PROFILES / "MxE4M3Fp4VpuGemminiRocketConfig.json")
NO_VPU = load_profile(PROFILES / "MxGemminiRocketConfig.json")


def test_rotary_broadcast_and_subtraction_emit_vpu_programs() -> None:
    rotary = emit_vpu_elementwise(
        VPU, result_shape=(1, 50, 15, 32), rhs_shape=(1, 50, 1, 32),
        dtype="f32", operation="mul", symbol="mx_rotary_probe")
    assert rotary.elements == 24000
    assert rotary.elements_per_chunk == 8192
    assert "((base + index) / 480u % 50u)" in rotary.source
    assert "((base + index) % 32u)" in rotary.source
    assert "case 7616:" in rotary.source
    sub = emit_vpu_elementwise(
        VPU, result_shape=(1, 50, 15, 32), rhs_shape=(1, 50, 15, 32),
        dtype="f32", operation="sub", symbol="mx_sub_probe")
    assert sub.broadcast == "none"
    assert sub.source != rotary.source
    conv_bias = emit_vpu_elementwise(
        VPU, result_shape=(1, 768, 4, 4), rhs_shape=(768,),
        rhs_axes=(1,), dtype="f32", operation="add",
        symbol="mx_conv_bias_probe")
    assert "(base + index) / 16u % 768u" in conv_bias.source
    mean_divide = emit_vpu_elementwise(
        VPU, result_shape=(1, 8), rhs_shape=(1, 8),
        dtype="f32", operation="div", symbol="mx_mean_probe")
    assert mean_divide.source.count('.insn r 0x7b, 3, 33') == 2
    scalar = emit_vpu_elementwise(
        VPU, result_shape=(1, 128, 2304), rhs_shape=(), rhs_axes=(),
        dtype="f32", operation="mul", symbol="mx_embedding_scale_probe")
    assert "rhs[0u]" in scalar.source
    two_sided = emit_vpu_elementwise(
        VPU, result_shape=(1, 8, 32),
        lhs_shape=(1, 8, 1), lhs_axes=(0, 1, 2),
        rhs_shape=(1, 1, 32), rhs_axes=(0, 1, 2),
        dtype="f32", operation="div", symbol="mx_rotary_outer_probe")
    assert "lhs[(((base + index) / 32u % 8u) * 1u)]" in two_sided.source
    assert "rhs[(((base + index) % 32u) * 1u)]" in two_sided.source
    with pytest.raises(ValueError, match="broadcast"):
        emit_vpu_elementwise(VPU, result_shape=(1, 4, 32),
                             rhs_shape=(1, 5, 32), dtype="f32",
                             operation="mul", symbol="bad")


@pytest.mark.parametrize("model,expected,mul,sub,add,div,swaps", (
    ("tinyllama", 31, 26, 0, 0, 5, 5),
    ("deepseek_r1_distill_qwen_1_5b", 18, 15, 0, 0, 3, 3),
    ("gemma2_2b", 53, 34, 0, 9, 10, 0),
    ("smolvla", 271, 129, 29, 53, 60, 5),
))
def test_frozen_elementwise_sites_are_source_bound(
        model: str, expected: int, mul: int, sub: int, add: int, div: int,
        swaps: int) -> None:
    pytest.importorskip("merlin.frontends.linalg_mlir")
    source = gzip.decompress((ARCHIVE / f"{model}.mlir.gz").read_bytes()).decode()
    sites = selected_elementwise_sites(source, VPU)
    assert len(sites) == expected
    assert Counter(site["operation"] for site in sites) == {
        key: count for key, count in (("mul", mul), ("sub", sub),
                                     ("add", add), ("div", div)) if count}
    assert sum(site["swap"] for site in sites) == swaps
    assert len({site["region"] for site in sites}) == expected
    assert all(site["dtype"] != "f32" or site["elements"] != 1 for site in sites)
    assert selected_elementwise_sites(source, NO_VPU) == []


@pytest.mark.parametrize("model,expected", (("tinyllama", 31), ("smolvla", 271),
                                            ("gemma2_2b", 53)))
def test_frozen_elementwise_rewrites_to_external_calls(
        tmp_path: Path, model: str, expected: int) -> None:
    pytest.importorskip("merlin.frontends.linalg_mlir")
    from merlin.frontends.linalg_mlir import parse_mlir_file

    source = gzip.decompress((ARCHIVE / f"{model}.mlir.gz").read_bytes()).decode()
    prepared = tmp_path / "source.mlir"
    prepared.write_text(source)
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text("{}")
    output = MXVPUElementwisePostOffload(VPU)(
        prepared, sidecar, tmp_path / "rewrite")
    module = parse_mlir_file(output)
    calls = [op for op in module.walk() if op.name == "func.call" and
             any(f"mx_vpu_{kind}_" in str(getattr(op, "callee", ""))
                 for kind in ("mul", "sub", "add", "div"))]
    assert len(calls) == expected
    receipt = json.loads((output.parent / "mx_vpu_elementwise_rewrite.json").read_text())
    assert len(receipt["regions"]) == expected
