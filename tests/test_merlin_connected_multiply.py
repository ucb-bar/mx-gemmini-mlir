"""Whole-stage rewrites retain source policy, ownership, and arithmetic."""

import json
from pathlib import Path

import pytest
from merlin.frontends.linalg_mlir import parse_mlir_file, parse_mlir_text

from mx_gemmini_support.merlin_connected_multiply import (
    emit_chain_kernel,
    emit_chain_shim,
    multiply_pairs,
)
from mx_gemmini_support.merlin_vpu_elementwise import (
    MXVPUElementwisePostOffload,
    selected_elementwise_sites,
)
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")
SOURCE = (ROOT / "docs/evidence/connected_model_section_20261016/tiny_mlp/source.mlir").read_text()


def pairs(source=SOURCE, enabled=None):
    module = parse_mlir_text(source)
    sites = selected_elementwise_sites(source, PROFILE)
    if enabled is not None:
        sites = [site for site in sites if site["region"] in enabled]
    return multiply_pairs(module, sites)


@pytest.mark.parametrize("shape", ["1x1x137", "1x3x2731", "1x64x5632"])
def test_source_pair_uses_shared_legality_and_existing_bf16_adapter(shape):
    pair, = pairs(SOURCE.replace("1x64x5632", shape))
    code, removed = emit_chain_kernel(pair, PROFILE)
    assert removed > 0
    assert "if(!safe)" in code
    assert pair["kernel_symbol"] in code
    assert "run_chain(" not in code
    assert "_mlir_ciface_" + pair["symbol"] in emit_chain_shim(pair)


def test_placing_either_multiply_on_host_prevents_connection():
    assert not pairs(enabled={"mul_0#mul0"})
    assert not pairs(enabled={"mul_1#mul0"})


@pytest.mark.parametrize("change", ["arithmetic", "operand", "fastmath"])
def test_names_cannot_override_actual_arithmetic(change):
    if change == "arithmetic":
        source = SOURCE.replace('"arith.mulf"', '"arith.addf"', 1)
    elif change == "operand":
        source = SOURCE.replace('"arith.mulf"(%5, %6)', '"arith.mulf"(%5, %7)', 1)
    else:
        source = SOURCE.replace("#arith.fastmath<none>", "#arith.fastmath<fast>", 1)
    assert not pairs(source)


def test_full_rewriter_emits_one_call_and_accounts_for_both_sites(tmp_path):
    source, sidecar = tmp_path / "source.mlir", tmp_path / "sidecar.json"
    source.write_text(SOURCE)
    sidecar.write_text("{}")
    result = MXVPUElementwisePostOffload(PROFILE, connect_multiplies=True)(source, sidecar, tmp_path / "out")
    module = parse_mlir_file(result)
    module.verify()
    calls = [op for op in module.walk() if op.name == "func.call"]
    assert len(calls) == 1 and len(calls[0].arguments) == 4
    record = json.loads((result.parent / "mx_vpu_elementwise_rewrite.json").read_text())
    assert len(record["regions"]) == 2
    assert record["connected_multiplies"][0]["regions"] == ["mul_0#mul0", "mul_1#mul0"]
    assert not [op for op in module.walk() if op.name == "arith.mulf"]


def test_connection_is_opt_in(tmp_path):
    source, sidecar = tmp_path / "source.mlir", tmp_path / "sidecar.json"
    source.write_text(SOURCE)
    sidecar.write_text("{}")
    result = MXVPUElementwisePostOffload(PROFILE)(source, sidecar, tmp_path / "out")
    assert len([op for op in parse_mlir_file(result).walk() if op.name == "func.call"]) == 2
