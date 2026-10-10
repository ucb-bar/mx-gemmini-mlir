"""Ordered scalar chains stay tied to typed SSA and the physical VPU stream."""

from __future__ import annotations

from pathlib import Path

import pytest

from mx_gemmini_support.bind_payload import append_tilewise_vpu_scalar_chain
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.quant_reference import bf16_add_scalar, bf16_mul_scalar
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_tilewise_vpu_x2_266c593"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


def test_ordered_tilewise_scalar_chain_lowers_and_derives_each_bf16_step(tmp_path):
    profile = load_profile(PROFILE)
    manifest, resources = load_bundle(EVIDENCE / "bundle")
    payload = (EVIDENCE / "payload_bound.mlir").read_text()
    bound = append_tilewise_vpu_scalar_chain(
        payload, profile, manifest, (("muls", 0x4000), ("adds", 0x3fc0)))
    program = lower_bound_source(bound, profile, manifest, resources)
    expected = bf16_add_scalar(bf16_mul_scalar(resources["golden_bf16"], 0x4000),
                               0x3fc0)
    assert program.derived_expected_bf16 == expected
    assert program.derived_vpu_scalar_chain == (("muls", 0x4000), ("adds", 0x3fc0))
    assert program.receipt()["golden_derivation"] == "bf16_scalar_chain_rne"
    assert program.receipt()["derived_vpu_scalar_chain"] == [
        {"kind": "muls", "immediate_bf16": 0x4000},
        {"kind": "adds", "immediate_bf16": 0x3fc0}]
    commands = [step.command for step in program.steps if step.phase == "vpu"]
    assert len(commands) == 8
    assert [command.rs2.immediate & 0x1f for command in commands] == [4, 3] * 4
    emitted = write_standalone_sources(tmp_path / "emitted", program, resources)
    assert emitted["golden_basis"] == "derived_bf16_scalar_chain"

    for changed in (
        bound.replace('adds', 'muls', 1),
        bound.replace('16320', '16128', 1),
        bound.replace('kind = "adds"', 'kind = "muls"', 1),
        bound.replace('immediate_bf16 = 16320 : i32',
                      'immediate_bf16 = 16128 : i32', 1),
        bound.replace('"mx_gemmini.vpu_execute"(%4)',
                      '"mx_gemmini.vpu_execute"(%acc)', 1),
    ):
        assert changed != bound
        with pytest.raises(ValueError):
            lower_bound_source(changed, profile, manifest, resources)

    with pytest.raises(ValueError):
        append_tilewise_vpu_scalar_chain(
            payload, profile, manifest, (("muls", 0x4000), ("adds", 0x7f80)))
    with pytest.raises(ValueError):
        append_tilewise_vpu_scalar_chain(
            payload, profile, manifest, (("muls", 0x4000),))
