"""Scalar tilewise MULS lowering keeps its MLIR immediate and reference tied."""

from __future__ import annotations

import gzip
from pathlib import Path
import struct

import pytest

from mx_gemmini_support.bind_payload import append_tilewise_vpu_muls
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.quant_reference import bf16_mul_scalar
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_tilewise_vpu_x2_266c593"
FP4_EVIDENCE = ROOT / "docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593"


def _archived(name: str) -> bytes:
    compressed = EVIDENCE / f"{name}.gz"
    return gzip.decompress(compressed.read_bytes()) if compressed.exists() else (
        EVIDENCE / name).read_bytes()


def test_bf16_scalar_reference_rounding_and_rejections():
    source = struct.pack("<7H", 0x3f80, 0x3f81, 0xbf80, 0x0001,
                         0x0003, 0x7f7f, 0x8000)
    assert struct.unpack("<7H", bf16_mul_scalar(source, 0x3fc0)) == (
        0x3fc0, 0x3fc2, 0xbfc0, 0x0002, 0x0004, 0x7f80, 0x8000)
    for scalar in (True, -1, 0x7f80, 0x7fc0, 0x10000):
        with pytest.raises(ValueError):
            bf16_mul_scalar(source, scalar)
    with pytest.raises(ValueError):
        bf16_mul_scalar(b"\x00", 0x3fc0)
    with pytest.raises(ValueError):
        bf16_mul_scalar(struct.pack("<H", 0x7f80), 0x3fc0)


def test_tilewise_scalar_binding_and_physical_immediate():
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    manifest, resources = load_bundle(EVIDENCE / "bundle")
    payload = _archived("payload_bound.mlir").decode()
    bound = append_tilewise_vpu_muls(payload, profile, manifest, 0x3fc0)
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.plan["vector_tile_policy"] == "bf16_muls_scalar_each_output_tile_v1"
    assert program.derived_vpu_scalar_bf16 == 0x3fc0
    assert program.derived_expected_bf16 == bf16_mul_scalar(resources["golden_bf16"], 0x3fc0)
    assert program.receipt()["golden_derivation"] == "bf16_scalar_muls_rne"
    vpu = [step for step in program.steps if step.phase == "vpu"]
    assert len(vpu) == 4
    assert len({(step.command.funct, step.command.rs1.immediate,
                 step.command.rs2.immediate) for step in vpu}) == 1

    for changed in (
        bound.replace("mx.vector_scalar_bf16 = 16320 : i32",
                      "mx.vector_scalar_bf16 = 16128 : i32"),
        bound.replace("immediate_bf16 = 16320 : i32",
                      "immediate_bf16 = 16128 : i32"),
        bound.replace("mx.vector_scalar_bf16 = 16320 : i32",
                      "mx.vector_scalar_bf16 = 32640 : i32"),
        bound.replace('mx.vector_tile_policy = "bf16_muls_scalar_each_output_tile_v1"',
                      'mx.vector_tile_policy = "unknown"'),
    ):
        assert changed != bound
        with pytest.raises(ValueError):
            lower_bound_source(changed, profile, manifest, resources)
    with pytest.raises(ValueError):
        append_tilewise_vpu_muls(payload, profile, manifest, 0x7f80)


def test_generated_fp4_tilewise_scalar_uses_same_physical_policy():
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    manifest, resources = load_bundle(FP4_EVIDENCE / "bundle")
    payload = (FP4_EVIDENCE / "payload_bound.mlir").read_text()
    bound = append_tilewise_vpu_muls(payload, profile, manifest, 0x3fc0)
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.derived_vpu_scalar_bf16 == 0x3fc0
    assert program.derived_expected_bf16 == bf16_mul_scalar(
        resources["golden_bf16"], 0x3fc0)
    assert sum(step.phase == "vpu" for step in program.steps) == 4
