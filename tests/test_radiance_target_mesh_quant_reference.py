"""Re-derive target-mesh quantized oracles from pinned Radiance source."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import pytest

from mx_gemmini_support.bind_payload import bind_payload
from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.mesh_reference import derive_mesh_reference
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.quant_reference import (quantize_bf16_radiance_header_fp6,
                                                quantize_bf16_radiance_header_fp8)
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import (load_bundle,
                                               replace_source_quantized_with_mesh_reference,
                                               validate_target_mesh_reference, write_bundle)
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22/frontend"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxDim8AllAsymGemminiRocketConfig.json"
CASES = ("mxgemm.fp8.singletile.tm64tn64tk64.requant",
         "mxgemm.fp4.singletile.tm64tn64tk64.requant",
         "mxgemm.fp6.singletile.tm128tn128tk128.requant")


@pytest.mark.skipif(not os.getenv("RADIANCE_KERNELS_ROOT"),
                    reason="set RADIANCE_KERNELS_ROOT to the pinned Radiance checkout")
@pytest.mark.parametrize("stem", CASES)
def test_target_mesh_quant_bundle_lowers_checked_header_epilogue(stem: str, tmp_path):
    source = Path(os.environ["RADIANCE_KERNELS_ROOT"])
    driver = source / "kernels/gemm_mxgemmini" / f"{stem}.cpp"
    kernel = read_source_gemm(driver)
    profile = load_profile(PROFILE)
    bundle = tmp_path / "bundle"
    write_bundle(bundle, kernel, site_id="functional:matmul",
                 profile_sha256=profile_sha256(profile))
    source_manifest, resources = load_bundle(bundle)
    source_codes = resources["source_fp6_packed" if kernel.datatype == "FP6"
                             else "golden_fp8"]
    source_scales = resources["golden_output_scales"]
    target, policy = derive_mesh_reference(
        source, resources, kernel.shape, kernel.datatype, 8, product_floor=True)
    manifest = replace_source_quantized_with_mesh_reference(bundle, target, policy)
    loaded, target_resources = load_bundle(bundle)
    assert loaded == manifest
    assert manifest["target_mesh_reference"] == policy
    assert manifest["target_quant_reference"]["source_code_sha256"] != (
        manifest["target_quant_reference"]["target_code_sha256"])
    assert resources["golden_bf16"] != target_resources["golden_bf16"]
    codes, scales = (
        quantize_bf16_radiance_header_fp6(
            target, kernel.shape[0], kernel.shape[1], target_resources["output_lut"])
        if kernel.datatype == "FP6" else
        quantize_bf16_radiance_header_fp8(target, kernel.shape[0], kernel.shape[1]))
    code_name = "source_fp6_packed" if kernel.datatype == "FP6" else "golden_fp8"
    assert source_codes != codes
    assert target_resources[code_name] == codes
    assert target_resources["golden_output_scales"] == scales
    assert source_scales != scales or source_codes != codes
    handoff = (FRONTEND / stem / "mx_gemm.handoff.mlir").read_text()
    selected = bind_handoff(handoff, profile)
    bound = bind_payload(selected, profile, manifest, source_header_quantized=True)
    program = lower_bound_source(bound, profile, manifest, target_resources)
    assert program.golden_origin == "target_mesh_reference"
    assert program.output_format == ("radiance_header_fp6" if kernel.datatype == "FP6"
                                     else "radiance_header_fp8")
    assert not program.source_golden_preserving
    emitted = write_standalone_sources(tmp_path / "emitted", program, target_resources)
    assert emitted["output_policy"] == ("radiance_header_fp6_v1" if
                                        kernel.datatype == "FP6" else
                                        "radiance_header_fp8_v1")
    damaged = deepcopy(manifest)
    damaged["target_quant_reference"]["target_code_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="quant reference lacks pinned"):
        validate_target_mesh_reference(damaged)
    assert source_manifest["origin"] == "radiance_source_header_specialization"
