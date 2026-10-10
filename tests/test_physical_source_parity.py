"""Rebuild the exact physical programs whose full outputs passed pinned Spike."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mx_gemmini_support.bind_payload import bind_payload
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import load_bundle, write_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))
FP4_SOURCE = Path(os.environ.get("RADIANCE_FP4_GENERATED_ROOT", "/nonexistent"))
PROFILE_DIR = ROOT / "profiles/gemmini-mx-cleanup-266c593"


@pytest.mark.parametrize("precision,source_root,driver,profile_name,capture,evidence", [
    ("FP8", SOURCE, "mxgemm.fp8.m128n128k512.tm128tn128tk128.fullout.cpp",
     "MxE4M3Fp4VpuGemminiRocketConfig", "gemm",
     "compiled_mx_fp8_128x128x512_tk128_20261009.json"),
    ("FP8", SOURCE, "mxgemm.fp8.m128n128k512.tm128tn128tk256.fullout.cpp",
     "MxE4M3Fp4VpuGemminiRocketConfig", "gemm",
     "compiled_mx_fp8_128x128x512_tk256_20261009.json"),
    ("FP8", SOURCE, "mxgemm.fp8.m64n64k128.tm64tn64tk64.fullout.cpp",
     "MxE4M3Fp4VpuGemminiRocketConfig",
     "docs/evidence/model2mlir_radiance_mx_fp8_64x64x128_bound.mlir",
     "compiled_mx_fp8_64x64x128_20261009.json"),
    ("FP8", SOURCE, "mxgemm.fp8.m128n128k256.tm128tn128tk128.fullout.cpp",
     "MxE4M3Fp4VpuGemminiRocketConfig",
     "docs/evidence/model2mlir_radiance_mx_fp8_128x128x256_bound.mlir",
     "compiled_mx_fp8_128x128x256_20261009.json"),
    ("FP8", SOURCE, "mxgemm.fp8.m256n256k256.tm128tn128tk256.fullout.cpp",
     "MxE4M3Fp4VpuGemminiRocketConfig",
     "docs/evidence/model2mlir_radiance_mx_fp8_256x256x256_bound.mlir",
     "compiled_mx_fp8_256x256x256_20261009.json"),
    ("FP4", FP4_SOURCE, "mxgemm.fp4.m64n64k128.tm64tn64tk64.fullout.cpp",
     "MxE4M3Fp4VpuGemminiRocketConfig", "fp4",
     "compiled_mx_fp4_64x64x128_20261009.json"),
    ("FP4", FP4_SOURCE, "mxgemm.fp4.singletile.tm128tn128tk128.fullout.cpp",
     "MxE4M3Fp4VpuGemminiRocketConfig",
     "docs/evidence/model2mlir_radiance_mx_fp4_128x128x128_bound.mlir",
     "compiled_mx_fp4_128x128x128_20261009.json"),
    ("FP4", FP4_SOURCE, "mxgemm.fp4.m128n128k256.tm128tn128tk128.fullout.cpp",
     "MxE4M3Fp4VpuGemminiRocketConfig",
     "docs/evidence/model2mlir_radiance_mx_fp4_128x128x256_bound.mlir",
     "compiled_mx_fp4_128x128x256_20261009.json"),
    ("FP6", SOURCE, "mxgemm.fp6.m128n128k2048.tm128tn128tk128.fullout.cpp",
     "MxE3M2OnlyGemminiRocketConfig", "fp6",
     "compiled_mx_fp6_128x128x2048_20261009.json"),
])
def test_compiler_regenerates_numerically_qualified_physical_stream(
        tmp_path, precision, source_root, driver, profile_name, capture, evidence):
    selected = source_root / "kernels/gemm_mxgemmini" / driver
    if not selected.is_file():
        pytest.skip("requires Radiance source driver")
    kernel = read_source_gemm(selected)
    if not kernel.data_header_present:
        pytest.skip("requires Radiance source data header")
    profile = load_profile(PROFILE_DIR / f"{profile_name}.json")
    manifest = write_bundle(tmp_path / "bundle", kernel,
                            site_id="functional:matmul",
                            profile_sha256=profile_sha256(profile))
    _, resources = load_bundle(tmp_path / "bundle")
    capture_file = (ROOT / capture if capture.endswith(".mlir") else
                    ROOT / f"docs/evidence/model2mlir_radiance_mx_{capture}_bound_20261009.mlir")
    mlir = capture_file.read_text()
    bound = bind_payload(mlir, profile, manifest)
    program = lower_bound_source(bound, profile, manifest, resources)
    receipt = write_standalone_sources(tmp_path / "artifact", program, resources)
    saved = json.loads((ROOT / "docs/evidence" / evidence).read_text())
    assert saved["status"] == "source_golden_matched_on_pinned_spike"
    assert saved["spike_exit_code"] == 0
    assert saved["compared_bf16_outputs"] == program.shape[0] * program.shape[1]
    assert saved["profile_sha256"] == receipt["profile_sha256"]
    assert saved["payload_manifest_sha256"] == receipt["payload_manifest_sha256"]
    assert saved["command_count"] == receipt["command_count"]
    assert saved["fence_count"] == receipt["fence_count"]
    assert saved["files_sha256"] == receipt["files_sha256"]
    # Historical receipts pin the earlier digest-only MLIR syntax. The new
    # source-resource SSA binding must preserve its emitted physical program.
    checked_ir = verify_ir(bound, profile)
    assert checked_ir["source_resources"] == (7 if precision == "FP6" else 4)
    assert checked_ir["lut_uploads"] == (3 if precision == "FP6" else 0)
    assert '#include "mxgemm.data.' not in (tmp_path / "artifact/mx_issue.c").read_text()
    assert saved["fp6_spike_scale_selector_workaround"] == (precision == "FP6")
    if kernel.shape[0] != kernel.tile[0] or kernel.shape[1] != kernel.tile[1]:
        assert len(program.plan["output_tiles"]) == 4
