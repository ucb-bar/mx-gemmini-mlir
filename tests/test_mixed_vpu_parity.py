"""A real MX contraction and nontrivial BF16 VPU op share one command stream."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mx_gemmini_support.bind_payload import bind_payload
from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import load_bundle, write_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))
FP4_SOURCE = Path(os.environ.get("RADIANCE_FP4_GENERATED_ROOT", "/nonexistent"))
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


@pytest.mark.parametrize("precision,source_root,driver,evidence", [
    ("fp8", SOURCE, "mxgemm.fp8.m128n128k512.tm128tn128tk128.fullout.cpp",
     "compiled_mx_fp8_vpu_x2_20261009.json"),
    ("fp4", FP4_SOURCE, "mxgemm.fp4.m64n64k128.tm64tn64tk64.fullout.cpp",
     "compiled_mx_fp4_vpu_x2_20261009.json"),
])
def test_compiler_regenerates_gemm_vpu_x2_spike_parity(
        tmp_path, precision, source_root, driver, evidence):
    selected = source_root / "kernels/gemm_mxgemmini" / driver
    if not selected.is_file() or not read_source_gemm(selected).data_header_present:
        pytest.skip("requires Radiance source data header")
    profile = load_profile(PROFILE)
    manifest = write_bundle(tmp_path / "bundle", read_source_gemm(selected),
                            site_id="functional:matmul",
                            profile_sha256=profile_sha256(profile))
    _, resources = load_bundle(tmp_path / "bundle")
    mlir = (ROOT / f"examples/mx_{precision}_vpu_x2_profile_bound.mlir").read_text()
    bound = bind_payload(mlir, profile, manifest)
    program = lower_bound_source(bound, profile, manifest, resources)
    phases = [step.phase for step in program.steps]
    assert phases.index("vpu") > max(i for i, phase in enumerate(phases) if phase == "compute")
    assert phases.index("vpu") < phases.index("readout")
    assert len([step for step in program.steps if isinstance(step.command, Command)
                and step.command.funct == 33]) == 1
    assert not program.source_golden_preserving
    assert program.derived_expected_bf16 != resources["golden_bf16"]
    receipt = write_standalone_sources(tmp_path / "artifact", program, resources)
    saved = json.loads((ROOT / "docs/evidence" / evidence).read_text())
    assert saved["status"] == "derived_vpu_golden_matched_on_pinned_spike"
    assert saved["spike_exit_code"] == 0
    assert saved["compared_bf16_outputs"] == program.shape[0] * program.shape[1]
    assert saved["golden_basis"] == "derived_bf16_x2"
    assert saved["files_sha256"] == receipt["files_sha256"]
    changed = bound.replace("immediate_bf16 = 16384 : i32", "immediate_bf16 = 16385 : i32")
    unsupported = lower_bound_source(changed, profile, manifest, resources)
    with pytest.raises(ValueError, match="golden does not cover"):
        write_standalone_sources(tmp_path / "unsupported", unsupported, resources)
