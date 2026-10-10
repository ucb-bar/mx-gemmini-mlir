"""One typed matrix→VPU→resident SPAD_REQUANT compiler program."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from mx_gemmini_support.bind_payload import append_vpu_spad_requant_x2, bind_payload
from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import (load_bundle, write_bundle,
                                               vpu_requant_shape_is_legal)
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("RADIANCE_KERNELS_ROOT", "/nonexistent"))
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
DRIVER = SOURCE / ("kernels/gemm_mxgemmini/"
                   "mxgemm.fp8.m64n64k128.tm64tn64tk64.fullout.cpp")


def _case(tmp_path):
    if not DRIVER.is_file() or not DRIVER.with_name(
            "mxgemm.data.fp8.m64n64k128.h").is_file() or not RTL.is_dir():
        pytest.skip("requires generated Radiance FP8 header and Nicolas RTL")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json",
        rtl_root=RTL)
    manifest = write_bundle(
        tmp_path / "bundle", read_source_gemm(DRIVER),
        site_id="functional:matmul", profile_sha256=profile_sha256(profile),
        vpu_spad_requant_x2=True)
    checked, resources = load_bundle(tmp_path / "bundle")
    assert checked == manifest
    frontend = (ROOT / "docs/evidence/model2mlir_radiance_mx_fp8_64x64x128_bound.mlir").read_text()
    bound = append_vpu_spad_requant_x2(bind_payload(frontend, profile, manifest),
                                        profile, manifest)
    return profile, manifest, resources, bound


def test_compiler_orders_matrix_vpu_requant_and_reuses_dead_operand_rows(tmp_path):
    profile, manifest, resources, bound = _case(tmp_path)
    assert '"mx_gemmini.contract"' in bound
    assert '"mx_gemmini.vpu_execute"' in bound
    assert '"mx_gemmini.spad_requant"' in bound
    assert '"mx_gemmini.readout_bf16"' in bound
    assert '"mx_gemmini.vpu_execute"(%4)' in bound
    assert '"mx_gemmini.spad_requant"(%5)' in bound
    assert 'func.return %6, %7' in bound
    opt = ROOT / "build/tools/mx-gemmini-opt"
    if opt.is_file():
        path = tmp_path / "bound.mlir"
        path.write_text(bound)
        subprocess.run([str(opt), str(path), "-o", "/dev/null"], check=True)
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.tiled_quant_readout
    assert program.plan["c_spad_dest"] == 256
    assert program.plan["c_rows"] == 512
    phases = [step.phase for step in program.steps]
    assert phases.index("compute") < phases.index("vpu") < phases.index("spad_requant") < phases.index("readout")
    requant = next(step.command for step in program.steps if step.phase == "spad_requant")
    assert isinstance(requant, Command)
    assert requant.rs1.buffer == "scratch_output_scales"
    assert requant.rs1.address_shift == 30
    readout = next(step.command for step in program.steps
                   if step.phase == "readout" and isinstance(step.command, Command) and
                   step.command.funct == 3)
    assert readout.rs2.immediate & 0xffffffff == 1024
    receipt = write_standalone_sources(tmp_path / "artifact", program, resources)
    assert receipt["golden_basis"] == "nicolas_vpu_x2_spad_requant_from_source_bf16"
    assert receipt["source_quant_code_differences"] > 0
    assert receipt["source_quant_scale_differences"] == 128
    driver = (tmp_path / "artifact/mx_driver.c").read_text()
    assert "uint32_t tiled" in driver
    assert "FP8 code mismatches" in driver
    evidence = ROOT / "docs/evidence"
    assert bound == (evidence / "matrix_vpu_requant_fp8_64x64x128_ssa_bound.mlir").read_text()
    ssa_saved = json.loads((evidence / "compiled_matrix_vpu_requant_fp8_64x64x128_ssa.json").read_text())
    ssa_repro = json.loads((evidence / "compiled_matrix_vpu_requant_fp8_64x64x128_ssa_repro.json").read_text())
    assert ssa_saved["compiler_revision"].startswith("358eb3b")
    assert ssa_saved["status"] == "nicolas_oracle_matched_on_pinned_spike"
    assert ssa_saved["bound_mlir_sha256"] == hashlib.sha256(bound.encode()).hexdigest()
    for field in ("bound_mlir_sha256", "files_sha256", "object_sha256", "elf_sha256",
                  "extension_sha256", "spike_log_sha256", "compiler_source_closure_sha256"):
        assert ssa_saved[field] == ssa_repro[field]
    assert manifest == json.loads((evidence / "matrix_vpu_requant_fp8_64x64x128_payload_manifest.json").read_text())
    checked_ir = verify_ir(bound, profile)
    assert (checked_ir["source_resources"], checked_ir["lut_uploads"]) == (4, 0)
    saved = json.loads((evidence / "compiled_matrix_vpu_requant_fp8_64x64x128_20261009.json").read_text())
    assert ssa_saved["elf_sha256"] == saved["elf_sha256"]
    assert ssa_saved["spike_log_sha256"] == saved["spike_log_sha256"]
    assert saved["files_sha256"] == receipt["files_sha256"]
    assert saved["compiler_revision"].startswith("9ebbb07")
    assert saved["status"] == "nicolas_oracle_matched_on_pinned_spike"
    assert saved["compared_fp8_codes"] == 4096
    assert saved["compared_e8m0_scales"] == 128
    assert saved["source_quant_code_differences"] == 4094
    assert saved["source_quant_scale_differences"] == 128
    qualification = json.loads((evidence / "matrix_vpu_requant_fp8_64x64x128_qualification.json").read_text())
    assert qualification["model2mlir_revision"] == "7485a829c0195af0ec42820837d609e62e466564"
    assert qualification["artifact_manifest_sha256"] == hashlib.sha256(
        (evidence / "compiled_matrix_vpu_requant_fp8_64x64x128_20261009.json").read_bytes()).hexdigest()
    reproduced = json.loads((evidence / "compiled_matrix_vpu_requant_fp8_64x64x128_repro_20261009.json").read_text())
    for field in qualification["reproducible_fields"]:
        assert saved[field] == reproduced[field]


def test_matrix_vpu_requant_rejects_changed_residency_or_pointer(tmp_path):
    profile, manifest, resources, bound = _case(tmp_path)
    with pytest.raises(ValueError, match="scratchpad lifetime or operation differs"):
        lower_bound_source(bound.replace("resident = true", "resident = false"),
                           profile, manifest, resources)
    with pytest.raises(ValueError, match="scratchpad lifetime or operation differs"):
        lower_bound_source(bound.replace('scale_buffer = "scratch_output_scales"',
                                         'scale_buffer = "other_scales"'),
                           profile, manifest, resources)
    with pytest.raises(ValueError, match="VPU/SPAD SSA handoff differs"):
        lower_bound_source(bound.replace('"mx_gemmini.spad_requant"(%5)',
                                         '"mx_gemmini.spad_requant"(%4)'),
                           profile, manifest, resources)


def test_matrix_vpu_requant_bundle_rejects_oracle_downgrade(tmp_path):
    _case(tmp_path)
    path = tmp_path / "bundle/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["output_oracle"] = "nicolas_mxquant_po2_rne_v1"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported format"):
        load_bundle(tmp_path / "bundle")


@pytest.mark.parametrize(("driver", "capture", "shape"), [
    ("mxgemm.fp8.singletile.tm64tn64tk64.fullout.cpp",
     "fp8_vpu_requant_shapes_e9ded36/64x64x64/profile_bound.mlir", (64, 64, 64)),
    ("mxgemm.fp8.m128n128k128.tm128tn128tk128.fullout.cpp",
     "fp8_vpu_requant_shapes_e9ded36/128x128x128/profile_bound.mlir", (128, 128, 128)),
])
def test_single_tile_vpu_requant_shape_lowering(tmp_path, driver, capture, shape):
    source = SOURCE / "kernels/gemm_mxgemmini" / driver
    if not source.is_file() or not source.with_name(
            "mxgemm.data.fp8.m%dn%dk%d.h" % shape).is_file() or not RTL.is_dir():
        pytest.skip("requires the matching Radiance FP8 source header and Nicolas RTL")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json",
        rtl_root=RTL)
    manifest = write_bundle(
        tmp_path / "bundle", read_source_gemm(source),
        site_id="functional:matmul", profile_sha256=profile_sha256(profile),
        vpu_spad_requant_x2=True)
    checked, resources = load_bundle(tmp_path / "bundle")
    assert checked == manifest
    bound = append_vpu_spad_requant_x2(
        bind_payload((ROOT / "docs/evidence" / capture).read_text(), profile, manifest),
        profile, manifest)
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.shape == shape
    assert program.tiled_quant_readout
    assert f"tensor<{shape[0]}x{shape[1]}xbf16>" in bound
    assert f"tensor<{shape[0]}x{shape[1] // 32}xi8>" in bound
    assert len([step for step in program.steps if step.phase == "spad_requant"]) == 1
    evidence = ROOT / "docs/evidence/fp8_vpu_requant_shapes_e9ded36"
    case = evidence / f"{shape[0]}x{shape[1]}x{shape[2]}"
    assert bound == (case / "payload_bound.mlir").read_text()
    capture_receipt = json.loads((case / "capture_receipt.json").read_text())
    spike_receipt = json.loads((case / "spike_receipt.json").read_text())
    spike_repro = json.loads((case / "spike_repro_receipt.json").read_text())
    assert capture_receipt["model2mlir_revision"].startswith("e9ded36")
    assert capture_receipt["source_revision"].startswith("ee22e0b")
    assert spike_receipt["status"] == "nicolas_oracle_matched_on_pinned_spike"
    assert spike_receipt["compared_fp8_codes"] == shape[0] * shape[1]
    assert spike_receipt["compared_e8m0_scales"] == shape[0] * shape[1] // 32
    assert spike_receipt["bound_mlir_sha256"] == hashlib.sha256(bound.encode()).hexdigest()
    for field in ("bound_mlir_sha256", "files_sha256", "object_sha256", "elf_sha256",
                  "extension_sha256", "spike_log_sha256", "compiler_source_closure_sha256"):
        assert spike_receipt[field] == spike_repro[field]
    emitted = write_standalone_sources(tmp_path / "artifact", program, resources)
    assert emitted["files_sha256"] == spike_receipt["files_sha256"]
    opt = ROOT / "build/tools/mx-gemmini-opt"
    if opt.is_file():
        path = tmp_path / "bound.mlir"
        path.write_text(bound)
        subprocess.run([str(opt), str(path), "-o", "/dev/null"], check=True)


def test_vpu_requant_rejects_multiple_output_tiles_and_excessive_blocks():
    assert not vpu_requant_shape_is_legal((128, 128, 128), (64, 64, 64))
    assert not vpu_requant_shape_is_legal((256, 288, 256), (256, 288, 256))


def test_latest_vpu_requant_roster_binds_frontend_and_spike_evidence():
    root = ROOT / "docs/evidence/fp8_vpu_requant_shapes_e9ded36"
    index = json.loads((root / "index.json").read_text())
    assert index["model2mlir_revision"].startswith("e9ded36")
    assert index["radiance_source_revision"].startswith("ee22e0b")
    assert {tuple(row["shape_mnk"]) for row in index["cases"]} == {
        (64, 64, 64), (128, 128, 128)}
    for row in index["cases"]:
        m, n, k = row["shape_mnk"]
        case = root / f"{m}x{n}x{k}"
        assert row["capture_receipt_sha256"] == hashlib.sha256(
            (case / "capture_receipt.json").read_bytes()).hexdigest()
        assert row["spike_receipt_sha256"] == hashlib.sha256(
            (case / "spike_receipt.json").read_bytes()).hexdigest()
        assert row["spike_log_sha256"] == hashlib.sha256(
            (case / "spike.log").read_bytes()).hexdigest()
        assert row["compared_fp8_codes"] == m * n
        assert row["compared_e8m0_scales"] == m * n // 32
