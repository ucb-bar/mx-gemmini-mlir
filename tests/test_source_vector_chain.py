"""Source-audited typed VPU→requant seam and executable command resources."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from mx_gemmini_support.source_vector_chain import capture_nicolas_vpu_requant
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.vector_lowering import lower_vector_commands
from mx_gemmini_support.vector_standalone import write_vector_requant_sources
from mx_gemmini_support.vector_standalone import write_resident_chain_sources
from mx_gemmini_support.resident_lowering import lower_resident_chain_commands


ROOT = Path(__file__).resolve().parents[1]
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
SOURCE = RTL / "software/gemmini-rocc-tests/bareMetalC/chain_vpu_spad_requant.c"
HEADER = RTL / "software/gemmini-rocc-tests/include/matmul_fp8_64x64_chain.h"


def _profile():
    return load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json",
        rtl_root=RTL)


def test_checked_source_captures_runtime_buffered_vector_mlir(tmp_path):
    if not SOURCE.is_file() or not HEADER.is_file():
        pytest.skip("requires Nicolas's pinned chain source and header")
    profile = _profile()
    mlir, resources, facts = capture_nicolas_vpu_requant(SOURCE, HEADER, profile)
    commands = lower_vector_commands(mlir, profile)
    assert [command.funct for command in commands] == [33, 34]
    assert commands[-1].rs1.buffer == "c1_scales"
    assert commands[-1].rs1.address_shift == 30
    assert len(resources["c1_bf16"]) == 8192
    assert len(resources["c1_codes_ref"]) == 4096
    assert len(resources["c1_scales_ref"]) == 128
    receipt = write_vector_requant_sources(tmp_path / "artifact", mlir, profile,
                                            resources, facts)
    assert receipt["command_count"] > 50
    issuer = (tmp_path / "artifact/mx_issue.c").read_text()
    assert ".insn r 0x7b, 3, 33" in issuer
    assert ".insn r 0x7b, 3, 34" in issuer
    assert "<< 30" in issuer
    assert "#include \"include/matmul_fp8_64x64_chain.h\"" not in issuer
    evidence = ROOT / "docs/evidence"
    assert mlir == (evidence / "nicolas_chain_vpu_requant_64x64_source_bound.mlir").read_text()
    saved = json.loads((evidence / "compiled_nicolas_chain_vpu_requant_64x64_20261009.json").read_text())
    assert saved["status"] == "source_vector_seam_matched_on_pinned_spike"
    assert saved["compiler_revision"].startswith("c642bbf")
    assert saved["typed_mlir_sha256"] == hashlib.sha256(mlir.encode()).hexdigest()
    assert saved["files_sha256"] == receipt["files_sha256"]
    assert saved["compared_fp8_codes"] == 4096
    assert saved["compared_e8m0_scales"] == 128
    reproduced = json.loads((evidence / "compiled_nicolas_chain_vpu_requant_64x64_repro_20261009.json").read_text())
    for field in ("typed_mlir_sha256", "files_sha256", "object_sha256", "elf_sha256",
                  "extension_sha256", "spike_log_sha256", "compiler_source_closure_sha256"):
        assert saved[field] == reproduced[field]
    opt = ROOT / "build/tools/mx-gemmini-opt"
    if opt.is_file():
        path = tmp_path / "source_bound.mlir"
        path.write_text(mlir)
        subprocess.run([str(opt), str(path), "-o", "/dev/null"], check=True)


def test_nicolas_vector_capture_rejects_changed_requant_command(tmp_path):
    if not SOURCE.is_file() or not HEADER.is_file():
        pytest.skip("requires Nicolas's pinned chain source and header")
    source = tmp_path / SOURCE.name
    source.write_text(SOURCE.read_text().replace(
        "gemmini_spad_requant(SP_C1, SP_BF16, M, N, 1, (uint64_t)c1_scales, 1);",
        "gemmini_spad_requant(SP_C1, SP_BF16, M, N, 0, (uint64_t)c1_scales, 1);"))
    with pytest.raises(ValueError, match="VPU/requant command changed"):
        capture_nicolas_vpu_requant(source, HEADER, _profile())


def test_source_bound_resident_second_matmul_lowers_and_checks_source(tmp_path):
    if not SOURCE.is_file() or not HEADER.is_file():
        pytest.skip("requires Nicolas's pinned chain source and header")
    profile = _profile()
    mlir, resources, facts = capture_nicolas_vpu_requant(
        SOURCE, HEADER, profile, include_resident_matmul=True)
    commands = lower_resident_chain_commands(mlir, profile)
    functs = [command.funct for command in commands if hasattr(command, "funct")]
    assert functs[:2] == [33, 34]
    assert functs[-4:] == [26, 9, 24, 8]
    assert len(resources["b2_weight"]) == 4096
    assert len(resources["b2_scales"]) == 128
    assert len(resources["c2_codes_ref"]) == 4096
    assert len(resources["c2_scales_ref"]) == 128
    assert "excludes first matmul" in facts["source_scope"]
    receipt = write_resident_chain_sources(tmp_path / "resident", mlir, profile,
                                            resources, facts)
    assert receipt["command_count"] == 93
    issuer = (tmp_path / "resident/mx_issue.c").read_text()
    assert ".insn r 0x7b, 3, 8" in issuer
    assert ".insn r 0x7b, 3, 34" in issuer
    assert "matmul_fp8_64x64_chain.h" not in issuer
    opt = ROOT / "build/tools/mx-gemmini-opt"
    if opt.is_file():
        path = tmp_path / "resident.mlir"
        path.write_text(mlir)
        subprocess.run([str(opt), str(path), "-o", "/dev/null"], check=True)


def test_resident_chain_refuses_broken_handoff_and_source_drift(tmp_path):
    if not SOURCE.is_file() or not HEADER.is_file():
        pytest.skip("requires Nicolas's pinned chain source and header")
    profile = _profile()
    mlir, _, _ = capture_nicolas_vpu_requant(
        SOURCE, HEADER, profile, include_resident_matmul=True)
    with pytest.raises(ValueError, match="handoff differs"):
        lower_resident_chain_commands(
            mlir.replace("activation_row = 128 : i32", "activation_row = 144 : i32"),
            profile)
    source = tmp_path / SOURCE.name
    source.write_text(SOURCE.read_text().replace(
        "gemmini_mxquant_config_mvout_resident((uint64_t)c2_scales",
        "gemmini_mxquant_config_mvout((uint64_t)c2_scales"))
    with pytest.raises(ValueError, match="resident second contraction changed"):
        capture_nicolas_vpu_requant(source, HEADER, profile,
                                    include_resident_matmul=True)
