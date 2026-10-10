"""Audit the Muon RV32 MMIO issuer without claiming SoC execution."""

from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command, Fence, WaitIdle, emit_c
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.transport_lowering import issuer_commands
from tools.emit_mx_mmio_object import _verify_gateway_header
from tools.emit_mx_object import _referenced_buffers


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/fp8_vpu_muon_mmio_object_266c593"
PROFILE = ROOT / ("profiles/gemmini-mx-cleanup-266c593/"
                  "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_vpu_mmio_object_matches_physical_program_and_waits_for_gateway():
    index = json.loads((EVIDENCE / "index.json").read_text())
    assert index["schema"] == "mx_gemmini.fp8_vpu_muon_mmio_object_archive.v1"
    assert index["status"] == "structural_object_only"
    assert "no Radiance VPU SoC image binding" in index["scope"]
    for name, digest in index["files_sha256"].items():
        assert _sha((EVIDENCE / name).read_bytes()) == digest
    receipt = json.loads((EVIDENCE / "object_manifest.json").read_text())
    assert receipt == json.loads((EVIDENCE / "object_manifest_repro.json").read_text())
    assert receipt["schema"] == "mx_gemmini.muon_mmio_object.v1"
    assert receipt["status"] == "muon_rv32_mmio_issuer_built_unqualified"
    assert receipt["qualification"] == "structural_object_only"
    assert receipt["transport"] == "muon_mmio"
    assert receipt["physical_mode"] == "rtl_alternating"
    assert receipt["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert receipt["object_emitter_sha256"] == _sha((
        ROOT / "tools/emit_mx_mmio_object.py").read_bytes())
    assert receipt["allocated_data_section_bytes"] == 0
    assert receipt["embedded_operand_bytes"] == receipt["embedded_golden_bytes"] == 0
    assert receipt["defined_symbol"] == "mx_issue"
    assert receipt["undefined_symbols"] == []
    obj = (EVIDENCE / "mx_issue.o").read_bytes()
    assert obj[:4] == b"\x7fELF" and obj[4] == 1
    assert int.from_bytes(obj[18:20], "little") == 243
    assert receipt["object_sha256"] == _sha(obj)

    mlir = (ROOT / index["source_mlir"]).read_text()
    manifest, resources = load_bundle(ROOT / index["source_bundle"])
    profile = load_profile(PROFILE)
    program = lower_bound_source(mlir, profile, manifest, resources,
                                 mode="rtl_alternating")
    archived = gzip.decompress((EVIDENCE / "physical_program.json.gz").read_bytes())
    assert program.receipt() == json.loads(archived)
    assert receipt["physical_program_sha256"] == _sha(archived)
    assert receipt["bound_mlir_sha256"] == _sha(mlir.encode())
    assert len(program.plan["output_tiles"]) == 4
    assert program.plan["bf16_output_layout"] == "row_major_bf16"
    assert sum(step.phase == "vpu" for step in program.steps) == 4
    assert len(program.steps) == receipt["command_count"]
    buffers = _referenced_buffers(program, manifest)
    assert buffers == receipt["buffer_abi"]
    assert buffers[2]["layout"] == "row_major_bf16"
    commands = issuer_commands(program, "muon_mmio")
    physical_fences = sum(isinstance(step.command, Fence) for step in program.steps)
    assert physical_fences == receipt["completion_fences"] == (
        receipt["gateway_busy_waits"])
    assert sum(isinstance(item, WaitIdle) for item in commands) == physical_fences
    for index, item in enumerate(commands):
        if isinstance(item, Fence):
            assert isinstance(commands[index + 1], WaitIdle)
    assert len(commands) == len(program.steps) + physical_fences
    source = gzip.decompress((EVIDENCE / "mx_issue.c.gz").read_bytes()).decode()
    assert source == emit_c(list(commands), transport="muon_mmio",
                            buffers=tuple(entry["name"] for entry in buffers))
    assert receipt["issuer_c_sha256"] == _sha(source.encode())
    assert source.count("mx_control_base + 0x20") == physical_fences
    assert source.count("__sync_synchronize();") == physical_fences
    assert source.count("mx_control_base + 0x10") == sum(
        isinstance(item, Command) for item in commands)
    assert "uintptr_t mx_control_base" in (
        EVIDENCE / "mx_issue.h").read_text()

    with pytest.raises(ValueError, match="completion fence"):
        issuer_commands(replace(program, steps=program.steps[:-1]), "muon_mmio")
    with pytest.raises(ValueError, match="transport"):
        issuer_commands(program, "unselected")


def test_gateway_header_must_match_radiance_register_protocol(tmp_path):
    header = tmp_path / "gemmini_mmio.h"
    header.write_text("\n".join([
        "#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)",
        "#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)",
        "#define GEMMINI_INST_ADDR (GEMMINI_CTRL + 0x0)",
        "#define GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)",
        "(0x7B) (3 << 12) ((funct) << 25)",
    ]))
    _verify_gateway_header(header)
    header.write_text(header.read_text().replace("GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x20)",
                                                "GEMMINI_BUSY_ADDR (GEMMINI_CTRL + 0x24)"))
    with pytest.raises(ValueError, match="GEMMINI_BUSY_ADDR"):
        _verify_gateway_header(header)
