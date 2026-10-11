"""Typed dense attention must lower every source stage into the pinned stream."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.nicolas_attention_io import ATTENTION_BUFFERS
from mx_gemmini_support.nicolas_attention_program import (
    lower_dense_attention_typed_program, render_dense_attention_program)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_object import classify


ROOT = Path(__file__).resolve().parents[1]
VECTOR = ROOT / "docs/evidence/nicolas_attn_vpu_generated_middle_e9ded36_0344bcb"
PHYSICAL = ROOT / "docs/evidence/nicolas_attn_vpu_single_object_e9ded36_0344bcb"
TYPED = ROOT / "docs/evidence/nicolas_attn_vpu_typed_program_e9ded36_0344bcb"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _program() -> tuple[str, str, dict]:
    vector = (VECTOR / "attention.vector_bound.mlir").read_text()
    binding = json.loads((VECTOR / "vector_binding.json").read_text())
    return render_dense_attention_program(vector, PROFILE, binding), vector, binding


def test_typed_attention_lowers_to_spike_tested_issuer() -> None:
    program, vector, binding = _program()
    report = verify_ir(program, PROFILE)
    assert (report["memory_phases"], report["spad_matmuls"],
            report["vpu_commands"], report["spad_requants"]) == (10, 2, 7, 1)
    assert classify(program, PROFILE)[0] == "attention_program"
    commands = lower_dense_attention_typed_program(program, vector, PROFILE, binding)
    physical = json.loads((PHYSICAL / "physical_program.json").read_text())
    assert [asdict(item) if isinstance(item, Command) else {"kind": "fence"}
            for item in commands] == physical["commands"]
    assert emit_c(list(commands), transport="rocket_rocc",
                  buffers=ATTENTION_BUFFERS) == (PHYSICAL / "mx_issue.c").read_text()


def test_typed_attention_rejects_changed_matrix_or_buffer_binding() -> None:
    program, vector, binding = _program()
    with pytest.raises(ValueError, match="operation or buffer binding differs"):
        lower_dense_attention_typed_program(
            program.replace("weight_half = 0 : i32", "weight_half = 1 : i32", 1),
            vector, PROFILE, binding)
    with pytest.raises(ValueError, match="operation or buffer binding differs"):
        lower_dense_attention_typed_program(
            program.replace('"mx_gemmini.dma_matrix"(%KT_IN)',
                            '"mx_gemmini.dma_matrix"(%V_IN)', 1),
            vector, PROFILE, binding)
    with pytest.raises(ValueError, match="source or target memory binding differs"):
        lower_dense_attention_typed_program(
            program.replace('mx.source_sha256 = "5', 'mx.source_sha256 = "0', 1),
            vector, PROFILE, binding)


def test_native_attention_ops_check_geometry_and_source(tmp_path: Path) -> None:
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if not executable.is_file():
        pytest.skip("build mx-gemmini-opt or set MX_GEMMINI_OPT")
    program, _, _ = _program()
    source = tmp_path / "attention.mlir"
    source.write_text(program)
    assert subprocess.run([str(executable.resolve()), str(source), "-o", "/dev/null"],
                          capture_output=True, text=True).returncode == 0
    source.write_text(program.replace("weight_end_row = 384 : i32",
                                      "weight_end_row = 128 : i32", 1))
    invalid = subprocess.run([str(executable.resolve()), str(source), "-o", "/dev/null"],
                             capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "nonoverlapping rows" in invalid.stderr


def test_archived_typed_attention_receipt_matches_object_and_source() -> None:
    receipt = json.loads((TYPED / "receipt.json").read_text())
    dispatch = json.loads((TYPED / "typed/compile_manifest.json").read_text())
    built = json.loads((TYPED / "typed/object_manifest.json").read_text())
    physical = json.loads((TYPED / "typed/physical_program.json").read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert receipt["typed_mlir_sha256"] == sha(TYPED / "attention.program_bound.mlir")
    assert receipt["typed_object_sha256"] == sha(TYPED / "typed/mx_issue.o")
    assert receipt["typed_object_sha256"] == sha(PHYSICAL / "mx_issue.o")
    assert receipt["typed_compile_manifest_sha256"] == sha(
        TYPED / "typed/compile_manifest.json")
    assert receipt["elf_sha256"] == sha(TYPED / "attn_vpu_typed_program.elf")
    assert receipt["spike_log_sha256"] == sha(TYPED / "spike_1.log")
    assert receipt["stage_hashes"] == json.loads(
        (PHYSICAL / "receipt.json").read_text())["stage_hashes"]
    assert dispatch["lowering_family"] == "attention_program"
    assert dispatch["verified_graph_counts"]["memory_phases"] == 10
    assert dispatch["verified_graph_counts"]["spad_matmuls"] == 2
    assert built["command_count"] == 126 and built["fence_count"] == 1
    assert built["allocated_data_section_bytes"] == 0
    assert len(physical["commands"]) == 127
