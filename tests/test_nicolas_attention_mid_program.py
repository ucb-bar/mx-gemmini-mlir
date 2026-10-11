"""The mid-shape source graph must lower to one profile-checked MX object."""

from pathlib import Path
import gzip
import hashlib
import json
import os
import re
import subprocess

import pytest

from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.nicolas_attention_io import lower_dense_attention_program
from mx_gemmini_support.nicolas_attention_program import (
    lower_dense_attention_typed_program, render_dense_attention_program)
from mx_gemmini_support.nicolas_attention_shape import DenseAttentionShape
from mx_gemmini_support.nicolas_attention_vector import render_dense_attention_vector
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_object import classify
from tools.qualify_nicolas_dense_attention import replace_source_kernel


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb"
BASE_DRIVER = ROOT / "docs/evidence/nicolas_attn_vpu_typed_program_e9ded36_0344bcb"
ARCHIVE = ROOT / "docs/evidence/nicolas_attn_vpu_mid_typed_program_a042643_0344bcb"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")
CORE = Path("/scratch/agustin/tmp/npu-exploration-mx-source-20261009/"
            "baremetal/mxgemmini/src/attn_vpu.c")


def _bound() -> tuple[str, str, dict]:
    if not CORE.is_file():
        pytest.skip("pinned Nicolas source checkout is unavailable")
    source = CORE.read_text()
    header = gzip.decompress((SOURCE / "attn_vpu_mid.h.gz").read_bytes()).decode()
    frontend = (SOURCE / "attention.model2mlir.mlir").read_text()
    contract = (ROOT / "mx_gemmini_support/contracts/"
                "software-spec-2029218-candidate.yaml").read_bytes()
    vector, binding = render_dense_attention_vector(
        frontend, source, header, PROFILE, contract)
    return render_dense_attention_program(vector, PROFILE, binding), vector, binding


def test_mid_shape_typed_graph_and_physical_schedule() -> None:
    program, vector, binding = _bound()
    report = verify_ir(program, PROFILE)
    assert (report["memory_phases"], report["spad_matmuls"],
            report["vpu_commands"], report["spad_requants"]) == (10, 2, 7, 1)
    assert classify(program, PROFILE)[0] == "attention_program"
    assert binding["plan"]["shape_sq_sk_d"] == [64, 128, 64]
    assert binding["plan"]["score_row"] == 768
    assert binding["plan"]["probability_row"] == 1792
    assert binding["plan"]["output_row"] == 2816
    assert "m = 64 : i32, n = 128 : i32, k = 64 : i32" in program
    assert "m = 64 : i32, n = 64 : i32, k = 128 : i32" in program
    assert "memref<16384xi8>" in program
    typed = lower_dense_attention_typed_program(program, vector, PROFILE, binding)
    assert typed == lower_dense_attention_program(vector, PROFILE, binding)
    assert len(typed) == 303
    assert sum(isinstance(item, Command) for item in typed) == 302


def test_mid_shape_rejects_frontend_or_scratchpad_mismatch() -> None:
    program, vector, binding = _bound()
    with pytest.raises(ValueError, match="operation or buffer binding differs"):
        lower_dense_attention_typed_program(
            program.replace("spad_row = 256 : i32", "spad_row = 128 : i32", 1),
            vector, PROFILE, binding)
    header = gzip.decompress((SOURCE / "attn_vpu_mid.h.gz").read_bytes()).decode()
    altered = header.replace("#define ATTN_SK 128", "#define ATTN_SK 256", 1)
    with pytest.raises(ValueError, match="frontend differs"):
        render_dense_attention_vector(
            (SOURCE / "attention.model2mlir.mlir").read_text(), CORE.read_text(),
            altered, PROFILE,
            (ROOT / "mx_gemmini_support/contracts/"
             "software-spec-2029218-candidate.yaml").read_bytes())
    with pytest.raises(ValueError, match="residency"):
        DenseAttentionShape(128, 2048, 128).validate(PROFILE)


def test_source_checker_transformation_keeps_single_compiler_call() -> None:
    if not CORE.is_file():
        pytest.skip("pinned Nicolas source checkout is unavailable")
    base = replace_source_kernel(CORE.read_text(), variant="base")
    archived = (BASE_DRIVER / "attn_vpu_typed_program.c").read_text()
    assert re.sub(r"\n{2,}", "\n\n", base) == re.sub(r"\n{2,}", "\n\n", archived)
    mid = replace_source_kernel(CORE.read_text(), variant="mid")
    assert mid.startswith("#define ATTN_HOST_SOFTMAX 0\n#define ATTN_ACC_ROWS 16\n")
    assert mid.count("mx_issue(Q_SCALES, KT_SCALES, V_SCALES") == 1
    assert "gemmini_loop_ws_spad(" not in mid


def test_native_verifier_accepts_mid_shape(tmp_path: Path) -> None:
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if not executable.is_file():
        pytest.skip("build mx-gemmini-opt or set MX_GEMMINI_OPT")
    program, _, _ = _bound()
    source = tmp_path / "attention_mid.mlir"
    source.write_text(program)
    run = subprocess.run([str(executable.resolve()), str(source), "-o", "/dev/null"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr


def test_mid_spike_archive_matches_source_and_object_manifest() -> None:
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    source = json.loads((SOURCE / "receipt.json").read_text())
    obj = json.loads((ARCHIVE / "object/object_manifest.json").read_text())
    dispatch = json.loads((ARCHIVE / "object/compile_manifest.json").read_text())
    physical = json.loads((ARCHIVE / "object/physical_program.json").read_text())
    log = (ARCHIVE / "spike_1.log").read_text()
    assert receipt["source_baseline_sha256"] == sha(SOURCE / "receipt.json")
    assert receipt["stage_hashes"] == source["source_stage_hashes"]
    assert receipt["spike_reproduced_runs"] == 2
    assert receipt["typed_mlir_sha256"] == sha(ARCHIVE / "attention.program_bound.mlir")
    assert receipt["vector_mlir_sha256"] == sha(ARCHIVE / "attention.vector_bound.mlir")
    assert receipt["binding_sha256"] == sha(ARCHIVE / "vector_binding.json")
    assert receipt["object_sha256"] == obj["object_sha256"] == sha(
        ARCHIVE / "object/mx_issue.o")
    assert receipt["object_manifest_sha256"] == sha(ARCHIVE / "object/object_manifest.json")
    assert receipt["compile_manifest_sha256"] == sha(ARCHIVE / "object/compile_manifest.json")
    assert receipt["driver_sha256"] == sha(ARCHIVE / "attn_vpu_typed_program.c")
    assert receipt["elf_sha256"] == sha(ARCHIVE / "attn_vpu_typed_program.elf")
    assert receipt["spike_log_sha256"] == sha(ARCHIVE / "spike_1.log")
    assert dispatch["lowering_family"] == "attention_program"
    assert dispatch["verified_graph_counts"]["memory_phases"] == 10
    assert dispatch["verified_graph_counts"]["spad_matmuls"] == 2
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert obj["command_count"] == 302 and obj["fence_count"] == 1
    assert len(physical["commands"]) == 303
    assert "check S vs mesh golden: 0/2048 4-element words differ" in log
    assert "check hashes vs Spike: match (0 of 4 differ)" in log
    assert "attn_vpu PASSED" in log
