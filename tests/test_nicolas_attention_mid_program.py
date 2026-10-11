"""The mid-shape source graph must lower to one profile-checked MX object."""

from pathlib import Path
import gzip
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
