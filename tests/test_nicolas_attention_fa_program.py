"""The largest dense attention source shape uses typed MX commands on Spike."""

import gzip
import os
from pathlib import Path
import subprocess

import pytest

from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.nicolas_attention_io import lower_dense_attention_program
from mx_gemmini_support.nicolas_attention_program import (
    lower_dense_attention_typed_program, render_dense_attention_program)
from mx_gemmini_support.nicolas_attention_vector import render_dense_attention_vector
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_object import classify
from tools.qualify_nicolas_dense_attention import replace_source_kernel


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/nicolas_attn_vpu_fa_source_baseline_a042643_0344bcb"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")
CORE = Path("/scratch/agustin/tmp/npu-exploration-mx-source-20261009/"
            "baremetal/mxgemmini/src/attn_vpu.c")


def _bound() -> tuple[str, str, dict]:
    if not CORE.is_file():
        pytest.skip("pinned Nicolas source checkout is unavailable")
    source = CORE.read_text()
    header = gzip.decompress((SOURCE / "attn_vpu_fa.h.gz").read_bytes()).decode()
    frontend = (SOURCE / "attention.model2mlir.mlir").read_text()
    contract = (ROOT / "mx_gemmini_support/contracts/"
                "software-spec-2029218-candidate.yaml").read_bytes()
    vector, binding = render_dense_attention_vector(
        frontend, source, header, PROFILE, contract)
    return render_dense_attention_program(vector, PROFILE, binding), vector, binding


def test_fa_typed_graph_and_source_shape_lowering() -> None:
    program, vector, binding = _bound()
    report = verify_ir(program, PROFILE)
    assert (report["memory_phases"], report["spad_matmuls"],
            report["vpu_commands"], report["spad_requants"]) == (10, 2, 7, 1)
    assert classify(program, PROFILE)[0] == "attention_program"
    assert binding["plan"]["shape_sq_sk_d"] == [64, 256, 128]
    assert binding["plan"]["score_row"] == 2560
    assert binding["plan"]["probability_row"] == 4608
    assert binding["plan"]["output_row"] == 7680
    assert binding["plan"]["steps"][0][-1] == 0x3db5
    assert "m = 64 : i32, n = 256 : i32, k = 128 : i32" in program
    assert "m = 64 : i32, n = 128 : i32, k = 256 : i32" in program
    typed = lower_dense_attention_typed_program(program, vector, PROFILE, binding)
    assert typed == lower_dense_attention_program(vector, PROFILE, binding)
    assert len(typed) == 703
    assert sum(isinstance(item, Command) for item in typed) == 702


def test_fa_frontend_scale_is_bound_to_source() -> None:
    if not CORE.is_file():
        pytest.skip("pinned Nicolas source checkout is unavailable")
    header = gzip.decompress((SOURCE / "attn_vpu_fa.h.gz").read_bytes()).decode()
    frontend = (SOURCE / "attention.model2mlir.mlir").read_text()
    contract = (ROOT / "mx_gemmini_support/contracts/"
                "software-spec-2029218-candidate.yaml").read_bytes()
    with pytest.raises(ValueError, match="source scale differs"):
        render_dense_attention_vector(
            frontend.replace("0.0883883461 : f32", "0.125 : f32", 1),
            CORE.read_text(), header, PROFILE, contract)
    driver = replace_source_kernel(CORE.read_text(), variant="fa")
    assert driver.startswith("#define ATTN_HOST_SOFTMAX 0\n#define ATTN_ACC_ROWS 16\n")
    assert driver.count("mx_issue(Q_SCALES, KT_SCALES, V_SCALES") == 1
    assert "gemmini_loop_ws_spad(" not in driver


def test_native_verifier_accepts_fa_shape(tmp_path: Path) -> None:
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if not executable.is_file():
        pytest.skip("build mx-gemmini-opt or set MX_GEMMINI_OPT")
    program, _, _ = _bound()
    source = tmp_path / "attention_fa.mlir"
    source.write_text(program)
    run = subprocess.run([str(executable.resolve()), str(source), "-o", "/dev/null"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
