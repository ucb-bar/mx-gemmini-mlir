"""The largest dense attention source shape uses typed MX commands on Spike."""

import gzip
import hashlib
import json
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
ARCHIVE = ROOT / "docs/evidence/nicolas_attn_vpu_fa_typed_program_a042643_0344bcb"
EQUAL_WORK = ROOT / "docs/evidence/attn_vpu_equal_work_20261013"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")
CORE = Path(os.getenv("NICOLAS_NPU_ROOT", "/path/to/npu-exploration")) / \
    "baremetal/mxgemmini/src/attn_vpu.c"


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


def test_fa_phased_typed_programs_match_checked_source_stream() -> None:
    vector = (ARCHIVE / "attention.vector_bound.mlir").read_text()
    binding = json.loads((ARCHIVE / "vector_binding.json").read_text())
    for phase, count in (("setup", 3), ("check", 699), ("timed", 376)):
        program = render_dense_attention_program(vector, PROFILE, binding,
                                                 phase=phase)
        assert classify(program, PROFILE)[0] == "attention_program"
        commands = lower_dense_attention_typed_program(
            program, vector, PROFILE, binding)
        assert commands == lower_dense_attention_program(
            vector, PROFILE, binding, phase=phase)
        assert len(commands) == count
    with pytest.raises(ValueError, match="selected issue phase"):
        render_dense_attention_program(vector, PROFILE, binding,
                                       phase="unknown")


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
    phased = replace_source_kernel(CORE.read_text(), variant="fa",
                                   phased=True)
    assert phased.count("mx_issue_setup(Q_SCALES, KT_SCALES, V_SCALES") == 1
    assert phased.count("mx_issue_check(Q_SCALES, KT_SCALES, V_SCALES") == 1
    assert phased.count("mx_issue_timed(Q_SCALES, KT_SCALES, V_SCALES") == 1
    assert "gemmini_mx_load_scales(" not in phased


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


def test_fa_spike_archive_matches_source_and_object_manifest() -> None:
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
    assert obj["command_count"] == 702 and obj["fence_count"] == 1
    assert len(physical["commands"]) == 703
    assert "check S vs mesh golden: 0/4096 4-element words differ" in log
    assert "check hashes vs Spike: match (0 of 4 differ)" in log
    assert "attn_vpu PASSED" in log


def test_equal_work_archive_matches_source_traffic_and_output() -> None:
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = json.loads((EQUAL_WORK / "receipt.json").read_text())
    manifests = {phase: json.loads(
        (EQUAL_WORK / "object" / phase / "object_manifest.json").read_text())
        for phase in ("setup", "check", "timed")}
    assert receipt["status"] == "equal_work_typed_attention_matches_source_on_spike"
    assert receipt["source_modeled_cycles"] == 43641
    assert receipt["compiler_modeled_cycles"] == 43523
    assert receipt["command_count"] == 1078
    assert receipt["fence_count"] == 0
    assert receipt["equal_work_counters"] == {
        "commands": 1079, "fences": 32, "load_bytes": 147456,
        "store_puts": 7168, "scale_bytes": 2304}
    assert receipt["stage_hashes"] == json.loads(
        (SOURCE / "receipt.json").read_text())["source_stage_hashes"]
    assert {phase: manifest["command_count"] for phase, manifest in
            manifests.items()} == {"setup": 3, "check": 699, "timed": 376}
    for phase, manifest in manifests.items():
        obj = EQUAL_WORK / "object" / phase / "mx_issue.o"
        assert receipt["object_sha256"][phase] == sha(obj)
        assert manifest["defined_symbol"] == f"mx_issue_{phase}"
        assert manifest["allocated_data_section_bytes"] == 0
    source_report = (EQUAL_WORK / "source_perf.txt").read_text()
    compiler_report = (EQUAL_WORK / "spike_1_perf.txt").read_text()
    for item in ("1079 commands", "32 fences", "147456 bytes in 9216 Gets",
                 "7168 Puts", "scales: 2304 bytes"):
        assert item in source_report and item in compiler_report
    for name in ("source_spike.log", "spike_1.log", "spike_2.log"):
        log = (EQUAL_WORK / name).read_text()
        assert "check hashes vs Spike: match (0 of 4 differ)" in log
        assert "attn_vpu PASSED" in log
    assert (EQUAL_WORK / "spike_1.log").read_bytes() == \
        (EQUAL_WORK / "spike_2.log").read_bytes()
