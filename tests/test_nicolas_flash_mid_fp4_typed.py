"""FP4 packed flash lowering matches the pinned source and target profile."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from mx_gemmini_support.command_ir import Command, Fence
from mx_gemmini_support.nicolas_flash_mid import (SHAPE, flash_buffer_abi,
                                                  lower_flash_mid,
                                                  render_flash_mid)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_object import classify


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_flash_mid_fp4_typed_a042643_0344bcb"
FRONTEND = ROOT / ("docs/evidence/nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb/"
                   "attention.model2mlir.mlir")
WRAPPER = ROOT / "tools/fixtures/nicolas_flash_mid_fp4.c"
EXPSUM_ARCHIVE = ROOT / "docs/evidence/nicolas_flash_mid_fp4_expsum_typed_a042643_0344bcb"
EXPSUM_WRAPPER = ROOT / "tools/fixtures/nicolas_flash_mid_fp4_expsum.c"
CORE = Path(os.getenv("NICOLAS_NPU_ROOT",
                  "/path/to/npu-exploration")) / \
    "baremetal/mxgemmini/src/attn_flash.c"
PROFILE = load_profile(ROOT / ("profiles/gemmini-mx-cleanup-266c593/"
                               "MxE4M3Fp4VpuGemminiRocketConfig.json"))


def test_fp4_flash_graph_and_command_inventory(tmp_path: Path) -> None:
    if not CORE.is_file():
        pytest.skip("pinned Nicolas flash source is unavailable")
    frontend = FRONTEND.read_text()
    header = gzip.decompress((ARCHIVE / "attn_vpu_mid_fp4.h.gz").read_bytes()).decode()
    graph, binding = render_flash_mid(frontend, CORE.read_text(),
                                      WRAPPER.read_text(), header, PROFILE,
                                      input_format="fp4_e2m1")
    assert graph == (ARCHIVE / "flash_mid_fp4.mlir").read_text()
    assert binding == json.loads((ARCHIVE / "binding.json").read_text())
    family, report = classify(graph, PROFILE)
    assert family == "flash_program"
    assert (report["memory_phases"], report["flash_setups"],
            report["spad_matmuls"], report["vpu_commands"],
            report["spad_requants"]) == (19, 1, 8, 39, 4)
    assert report == verify_ir(graph, PROFILE)
    commands = lower_flash_mid(graph, frontend, CORE.read_text(),
                               WRAPPER.read_text(), header, PROFILE,
                               input_format="fp4_e2m1")
    assert sum(isinstance(item, Command) for item in commands) == 154
    assert sum(isinstance(item, Fence) for item in commands) == 21
    assert commands[0].funct == 0  # FP4 format before Q-scale DMA.
    assert commands[0].rs1.immediate & (2 << 12 | 2 << 10) == 2 << 12 | 2 << 10
    assert sum(item.funct == 34 and item.rs2.immediate & (1 << 32) != 0
               for item in commands if isinstance(item, Command)) == 4
    buffers, sizes = flash_buffer_abi(SHAPE, 32, "fp4_e2m1")
    assert buffers[3:6] == ("Q4_IN", "KT4_IN", "V4_IN")
    assert sizes[3:6] == (2048, 4096, 4096)
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if executable.is_file():
        source = tmp_path / "flash_mid_fp4.mlir"
        source.write_text(graph)
        run = subprocess.run([str(executable.resolve()), str(source),
                              "-o", "/dev/null"], capture_output=True, text=True)
        assert run.returncode == 0, run.stderr


def test_fp4_flash_rejects_fp8_policy_and_profile() -> None:
    if not CORE.is_file():
        pytest.skip("pinned Nicolas flash source is unavailable")
    header = gzip.decompress((ARCHIVE / "attn_vpu_mid_fp4.h.gz").read_bytes()).decode()
    frontend = FRONTEND.read_text()
    with pytest.raises(ValueError, match="unsupported policy"):
        render_flash_mid(frontend, CORE.read_text(), WRAPPER.read_text(),
                         header, PROFILE)
    profile = load_profile(ROOT / ("profiles/gemmini-mx-cleanup-266c593/"
                                   "MxE4M3VpuGemminiRocketConfig.json"))
    with pytest.raises(ValueError, match="compute tuple is absent"):
        render_flash_mid(frontend, CORE.read_text(), WRAPPER.read_text(),
                         header, profile, input_format="fp4_e2m1")


def test_fp4_expsum_flash_graph_and_vpu_gate(tmp_path: Path) -> None:
    if not CORE.is_file():
        pytest.skip("pinned Nicolas flash source is unavailable")
    frontend = FRONTEND.read_text()
    header = gzip.decompress((EXPSUM_ARCHIVE / "attn_vpu_mid_fp4.h.gz").read_bytes()).decode()
    graph, binding = render_flash_mid(frontend, CORE.read_text(),
                                      EXPSUM_WRAPPER.read_text(), header, PROFILE,
                                      input_format="fp4_e2m1", softmax_mode="expsum")
    assert graph == (EXPSUM_ARCHIVE / "flash_mid_fp4_expsum.mlir").read_text()
    assert binding == json.loads((EXPSUM_ARCHIVE / "binding.json").read_text())
    assert binding["policy"]["input_format"] == "fp4_e2m1"
    assert binding["policy"]["softmax_mode"] == "expsum"
    family, report = classify(graph, PROFILE)
    assert family == "flash_program"
    assert (report["memory_phases"], report["flash_setups"],
            report["spad_matmuls"], report["vpu_commands"],
            report["spad_requants"]) == (19, 1, 8, 31, 4)
    assert graph.count('kind = "expsum"') == 4
    commands = lower_flash_mid(graph, frontend, CORE.read_text(),
                               EXPSUM_WRAPPER.read_text(), header, PROFILE,
                               input_format="fp4_e2m1", softmax_mode="expsum")
    assert sum(isinstance(item, Command) for item in commands) == 146
    assert sum(isinstance(item, Fence) for item in commands) == 21
    disabled = deepcopy(PROFILE)
    disabled["resources"]["vpu_config"]["exp_sum"] = False
    with pytest.raises(ValueError, match="expsum is not built"):
        render_flash_mid(frontend, CORE.read_text(), EXPSUM_WRAPPER.read_text(),
                         header, disabled, input_format="fp4_e2m1",
                         softmax_mode="expsum")
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if executable.is_file():
        source = tmp_path / "flash_mid_fp4_expsum.mlir"
        source.write_text(graph)
        result = subprocess.run([str(executable.resolve()), str(source),
                                 "-o", "/dev/null"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("variant", "command_count", "mode"), [
    ("mid_fp4", 154, "split"), ("mid_fp4_expsum", 146, "expsum")])
def test_archived_fp4_flash_matches_all_source_output_words(
        variant: str, command_count: int, mode: str) -> None:
    archive = (EXPSUM_ARCHIVE if mode == "expsum" else ARCHIVE)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = json.loads((archive / "receipt.json").read_text())
    obj = json.loads((archive / "object/object_manifest.json").read_text())
    dispatch = json.loads((archive / "object/compile_manifest.json").read_text())
    assert receipt["status"] == f"typed_flash_{variant}_full_bf16_source_parity"
    assert receipt["compared_bf16_words"] == 4096
    assert receipt["source_runs"] == receipt["compiler_runs"] == 2
    assert receipt["expected_output_hash"] == "f0caacc02eab4345"
    assert receipt["object_sha256"] == obj["object_sha256"] == sha(
        archive / "object/mx_issue.o")
    assert receipt["source_elf_sha256"] == sha(archive / f"flash_{variant}_source.elf")
    assert receipt["generated_elf_sha256"] == sha(archive / f"flash_{variant}_generated.elf")
    assert receipt["source_header_gzip_sha256"] == sha(archive / "attn_vpu_mid_fp4.h.gz")
    assert obj["input_format"] == "fp4_e2m1"
    assert obj.get("softmax_mode", "split") == mode
    assert obj["command_count"] == command_count and obj["fence_count"] == 21
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert dispatch["lowering_family"] == "flash_program"
    assert dispatch["verified_graph_counts"]["spad_requants"] == 4
    outputs = []
    for kind in ("source", "generated"):
        for run in (1, 2):
            log = (archive / f"{kind}_spike_{run}.log").read_text()
            line = next(line for line in log.splitlines()
                        if line.startswith("O BF16 words "))
            words = line.removeprefix("O BF16 words ")
            assert len(words) == 4096 * 4
            outputs.append(words)
    assert len(set(outputs)) == 1
