"""Typed streamed flash lowers to the archived source-matched RV64 object."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mx_gemmini_support.command_ir import Command, Fence
from mx_gemmini_support.nicolas_flash_mid import (
    STREAM_SHAPE, flash_buffer_abi, lower_flash_mid, render_flash_mid)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_object import classify


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/nicolas_flash_llama_stream_source_baseline_e9ded36_0344bcb"
ARCHIVE = ROOT / "docs/evidence/nicolas_flash_llama_stream_typed_e9ded36_0344bcb"
NPU = Path(os.getenv("NICOLAS_NPU_ROOT",
                  "/path/to/npu-exploration"))
RTL = Path(os.getenv("NICOLAS_MX_RTL_ROOT",
                  "/path/to/gemmini-mx-rtl"))
RISCV = Path(os.getenv("RISCV_ROOT",
                    "/path/to/riscv-tools"))
CORE = NPU / "baremetal/mxgemmini/src/attn_flash.c"
WRAPPER = NPU / "baremetal/mxgemmini/src/attn_flash_llama.c"
PROFILE_PATH = ROOT / ("profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")
PROFILE = load_profile(PROFILE_PATH)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs() -> tuple[str, str]:
    return ((SOURCE / "attention.model2mlir.mlir").read_text(),
            gzip.decompress((SOURCE / "attn_vpu_llama.h.gz").read_bytes()).decode())


def test_streamed_flash_graph_and_ring_schedule(tmp_path: Path) -> None:
    if not CORE.is_file() or not WRAPPER.is_file():
        pytest.skip("pinned Nicolas flash source is unavailable")
    frontend, header = _inputs()
    graph, binding = render_flash_mid(frontend, CORE.read_text(),
                                      WRAPPER.read_text(), header, PROFILE,
                                      shape=STREAM_SHAPE, block_k=256)
    assert graph == (ARCHIVE / "flash_llama_stream.mlir").read_text()
    assert binding == json.loads((ARCHIVE / "binding.json").read_text())
    assert binding["policy"]["kv_stream_buffers"] == [2, 3]
    assert binding["policy"]["spad_q_v_kt_p_st_o"] == [0, 256, 4096, 6144, 3328, 14848]
    assert binding["policy"]["score_rows"] == [8192, 12288] * 4
    assert binding["policy"]["probability_rows"] == [6144, 7168] * 4
    names, sizes = flash_buffer_abi(STREAM_SHAPE, 256)
    assert names[3:6] == ("Q_IN", "KT_IN", "V_IN")
    assert sizes[3:6] == (4096, 131072, 131072)
    assert sizes[7:9] == (512, 512)
    family, report = classify(graph, PROFILE)
    assert family == "flash_program" and report == verify_ir(graph, PROFILE)
    assert (report["memory_phases"], report["flash_setups"],
            report["spad_matmuls"], report["vpu_commands"],
            report["spad_requants"]) == (35, 1, 16, 83, 8)
    commands = lower_flash_mid(graph, frontend, CORE.read_text(),
                               WRAPPER.read_text(), header, PROFILE,
                               shape=STREAM_SHAPE, block_k=256)
    assert sum(isinstance(item, Command) for item in commands) == 484
    assert sum(isinstance(item, Fence) for item in commands) == 50
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if executable.is_file():
        source = tmp_path / "flash_llama_stream.mlir"
        source.write_text(graph)
        result = subprocess.run([str(executable.resolve()), str(source),
                                 "-o", "/dev/null"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_streamed_flash_fails_closed_on_layout_or_graph_change() -> None:
    if not CORE.is_file() or not WRAPPER.is_file():
        pytest.skip("pinned Nicolas flash source is unavailable")
    frontend, header = _inputs()
    disabled = deepcopy(PROFILE)
    disabled["resources"]["scratchpad_banks"] = 2
    with pytest.raises(ValueError, match="four-bank"):
        render_flash_mid(frontend, CORE.read_text(), WRAPPER.read_text(),
                         header, disabled, shape=STREAM_SHAPE, block_k=256)
    graph = (ARCHIVE / "flash_llama_stream.mlir").read_text()
    with pytest.raises(ValueError, match="typed graph"):
        lower_flash_mid(graph.replace("weight_half = 0 : i32",
                                      "weight_half = 1 : i32", 1),
                        frontend, CORE.read_text(), WRAPPER.read_text(),
                        header, PROFILE, shape=STREAM_SHAPE, block_k=256)


def test_archived_streamed_object_matches_every_source_word() -> None:
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    manifest = json.loads((ARCHIVE / "object/object_manifest.json").read_text())
    dispatch = json.loads((ARCHIVE / "object/compile_manifest.json").read_text())
    assert receipt["status"] == "typed_streamed_flash_full_bf16_source_parity"
    assert receipt["compared_bf16_words"] == 4096
    assert receipt["source_runs"] == receipt["compiler_runs"] == 2
    assert receipt["expected_output_hash"] == "dc33d078bc537b19"
    assert receipt["object_sha256"] == manifest["object_sha256"] == _sha(
        ARCHIVE / "object/mx_issue.o")
    assert receipt["source_elf_sha256"] == _sha(ARCHIVE / "flash_llama_stream_source.elf")
    assert receipt["generated_elf_sha256"] == _sha(
        ARCHIVE / "flash_llama_stream_generated.elf")
    assert receipt["source_baseline_sha256"] == _sha(SOURCE / "receipt.json")
    assert manifest["command_count"] == 484 and manifest["fence_count"] == 50
    assert manifest["allocated_data_section_bytes"] == 0
    assert manifest["embedded_operand_bytes"] == manifest["embedded_golden_bytes"] == 0
    assert dispatch["lowering_family"] == "flash_program"
    assert dispatch["verified_graph_counts"]["spad_matmuls"] == 16
    outputs = []
    for kind in ("source", "generated"):
        for run in (1, 2):
            log = (ARCHIVE / f"{kind}_spike_{run}.log").read_text()
            line = next(line for line in log.splitlines()
                        if line.startswith("O BF16 words "))
            words = line.removeprefix("O BF16 words ")
            assert len(words) == 4096 * 4
            outputs.append(words)
    assert len(set(outputs)) == 1


def test_public_compiler_reproduces_streamed_object(tmp_path: Path) -> None:
    if not all(path.is_file() for path in (
            CORE, WRAPPER, RISCV / "bin/riscv64-unknown-elf-gcc")) or not RTL.is_dir():
        pytest.skip("pinned Nicolas source, RTL, or toolchain is unavailable")
    out = tmp_path / "object"
    command = [sys.executable, "-m", "tools.compile_object",
               "--mlir", str(ARCHIVE / "flash_llama_stream.mlir"),
               "--precursor-mlir", str(SOURCE / "attention.model2mlir.mlir"),
               "--binding-json", str(ARCHIVE / "binding.json"),
               "--source-core", str(CORE), "--source-wrapper", str(WRAPPER),
               "--source-header", str(SOURCE / "attn_vpu_llama.h.gz"),
               "--profile", str(PROFILE_PATH), "--rtl-root", str(RTL),
               "--riscv-root", str(RISCV), "--out-dir", str(out)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for name in ("mx_issue.c", "mx_issue.h", "mx_issue.o", "physical_program.json"):
        assert _sha(out / name) == _sha(ARCHIVE / "object" / name)
    changed = tmp_path / "changed_wrapper.c"
    changed.write_text(WRAPPER.read_text() + "\n/* changed */\n")
    rejected = command.copy()
    rejected[rejected.index("--source-wrapper") + 1] = str(changed)
    rejected[-1] = str(tmp_path / "rejected")
    result = subprocess.run(rejected, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "pinned Nicolas source core and wrapper" in result.stderr
    assert not (tmp_path / "rejected").exists()
