"""Variable key blocks preserve source offsets and full-output Spike parity."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command, Fence
from mx_gemmini_support.nicolas_flash_mid import (
    STREAM_SHAPE, lower_flash_mid, render_flash_mid)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_object import classify


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/nicolas_flash_llama_stream_source_baseline_e9ded36_0344bcb"
ARCHIVE = ROOT / "docs/evidence/nicolas_flash_llama_vb_stream_typed_e9ded36_0344bcb"
NPU = Path(os.getenv("NICOLAS_NPU_ROOT",
                  "/path/to/npu-exploration"))
CORE = NPU / "baremetal/mxgemmini/src/attn_flash.c"
WRAPPER = NPU / "baremetal/mxgemmini/src/attn_flash_llama_vb.c"
PROFILE = load_profile(ROOT / ("profiles/gemmini-mx-cleanup-266c593/"
                               "MxE4M3Fp4VpuGemminiRocketConfig.json"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs() -> tuple[str, str]:
    return ((SOURCE / "attention.model2mlir.mlir").read_text(),
            gzip.decompress((SOURCE / "attn_vpu_llama.h.gz").read_bytes()).decode())


def test_variable_block_graph_and_source_offsets() -> None:
    if not CORE.is_file() or not WRAPPER.is_file():
        pytest.skip("pinned Nicolas variable-block source is unavailable")
    frontend, header = _inputs()
    graph, binding = render_flash_mid(
        frontend, CORE.read_text(), WRAPPER.read_text(), header, PROFILE,
        shape=STREAM_SHAPE, block_k=256, block_first=64, block_last=192)
    assert graph == (ARCHIVE / "flash_llama_vb_stream.mlir").read_text()
    assert binding == json.loads((ARCHIVE / "binding.json").read_text())
    assert binding["policy"]["block_lengths"] == [64] + [256] * 7 + [192]
    assert binding["policy"]["key_offsets"] == [0, 64, 320, 576, 832,
                                                 1088, 1344, 1600, 1856]
    family, report = classify(graph, PROFILE)
    assert family == "flash_program" and report == verify_ir(graph, PROFILE)
    assert (report["memory_phases"], report["flash_setups"],
            report["spad_matmuls"], report["vpu_commands"],
            report["spad_requants"]) == (39, 1, 18, 94, 9)
    commands = lower_flash_mid(graph, frontend, CORE.read_text(),
                               WRAPPER.read_text(), header, PROFILE,
                               shape=STREAM_SHAPE, block_k=256,
                               block_first=64, block_last=192)
    assert sum(isinstance(item, Command) for item in commands) == 508
    assert sum(isinstance(item, Fence) for item in commands) == 56
    with pytest.raises(ValueError, match="selected key-block schedule"):
        render_flash_mid(frontend, CORE.read_text(), WRAPPER.read_text(),
                         header, PROFILE, shape=STREAM_SHAPE, block_k=256,
                         block_first=128, block_last=128)


def test_archived_variable_blocks_match_every_source_word() -> None:
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    manifest = json.loads((ARCHIVE / "object/object_manifest.json").read_text())
    assert receipt["status"] == "typed_variable_block_flash_full_bf16_source_parity"
    assert receipt["source_baseline_sha256"] == _sha(SOURCE / "receipt.json")
    assert receipt["expected_output_hash"] == "747ca99b19c32066"
    assert receipt["compared_bf16_words"] == 4096
    assert receipt["source_runs"] == receipt["compiler_runs"] == 2
    assert receipt["object_sha256"] == manifest["object_sha256"] == _sha(
        ARCHIVE / "object/mx_issue.o")
    assert (manifest["block_first"], manifest["block_last"]) == (64, 192)
    assert (manifest["command_count"], manifest["fence_count"]) == (508, 56)
    assert manifest["allocated_data_section_bytes"] == 0
    outputs = []
    for kind in ("source", "generated"):
        for run in (1, 2):
            log = (ARCHIVE / f"{kind}_spike_{run}.log").read_text()
            words = next(line.removeprefix("O BF16 words ")
                         for line in log.splitlines()
                         if line.startswith("O BF16 words "))
            assert len(words) == 4096 * 4
            outputs.append(words)
    assert len(set(outputs)) == 1
