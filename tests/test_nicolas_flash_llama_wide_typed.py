"""Two-V-buffer streamed schedule preserves source placement and full output."""

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
ARCHIVE = ROOT / "docs/evidence/nicolas_flash_llama_wide_stream_typed_e9ded36_0344bcb"
NPU = Path(os.getenv("NICOLAS_NPU_ROOT",
                  "/scratch/agustin/tmp/npu-exploration-mx-source-20261009"))
CORE = NPU / "baremetal/mxgemmini/src/attn_flash.c"
WRAPPER = NPU / "baremetal/mxgemmini/src/attn_flash_llama_wide.c"
PROFILE = load_profile(ROOT / ("profiles/gemmini-mx-cleanup-266c593/"
                               "MxE4M3Fp4VpuGemminiRocketConfig.json"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs() -> tuple[str, str]:
    return ((SOURCE / "attention.model2mlir.mlir").read_text(),
            gzip.decompress((SOURCE / "attn_vpu_llama.h.gz").read_bytes()).decode())


def test_wide_stream_two_v_buffers() -> None:
    if not CORE.is_file() or not WRAPPER.is_file():
        pytest.skip("pinned Nicolas wide source is unavailable")
    frontend, header = _inputs()
    graph, binding = render_flash_mid(
        frontend, CORE.read_text(), WRAPPER.read_text(), header, PROFILE,
        shape=STREAM_SHAPE, block_k=256, v3=False)
    assert graph == (ARCHIVE / "flash_llama_wide_stream.mlir").read_text()
    assert binding == json.loads((ARCHIVE / "binding.json").read_text())
    assert binding["policy"]["kv_stream_buffers"] == [2, 2]
    assert binding["policy"]["spad_q_v_kt_p_st_o"] == [0, 256, 4096, 6144, 2304, 14848]
    family, report = classify(graph, PROFILE)
    assert family == "flash_program" and report == verify_ir(graph, PROFILE)
    assert (report["memory_phases"], report["flash_setups"],
            report["spad_matmuls"], report["vpu_commands"],
            report["spad_requants"]) == (35, 1, 16, 83, 8)
    commands = lower_flash_mid(graph, frontend, CORE.read_text(),
                               WRAPPER.read_text(), header, PROFILE,
                               shape=STREAM_SHAPE, block_k=256,
                               v3=False)
    assert sum(isinstance(item, Command) for item in commands) == 484
    assert sum(isinstance(item, Fence) for item in commands) == 50
    with pytest.raises(ValueError, match="V-buffer policy"):
        render_flash_mid(frontend, CORE.read_text(), WRAPPER.read_text(),
                         header, PROFILE, shape=STREAM_SHAPE, block_k=256,
                         v3=0)


def test_archived_wide_stream_matches_every_source_word() -> None:
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    manifest = json.loads((ARCHIVE / "object/object_manifest.json").read_text())
    assert receipt["status"] == "typed_two_v_buffer_flash_full_bf16_source_parity"
    assert receipt["source_baseline_sha256"] == _sha(SOURCE / "receipt.json")
    assert receipt["expected_output_hash"] == "dc33d078bc537b19"
    assert receipt["compared_bf16_words"] == 4096
    assert receipt["source_runs"] == receipt["compiler_runs"] == 2
    assert receipt["object_sha256"] == manifest["object_sha256"] == _sha(
        ARCHIVE / "object/mx_issue.o")
    assert manifest["v3"] is False
    assert (manifest["command_count"], manifest["fence_count"]) == (484, 50)
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
