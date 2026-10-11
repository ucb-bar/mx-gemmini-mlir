"""The causal streamed graph preserves the source mask and full Spike output."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command, Fence
from mx_gemmini_support.nicolas_flash_mid import (
    STREAM_SHAPE, flash_buffer_abi, lower_flash_mid, render_flash_mid)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_object import classify


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_flash_llama_causal_stream_typed_e9ded36_0344bcb"
NPU = Path(os.getenv("NICOLAS_NPU_ROOT",
                  "/scratch/agustin/tmp/npu-exploration-mx-source-20261009"))
CORE = NPU / "baremetal/mxgemmini/src/attn_flash.c"
WRAPPER = NPU / "baremetal/mxgemmini/src/attn_flash_llama_causal.c"
PROFILE = load_profile(ROOT / ("profiles/gemmini-mx-cleanup-266c593/"
                               "MxE4M3Fp4VpuGemminiRocketConfig.json"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs() -> tuple[str, str]:
    return ((ARCHIVE / "attention.model2mlir.mlir").read_text(),
            gzip.decompress((ARCHIVE / "attn_vpu_llama_causal.h.gz").read_bytes()).decode())


def test_causal_stream_graph_and_mask_schedule() -> None:
    if not CORE.is_file() or not WRAPPER.is_file():
        pytest.skip("pinned Nicolas causal source is unavailable")
    frontend, header = _inputs()
    graph, binding = render_flash_mid(
        frontend, CORE.read_text(), WRAPPER.read_text(), header, PROFILE,
        shape=STREAM_SHAPE, block_k=256, causal=True)
    assert graph == (ARCHIVE / "flash_llama_causal_stream.mlir").read_text()
    assert binding == json.loads((ARCHIVE / "binding.json").read_text())
    assert binding["policy"]["mask_spad_row"] == 10752
    assert binding["policy"]["mask_key_start"] == 1984
    names, sizes = flash_buffer_abi(STREAM_SHAPE, 256, causal=True)
    assert names[-1] == "MASK_BF16" and sizes[-1] == 8192
    assert names[:10] == flash_buffer_abi(STREAM_SHAPE, 256)[0]
    assert 'site_id = "operand:causal_mask"' in graph
    assert graph.count('site_id = "softmax7:mask') == 64
    family, report = classify(graph, PROFILE)
    assert family == "flash_program" and report == verify_ir(graph, PROFILE)
    assert (report["memory_phases"], report["flash_setups"],
            report["spad_matmuls"], report["vpu_commands"],
            report["spad_requants"]) == (36, 1, 16, 147, 8)
    commands = lower_flash_mid(graph, frontend, CORE.read_text(),
                               WRAPPER.read_text(), header, PROFILE,
                               shape=STREAM_SHAPE, block_k=256, causal=True)
    assert sum(isinstance(item, Command) for item in commands) == 581
    assert sum(isinstance(item, Fence) for item in commands) == 50


def test_causal_stream_rejects_missing_mask_or_different_bank() -> None:
    if not CORE.is_file() or not WRAPPER.is_file():
        pytest.skip("pinned Nicolas causal source is unavailable")
    frontend, header = _inputs()
    with pytest.raises(ValueError, match="score-mask"):
        render_flash_mid(frontend.replace('prov.region_id = "add_0"',
                                          'prov.region_id = "other"'),
                         CORE.read_text(), WRAPPER.read_text(), header, PROFILE,
                         shape=STREAM_SHAPE, block_k=256, causal=True)
    disabled = deepcopy(PROFILE)
    disabled["resources"]["scratchpad_banks"] = 2
    with pytest.raises(ValueError, match="four-bank"):
        render_flash_mid(frontend, CORE.read_text(), WRAPPER.read_text(),
                         header, disabled, shape=STREAM_SHAPE, block_k=256,
                         causal=True)
    with pytest.raises(ValueError, match="source header"):
        render_flash_mid(frontend, CORE.read_text(), WRAPPER.read_text(),
                         header, PROFILE, shape=STREAM_SHAPE, block_k=256)


def test_archived_causal_source_matches_every_compiler_word() -> None:
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    manifest = json.loads((ARCHIVE / "object/object_manifest.json").read_text())
    assert receipt["status"] == "typed_causal_streamed_flash_full_bf16_source_parity"
    assert receipt["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert receipt["expected_output_hash"] == "23d84ffca3505276"
    assert receipt["compared_bf16_words"] == 4096
    assert receipt["source_runs"] == receipt["compiler_runs"] == 2
    assert receipt["object_sha256"] == manifest["object_sha256"] == _sha(
        ARCHIVE / "object/mx_issue.o")
    assert manifest["causal"] is True
    assert (manifest["command_count"], manifest["fence_count"]) == (581, 50)
    assert manifest["allocated_data_section_bytes"] == 0
    assert manifest["buffer_abi"][-1]["name"] == "MASK_BF16"
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
