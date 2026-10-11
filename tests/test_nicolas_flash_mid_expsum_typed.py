"""Fused EXPSUM flash lowering matches Nicolas's source on pinned Spike."""

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
from mx_gemmini_support.nicolas_flash_mid import lower_flash_mid, render_flash_mid
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_object import classify


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_flash_mid_expsum_typed_a042643_0344bcb"
DATA = ROOT / "docs/evidence/nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb"
WRAPPER = ROOT / "tools/fixtures/nicolas_flash_mid_expsum.c"
CORE = Path(os.getenv("NICOLAS_NPU_ROOT",
                   "/scratch/agustin/tmp/npu-exploration-mx-source-20261009")) / \
    "baremetal/mxgemmini/src/attn_flash.c"
PROFILE = load_profile(ROOT / ("profiles/gemmini-mx-cleanup-266c593/"
                               "MxE4M3Fp4VpuGemminiRocketConfig.json"))


def _inputs() -> tuple[str, str]:
    return ((DATA / "attention.model2mlir.mlir").read_text(),
            gzip.decompress((DATA / "attn_vpu_mid.h.gz").read_bytes()).decode())


def test_expsum_flash_graph_and_commands(tmp_path: Path) -> None:
    if not CORE.is_file():
        pytest.skip("pinned Nicolas flash source is unavailable")
    frontend, header = _inputs()
    graph, binding = render_flash_mid(frontend, CORE.read_text(),
                                      WRAPPER.read_text(), header, PROFILE,
                                      softmax_mode="expsum")
    assert graph == (ARCHIVE / "flash_mid_expsum.mlir").read_text()
    assert binding == json.loads((ARCHIVE / "binding.json").read_text())
    assert binding["policy"]["softmax_mode"] == "expsum"
    family, report = classify(graph, PROFILE)
    assert family == "flash_program"
    assert report == verify_ir(graph, PROFILE)
    assert (report["memory_phases"], report["flash_setups"],
            report["spad_matmuls"], report["vpu_commands"],
            report["spad_requants"]) == (19, 1, 8, 31, 4)
    assert graph.count('kind = "expsum"') == 4
    commands = lower_flash_mid(graph, frontend, CORE.read_text(),
                               WRAPPER.read_text(), header, PROFILE,
                               softmax_mode="expsum")
    assert sum(isinstance(item, Command) for item in commands) == 148
    assert sum(isinstance(item, Fence) for item in commands) == 21
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if executable.is_file():
        source = tmp_path / "flash_mid_expsum.mlir"
        source.write_text(graph)
        result = subprocess.run([str(executable.resolve()), str(source),
                                 "-o", "/dev/null"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_expsum_flash_rejects_wrong_mode_or_unavailable_vpu() -> None:
    if not CORE.is_file():
        pytest.skip("pinned Nicolas flash source is unavailable")
    frontend, header = _inputs()
    with pytest.raises(ValueError, match="unsupported policy"):
        render_flash_mid(frontend, CORE.read_text(), WRAPPER.read_text(),
                         header, PROFILE)
    disabled = deepcopy(PROFILE)
    disabled["resources"]["vpu_config"]["exp_sum"] = False
    with pytest.raises(ValueError, match="expsum is not built"):
        render_flash_mid(frontend, CORE.read_text(), WRAPPER.read_text(),
                         header, disabled, softmax_mode="expsum")


def test_archived_expsum_flash_matches_all_source_output_words() -> None:
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    obj = json.loads((ARCHIVE / "object/object_manifest.json").read_text())
    dispatch = json.loads((ARCHIVE / "object/compile_manifest.json").read_text())
    assert receipt["status"] == "typed_flash_mid_expsum_full_bf16_source_parity"
    assert receipt["compared_bf16_words"] == 4096
    assert receipt["source_runs"] == receipt["compiler_runs"] == 2
    assert receipt["expected_output_hash"] == "cd7aef4d68d1e0d9"
    assert receipt["object_sha256"] == obj["object_sha256"] == sha(
        ARCHIVE / "object/mx_issue.o")
    assert receipt["source_elf_sha256"] == sha(ARCHIVE / "flash_mid_expsum_source.elf")
    assert receipt["generated_elf_sha256"] == sha(ARCHIVE / "flash_mid_expsum_generated.elf")
    assert obj["softmax_mode"] == "expsum"
    assert obj["command_count"] == 148 and obj["fence_count"] == 21
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert dispatch["lowering_family"] == "flash_program"
    assert dispatch["verified_graph_counts"]["vpu_commands"] == 31
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
