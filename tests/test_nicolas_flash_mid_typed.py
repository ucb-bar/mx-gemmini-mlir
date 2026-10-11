"""Whole four-block flash graph has an executable, source-bound serial lowering."""

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from mx_gemmini_support.command_ir import Command, Fence
from mx_gemmini_support.nicolas_flash_commands import lower_flash_commands
from mx_gemmini_support.nicolas_flash_mid import (BUFFERS, BUFFER_BYTES,
                                                  lower_flash_mid,
                                                  render_flash_mid)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/evidence/nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")
CORE = ("static uint64_t run_serial(void)\n"
        "gemmini_mx_load_scales_2d_gated\n"
        "gemmini_mxquant_config_mvout_managed\n")
WRAPPER = '#define BK 32\n#include "attn_flash.c"\n'
EVIDENCE = ROOT / "docs/evidence/nicolas_flash_mid_typed_a042643_0344bcb"


def _inputs() -> tuple[str, str]:
    return ((DATA / "attention.model2mlir.mlir").read_text(),
            gzip.decompress((DATA / "attn_vpu_mid.h.gz").read_bytes()).decode())


def test_serial_flash_mid_graph_lowers_all_blocks_and_vector_stages(tmp_path: Path) -> None:
    frontend, header = _inputs()
    mlir, binding = render_flash_mid(frontend, CORE, WRAPPER, header, PROFILE)
    report = verify_ir(mlir, PROFILE)
    assert (report["memory_phases"], report["flash_setups"],
            report["spad_matmuls"], report["vpu_commands"],
            report["spad_requants"]) == (19, 1, 8, 39, 4)
    commands = lower_flash_mid(mlir, frontend, CORE, WRAPPER, header, PROFILE)
    assert sum(isinstance(item, Command) for item in commands) == 156
    assert sum(isinstance(item, Fence) for item in commands) == 21
    assert binding["policy"]["score_rows"] == [8192, 12288, 8192, 12288]
    assert binding["policy"]["fenced"] is True
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if executable.is_file():
        source = tmp_path / "flash_mid.mlir"
        source.write_text(mlir)
        result = subprocess.run([str(executable.resolve()), str(source),
                                 "-o", "/dev/null"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_flash_mid_requires_gated_weight_slice_before_managed_matmul() -> None:
    frontend, header = _inputs()
    mlir, _ = render_flash_mid(frontend, CORE, WRAPPER, header, PROFILE)
    lines = mlir.splitlines()
    lines = [line for line in lines if 'site_id = "qk0:scales"' not in line]
    with pytest.raises(ValueError, match="lacks its gated"):
        lower_flash_commands("\n".join(lines) + "\n", PROFILE, BUFFERS,
                             BUFFER_BYTES)
    with pytest.raises(ValueError, match="typed graph"):
        lower_flash_mid(mlir.replace("weight_half = 0 : i32",
                                     "weight_half = 1 : i32", 1),
                        frontend, CORE, WRAPPER, header, PROFILE)


def test_archived_flash_mid_matches_every_bf16_output_twice() -> None:
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    manifest = json.loads((EVIDENCE / "object_manifest.json").read_text())
    physical = json.loads((EVIDENCE / "physical_program.json").read_text())
    sha = lambda name: hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
    assert receipt["status"] == "typed_flash_mid_full_bf16_source_parity"
    assert receipt["schedule"] == "fenced_serial"
    assert receipt["compared_bf16_words"] == 4096
    assert receipt["source_runs"] == receipt["compiler_runs"] == 2
    assert receipt["source_output_hash"] == "cd7aef4d68d1e0d9"
    assert manifest["command_count"] == 156 and manifest["fence_count"] == 21
    assert manifest["allocated_data_section_bytes"] == 0
    assert manifest["embedded_operand_bytes"] == manifest["embedded_golden_bytes"] == 0
    assert receipt["object_sha256"] == manifest["object_sha256"] == sha("mx_issue.o")
    assert receipt["bound_mlir_sha256"] == sha("flash_mid.mlir")
    assert receipt["physical_program_sha256"] == sha("physical_program.json")
    assert len(physical["commands"]) == 177
    words = []
    for kind in ("source", "generated"):
        for run in (1, 2):
            log = (EVIDENCE / f"{kind}_spike_{run}.log").read_text()
            line = next(line for line in log.splitlines()
                        if line.startswith("O BF16 words "))
            values = line.removeprefix("O BF16 words ")
            assert len(values) == 16384
            words.append(values)
    assert len(set(words)) == 1


