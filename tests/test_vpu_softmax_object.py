"""The public MX object driver emits the proven BF16 softmax issuer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mx_gemmini_support.source_softmax import lower_softmax_program_from_ir
from mx_gemmini_support.target_profile import load_profile
from tools.compile_object import classify


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_vpu_softmax_full_ce54256"
PROFILE = (ROOT / "profiles/gemmini-mx-cleanup-266c593/"
           "MxE4M3Fp4VpuGemminiRocketConfig.json")


def test_public_softmax_family_rejects_changed_graph() -> None:
    mlir = (EVIDENCE / "softmax.profile_bound.mlir").read_text()
    profile = load_profile(PROFILE)
    assert classify(mlir, profile)[0] == "vpu_softmax"
    assert len(lower_softmax_program_from_ir(mlir, profile)) == 17
    mutated = mlir.replace('site_id = "softmax:5"',
                           'site_id = "softmax:unexpected"')
    with pytest.raises(ValueError, match="typed operation graph"):
        lower_softmax_program_from_ir(mutated, profile)
    mutated = mlir.replace("dst_row = 1024 : i32", "dst_row = 1025 : i32")
    with pytest.raises(ValueError, match="schedule differs"):
        lower_softmax_program_from_ir(mutated, profile)


def test_public_softmax_object_matches_spike_issuer(tmp_path: Path) -> None:
    rtl = Path(os.environ.get("MX_GEMMINI_RTL_ROOT",
                              "/scratch/agustin/tmp/gemmini-mx-cleanup-20261009"))
    riscv = Path(os.environ.get("MX_GEMMINI_RISCV_ROOT",
                                "/scratch/agustin/projects/chipyard/.conda-env/riscv-tools"))
    if not rtl.is_dir() or not (riscv / "bin/riscv64-unknown-elf-gcc").is_file():
        pytest.skip("pinned MX RTL or RISC-V toolchain is not installed")
    abi = tmp_path / "abi.json"
    abi.write_text(json.dumps({"schema": "mx_gemmini.vpu_softmax_buffer_map.v1",
                               "inputs": {"score": "score"},
                               "outputs": {"output": "output"}}))
    out = tmp_path / "object"
    subprocess.run([sys.executable, "-m", "tools.compile_object",
                    "--mlir", str(EVIDENCE / "softmax.profile_bound.mlir"),
                    "--profile", str(PROFILE), "--rtl-root", str(rtl),
                    "--riscv-root", str(riscv), "--abi-json", str(abi),
                    "--out-dir", str(out)], cwd=ROOT, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert (out / "mx_issue.c").read_bytes() == (EVIDENCE / "mx_issue.c").read_bytes()
    obj = json.loads((out / "object_manifest.json").read_text())
    dispatch = json.loads((out / "compile_manifest.json").read_text())
    assert dispatch["lowering_family"] == "vpu_softmax"
    assert obj["buffer_abi"][0]["slot"] == "score"
    assert obj["buffer_abi"][1]["slot"] == "output"
    assert obj["command_count"] == 17
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["object_sha256"] == hashlib.sha256((out / "mx_issue.o").read_bytes()).hexdigest()
