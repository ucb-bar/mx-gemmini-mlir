"""The public object compiler reproduces both Spike-qualified flash issuers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
NPU = Path(os.getenv("NICOLAS_NPU_ROOT",
                  "/scratch/agustin/tmp/npu-exploration-mx-source-20261009"))
RTL = Path(os.getenv("NICOLAS_MX_RTL_ROOT",
                  "/scratch/agustin/tmp/gemmini-mx-cleanup-20261009"))
RISCV = Path(os.getenv("RISCV_ROOT",
                  "/scratch/agustin/projects/chipyard/.conda-env/riscv-tools"))
PROFILE = ROOT / ("profiles/gemmini-mx-cleanup-266c593/"
                  "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(("variant", "precursor_variant", "header_name", "wrapper_name"), [
    ("mid", "mid", "attn_vpu_mid.h.gz", "attn_flash_mid.c"),
    ("fa", "fa", "attn_vpu_fa.h.gz", "attn_flash.c"),
    ("mid_fp4", "mid", "attn_vpu_mid_fp4.h.gz",
     "tools/fixtures/nicolas_flash_mid_fp4.c"),
])
def test_public_flash_object_reproduces_archived_spike_issuer(
        tmp_path: Path, variant: str, precursor_variant: str, header_name: str,
        wrapper_name: str) -> None:
    core = NPU / "baremetal/mxgemmini/src/attn_flash.c"
    wrapper = (ROOT / wrapper_name if wrapper_name.startswith("tools/") else
               NPU / "baremetal/mxgemmini/src" / wrapper_name)
    gcc = RISCV / "bin/riscv64-unknown-elf-gcc"
    if not all(path.is_file() for path in (core, wrapper, gcc)) or not RTL.is_dir():
        pytest.skip("pinned Nicolas source, RTL, or RV64 compiler is unavailable")
    archive = ROOT / f"docs/evidence/nicolas_flash_{variant}_typed_a042643_0344bcb"
    archived_object = archive / "object" if variant == "mid_fp4" else archive
    precursor = ROOT / f"docs/evidence/nicolas_attn_vpu_{precursor_variant}_source_baseline_a042643_0344bcb"
    header = archive / header_name if variant == "mid_fp4" else precursor / header_name
    out = tmp_path / "object"
    command = [sys.executable, "-m", "tools.compile_object",
               "--mlir", str(archive / f"flash_{variant}.mlir"),
               "--precursor-mlir", str(precursor / "attention.model2mlir.mlir"),
               "--binding-json", str(archive / "binding.json"),
               "--source-core", str(core), "--source-wrapper", str(wrapper),
               "--source-header", str(header),
               "--profile", str(PROFILE), "--rtl-root", str(RTL),
               "--riscv-root", str(RISCV), "--out-dir", str(out)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for name in ("mx_issue.c", "mx_issue.h", "physical_program.json", "mx_issue.o"):
        assert _sha(out / name) == _sha(archived_object / name)
    dispatch = json.loads((out / "compile_manifest.json").read_text())
    assert dispatch["lowering_family"] == "flash_program"
    assert dispatch["verified_graph_counts"]["flash_setups"] == 1
    assert dispatch["object_sha256"] == _sha(archived_object / "mx_issue.o")

    changed = tmp_path / "changed_attn_flash.c"
    changed.write_text(core.read_text() + "\n/* changed source */\n")
    rejected_command = command.copy()
    rejected_command[rejected_command.index("--source-core") + 1] = str(changed)
    if wrapper == core:
        rejected_command[rejected_command.index("--source-wrapper") + 1] = str(changed)
    rejected_command[-1] = str(tmp_path / "rejected")
    rejected = subprocess.run(
        rejected_command,
        cwd=ROOT, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "flash object needs pinned Nicolas source core and wrapper" in rejected.stderr
    assert not (tmp_path / "rejected").exists()
