"""Check the source-bound EXPSUB/EXPSUM compiler and Spike evidence."""

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.command_ir import emit_c
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vector_lowering import lower_vector_commands
from tools.qualify_nicolas_vpu_fused import commands


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_vpu_fused_compiled_266c593"
SOURCE = ROOT / "docs/evidence/nicolas_vpu_source_all_ops_266c593/receipt.json"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compiled_fused_vpu_spike_evidence():
    index = json.loads((EVIDENCE / "index.json").read_text())
    baseline = json.loads(SOURCE.read_text())
    profile = load_profile(PROFILE)
    assert index["status"] == "compiler_fused_expsub_expsum_matched_source_reference"
    assert index["source_qualification_sha256"] == sha(SOURCE)
    assert index["source_sha256"] == baseline["source_sha256"]
    assert index["reference_sha256"] == baseline["reference_sha256"]
    assert index["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert index["profile_sha256"] == profile_sha256(profile)
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert [row["kind"] for row in index["rows"]] == ["expsub", "expsum"]
    for row in index["rows"]:
        kind = row["kind"]
        folder = EVIDENCE / kind
        for name, key in (("frontend.mlir", "frontend_mlir_sha256"),
                          ("bound.mlir", "bound_mlir_sha256"),
                          ("binding.json", "binding_sha256"),
                          ("mx_issue.c", "issuer_sha256"),
                          ("mx_driver.c", "driver_sha256"),
                          ("spike.log", "spike_log_sha256")):
            assert sha(folder / name) == row[key]
        binding = json.loads((folder / "binding.json").read_text())
        assert binding["source_sha256"] == index["source_sha256"]
        assert binding["frontend_mlir_sha256"] == row["frontend_mlir_sha256"]
        assert binding["bound_mlir_sha256"] == row["bound_mlir_sha256"]
        bound = (folder / "bound.mlir").read_text()
        vector = lower_vector_commands(bound, profile)
        assert len(vector) == 1 and vector[0].funct == 33
        assert vector[0].rs2.immediate & 0xf == (12 if kind == "expsub" else 13)
        stream = commands(kind, bound, profile)
        assert [command.funct for command in stream] == row["ordered_functs"]
        assert emit_c(stream, transport="rocket_rocc",
                      buffers=("a", "b", "output", "sums")) == (
                          folder / "mx_issue.c").read_text()
        driver = (folder / "mx_driver.c").read_text()
        assert all(token not in driver for token in (
            "gemmini_vpu(", "gemmini_config_ld(", "gemmini_extended_mvin("))
        assert row["status"] == "source_vpu_reference_matched_on_pinned_spike"
        assert row["compared_output_bf16"] == 512
        assert row["compared_sum_bf16"] == (128 if kind == "expsum" else 0)
        assert f"compiled fused {kind}: 0 output mismatches, 0 sum mismatches" in (
            folder / "spike.log").read_text()
