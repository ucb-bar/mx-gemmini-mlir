"""Check archived compiler-issued VPU dependencies and source coverage."""

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.command_ir import Fence, VPU_OPCODES, emit_c
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vector_lowering import lower_vector_commands
from tools.qualify_nicolas_vpu_fused import PROFILE, SOURCE_RECEIPT
from tools.qualify_nicolas_vpu_ordering import (
    CASES, SOURCE_CHECKS, SP_A, SP_B, SP_D, SP_R, commands, vector_specs,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_vpu_ordering_compiled_266c593"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compiled_vpu_ordering_spike_evidence():
    index = json.loads((EVIDENCE / "index.json").read_text())
    source = json.loads(SOURCE_RECEIPT.read_text())
    reproducibility = json.loads((EVIDENCE / "reproducibility.json").read_text())
    profile = load_profile(PROFILE)
    assert index["status"] == "compiler_vpu_ordering_source_checks_matched_reference"
    assert index["source_qualification_sha256"] == sha(SOURCE_RECEIPT)
    assert index["source_sha256"] == source["source_sha256"]
    assert index["reference_sha256"] == source["reference_sha256"]
    assert index["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["profile_sha256"] == profile_sha256(profile)
    assert [row["name"] for row in index["rows"]] == list(CASES)
    assert reproducibility["status"] == "two_independent_runs_identical"
    assert reproducibility["run_count"] == 2
    assert reproducibility["index_sha256"] == sha(EVIDENCE / "index.json")
    assert reproducibility["extension_sha256"] == index["extension_sha256"]
    assert reproducibility["compiler_source_closure_sha256"] == index[
        "compiler_source_closure_sha256"]
    assert len(reproducibility["matching_artifacts"]) == len(CASES) * 7
    assert sum(sum(row["compared_bf16"]) for row in index["rows"]) == 2176
    for row in index["rows"]:
        case = row["name"]
        folder = EVIDENCE / case
        assert row["source_checks"] == list(SOURCE_CHECKS[case])
        for name, key in (("frontend.mlir", "frontend_mlir_sha256"),
                          ("bound.mlir", "bound_mlir_sha256"),
                          ("binding.json", "binding_sha256"),
                          ("mx_issue.c", "issuer_sha256"),
                          ("mx_driver.c", "driver_sha256"),
                          ("spike.log", "spike_log_sha256")):
            assert sha(folder / name) == row[key]
        assert reproducibility["matching_artifacts"][
            f"{case}/ordering.elf"] == row["elf_sha256"]
        binding = json.loads((folder / "binding.json").read_text())
        assert binding["source_sha256"] == index["source_sha256"]
        assert binding["frontend_mlir_sha256"] == row["frontend_mlir_sha256"]
        assert binding["bound_mlir_sha256"] == row["bound_mlir_sha256"]
        assert binding["policy"]["source_case"] == list(SOURCE_CHECKS[case])
        frontend = (folder / "frontend.mlir").read_text()
        assert "torch.operator" not in frontend
        mlir = (folder / "bound.mlir").read_text()
        vector = lower_vector_commands(mlir, profile)
        assert [command.funct for command in vector] == [33] * len(vector_specs(case))
        for command, spec in zip(vector, vector_specs(case)):
            assert command.rs2.immediate & 0xf == VPU_OPCODES[spec["kind"]]
            assert command.rs1.immediate & 0x3fff == spec["src1"]
            assert command.rs1.immediate >> 14 & 0x3fff == spec["src2"]
            assert command.rs1.immediate >> 28 & 0x3fff == spec["dst"]
            assert command.rs2.immediate >> 16 == spec["imm"]
        stream = commands(case, mlir, profile)
        assert [command.funct for command in stream] == row["ordered_functs"]
        assert not any(isinstance(command, Fence) for command in stream)
        if case == "chain":
            assert [(spec["kind"], spec["src1"], spec["dst"])
                    for spec in vector_specs(case)] == [
                        ("add", SP_A, SP_D), ("muls", SP_D, SP_D),
                        ("rmax", SP_D, SP_R)]
        if case == "war":
            assert [command.funct for command in stream[11:16]] == [33, 2, 2, 2, 2]
            assert all(command.rs1.buffer == "a2" for command in stream[12:16])
        if case == "dual":
            assert [(spec["kind"], spec["dst"])
                    for spec in vector_specs(case)] == [
                        ("adds", SP_A + 0x400), ("exp", SP_R), ("mul", SP_D)]
        issuer = (folder / "mx_issue.c").read_text()
        assert emit_c(stream, transport="rocket_rocc",
                      buffers=("a", "a2", "b", "x", "out_x", "out_y", "output")) == issuer
        assert "fence" not in issuer
        driver = (folder / "mx_driver.c").read_text()
        assert all(token not in driver for token in (
            "gemmini_vpu(", "gemmini_config_ld(", "gemmini_extended_mvin(",
            "gemmini_extended_mvout("))
        assert row["status"] == "source_vpu_reference_matched_on_pinned_spike"
        log = (folder / "spike.log").read_text()
        expected = ("compiled ordering dual: 0 XY mismatches, 0 Z mismatches"
                    if case == "dual" else f"compiled ordering {case}: 0 mismatches")
        assert expected in log


def test_ordering_receipts_close_remaining_source_checks():
    source = json.loads(SOURCE_RECEIPT.read_text())
    index = json.loads((EVIDENCE / "index.json").read_text())
    checked = [name for row in index["rows"] for name in row["source_checks"]]
    assert checked == source["checks"][25:]
    assert len(source["checks"]) == 29
