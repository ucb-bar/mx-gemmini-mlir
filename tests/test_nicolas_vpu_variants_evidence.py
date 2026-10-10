"""Verify archived compiler-issued VPU source variants and Spike receipts."""

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.command_ir import VPU_OPCODES, emit_c
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vector_lowering import lower_vector_commands
from tools.qualify_nicolas_vpu_fused import PROFILE, SOURCE_RECEIPT
from tools.qualify_nicolas_vpu_variants import CASES, SP_R, commands


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_vpu_variants_compiled_266c593"
BASE_EVIDENCE = ROOT / "docs/evidence/nicolas_vpu_elementwise_compiled_266c593/index.json"
SOURCE_CHECKS = {
    "mul_same_bank": ("mul same-bank",),
    "mul_bcast": ("mul bcast",),
    "max_same_bank": ("max same-bank",),
    "max_bcast": ("max bcast",),
    "sub_bcast_same_bank": ("sub bcast sb",),
    "expsub_plain": ("expsub",),
    "expsub_bcast": ("expsub bcast",),
    "expsub_same_bank": ("expsub sb",),
    "expsum_bcast": ("expsum bcast", "expsum sums"),
    "expsum_same_bank": ("expsum sb", "expsum sb sums"),
    "rsum_rlen1": ("rsum rlen1",),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compiled_vpu_variant_spike_evidence():
    index = json.loads((EVIDENCE / "index.json").read_text())
    baseline = json.loads(SOURCE_RECEIPT.read_text())
    reproducibility = json.loads((EVIDENCE / "reproducibility.json").read_text())
    profile = load_profile(PROFILE)
    assert index["status"] == "compiler_vpu_source_variants_matched_reference"
    assert index["source_qualification_sha256"] == sha(SOURCE_RECEIPT)
    assert index["source_sha256"] == baseline["source_sha256"]
    assert index["reference_sha256"] == baseline["reference_sha256"]
    assert index["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["profile_sha256"] == profile_sha256(profile)
    assert [row["name"] for row in index["rows"]] == [case.name for case in CASES]
    assert reproducibility["status"] == "two_independent_runs_identical"
    assert reproducibility["run_count"] == 2
    assert reproducibility["index_sha256"] == sha(EVIDENCE / "index.json")
    assert reproducibility["extension_sha256"] == index["extension_sha256"]
    assert reproducibility["compiler_source_closure_sha256"] == index[
        "compiler_source_closure_sha256"]
    assert len(reproducibility["matching_artifacts"]) == len(CASES) * 7
    assert sum(row["compared_output_bf16"] + row["compared_sum_bf16"]
               for row in index["rows"]) == 5888
    for case, row in zip(CASES, index["rows"]):
        folder = EVIDENCE / case.name
        for name, key in (("frontend.mlir", "frontend_mlir_sha256"),
                          ("bound.mlir", "bound_mlir_sha256"),
                          ("binding.json", "binding_sha256"),
                          ("mx_issue.c", "issuer_sha256"),
                          ("mx_driver.c", "driver_sha256"),
                          ("spike.log", "spike_log_sha256")):
            assert sha(folder / name) == row[key]
        assert reproducibility["matching_artifacts"][
            f"{case.name}/variant.elf"] == row["elf_sha256"]
        binding = json.loads((folder / "binding.json").read_text())
        policy = binding["policy"]
        assert binding["source_sha256"] == index["source_sha256"]
        assert binding["frontend_mlir_sha256"] == row["frontend_mlir_sha256"]
        assert binding["bound_mlir_sha256"] == row["bound_mlir_sha256"]
        assert policy["source_case"] == case.name
        assert policy["source2_array"] == case.second
        assert policy["src2_row"] == case.second_row
        assert policy["broadcast"] == case.broadcast
        assert policy["reduction_length"] == case.reduction_length
        frontend = (folder / "frontend.mlir").read_text()
        assert "torch.operator" not in frontend
        if case.broadcast:
            assert "tensor<16x8xbf16>" in frontend
            assert "output_shape [16, 4, 8]" in frontend
        mlir = (folder / "bound.mlir").read_text()
        vector = lower_vector_commands(mlir, profile)
        assert len(vector) == 1 and vector[0].funct == 33
        assert vector[0].rs2.immediate & 0xf == VPU_OPCODES[case.kind]
        assert bool(vector[0].rs2.immediate & 0x10) == case.broadcast
        assert (vector[0].rs1.immediate >> 14) & 0x3fff == case.second_row
        assert vector[0].rs2.immediate >> 5 & 0x3ff == case.reduction_length
        if case.kind == "expsum":
            assert vector[0].rs2.immediate >> 16 == SP_R
        stream = commands(case, mlir, profile)
        assert [command.funct for command in stream] == row["ordered_functs"]
        assert sum(command.funct == 2 for command in stream) == (
            4 + (1 if case.broadcast else 4) if case.second else 4)
        assert sum(command.funct == 3 for command in stream) == (
            case.output_rows // 16 + int(case.kind == "expsum"))
        assert emit_c(stream, transport="rocket_rocc",
                      buffers=("a", "a2", "b", "output", "sums")) == (
                          folder / "mx_issue.c").read_text()
        driver = (folder / "mx_driver.c").read_text()
        assert all(token not in driver for token in (
            "gemmini_vpu(", "gemmini_config_ld(", "gemmini_extended_mvin(",
            "gemmini_extended_mvout("))
        assert row["status"] == "source_vpu_reference_matched_on_pinned_spike"
        assert row["compared_output_bf16"] == case.output_rows * 8
        assert row["compared_sum_bf16"] == (128 if case.kind == "expsum" else 0)
        assert (f"compiled variant {case.name}: 0 output mismatches, "
                "0 sum mismatches") in (folder / "spike.log").read_text()


def test_all_single_operation_source_checks_have_compiler_receipts():
    source = json.loads(SOURCE_RECEIPT.read_text())
    base = json.loads(BASE_EVIDENCE.read_text())
    variants = json.loads((EVIDENCE / "index.json").read_text())
    assert [row["name"] for row in variants["rows"]] == list(SOURCE_CHECKS)
    covered = [row["kind"] for row in base["rows"]]
    covered += [name for row in variants["rows"]
                for name in SOURCE_CHECKS[row["name"]]]
    assert len(covered) == len(set(covered)) == 25
    assert set(covered) == set(source["checks"][:25])
    assert source["checks"][25:] == ["chain", "war mvin", "dual X,Y", "dual Z=X*B"]
