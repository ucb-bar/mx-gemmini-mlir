"""Model-derived QK scores must keep their source mask and VPU Spike result."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest

from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.source_softmax import lower_softmax_program_from_ir
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/model2mlir_tinyllama_attention_softmax_a042643_266c593"
QK = ROOT / "docs/evidence/model2mlir_four_model_fp8_qk_vpu_a042643_266c593/tinyllama"
FRONTEND = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984/tinyllama.mlir.gz"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_qk_scale_mask_and_vpu_probabilities() -> None:
    summary = json.loads((ARCHIVE / "summary.json").read_text())
    inputs = json.loads((ARCHIVE / "input_receipt.json").read_text())
    qk_manifest, qk_resources = load_bundle(QK / "bundle")
    frontend = gzip.decompress(FRONTEND.read_bytes())
    assert summary["status"] == "model_qk_to_vpu_softmax_matched_on_pinned_spike"
    assert summary["scope"] == (
        "model-derived QK BF16 scores are host scaled and causally masked, "
        "then compiled VPU softmax runs on Spike; PV remains separate")
    assert hashlib.sha256(frontend).hexdigest() == summary["full_model_mlir_sha256"]
    assert inputs["full_model_mlir_sha256"] == summary["full_model_mlir_sha256"]
    assert inputs["qk_region_id"] == qk_manifest["model2mlir_attention_qk"]["region_id"]
    assert inputs["fqn"] == qk_manifest["model2mlir_attention_qk"]["fqn"]
    assert inputs["softmax_region_id"] == summary["source_softmax_region_id"] == "softmax_0"
    assert b'prov.region_id = "softmax_0"' in frontend
    assert b'prov.aten = "aten._softmax.default"' in frontend
    assert inputs["attention_scale"] == 0.125
    assert inputs["mask"] == "strict_causal_upper_triangle_negative_infinity"
    assert inputs["masked_values"] == 28
    assert _sha(ARCHIVE / "input_receipt.json") == summary[
        "captured_softmax_receipt_sha256"]
    assert _sha(ARCHIVE / "pre_softmax.npy") == inputs["pre_softmax_npy_sha256"]
    assert _sha(ARCHIVE / "probability.npy") == inputs["probability_npy_sha256"]
    pre = np.load(ARCHIVE / "pre_softmax.npy", allow_pickle=False)
    source_prob = np.load(ARCHIVE / "probability.npy", allow_pickle=False)
    mask = np.triu(np.ones((8, 8), dtype=bool), 1)
    assert pre.shape == source_prob.shape == (8, 8)
    assert hashlib.sha256(pre.tobytes()).hexdigest() == inputs["pre_softmax_sha256"]
    assert hashlib.sha256(source_prob.tobytes()).hexdigest() == inputs["probability_sha256"]
    assert np.all(pre[mask] < -1e20) and np.all(source_prob[mask] == 0)
    assert np.allclose(source_prob.sum(axis=1), 1, atol=1e-5)
    assert _sha(QK / "bundle/golden_bf16.bin") == summary["qk_golden_bf16_sha256"]
    assert hashlib.sha256(qk_resources["golden_bf16"]).hexdigest() == (
        summary["qk_golden_bf16_sha256"])
    score = np.frombuffer((ARCHIVE / "score_bf16.bin").read_bytes(),
                          dtype="<u2").reshape(16, 32)
    qk = np.frombuffer(qk_resources["golden_bf16"], dtype="<u2").reshape(16, 32)
    assert _sha(ARCHIVE / "score_bf16.bin") == summary["score_bf16_sha256"]
    assert np.all(score[:8, :8][mask] == 0xff80)
    assert np.all(score[:8, 8:] == 0xff80) and np.all(score[8:] == 0)
    qk_f32 = (qk.astype(np.uint32) << 16).view(np.float32)
    score_f32 = (score.astype(np.uint32) << 16).view(np.float32)
    assert np.allclose(score_f32[:8, :8][~mask],
                       (qk_f32[:8, :8] * 0.125)[~mask], rtol=0.005, atol=0.005)
    output = np.frombuffer((ARCHIVE / "probability_head0_bf16.bin").read_bytes(),
                           dtype="<u2").reshape(8, 8)
    prob_f32 = (output.astype(np.uint32) << 16).view(np.float32)
    assert _sha(ARCHIVE / "probability_head0_bf16.bin") == summary[
        "probability_head0_bf16_sha256"]
    assert np.all(prob_f32[mask] == 0)
    assert np.max(np.abs(prob_f32.sum(axis=1) - 1)) == pytest.approx(
        summary["quality_diagnostics"]["probability_row_sum_max_absolute_error"])
    assert np.abs(prob_f32 - source_prob).mean() == pytest.approx(
        summary["quality_diagnostics"]["probability_mean_absolute_error_vs_source_f32"])


def test_typed_softmax_object_and_spike_reference() -> None:
    summary = json.loads((ARCHIVE / "summary.json").read_text())
    binding = json.loads((ARCHIVE / "binding_manifest.json").read_text())
    obj = json.loads((ARCHIVE / "object/object_manifest.json").read_text())
    mlir = (ARCHIVE / "softmax.profile_bound.mlir").read_text()
    report = verify_ir(mlir, PROFILE)
    assert report["vpu_commands"] == 6
    assert summary["profile_sha256"] == profile_sha256(PROFILE)
    assert _sha(ARCHIVE / "softmax.profile_bound.mlir") == summary["bound_mlir_sha256"]
    assert binding["bound_mlir_sha256"] == summary["bound_mlir_sha256"]
    physical = json.loads(gzip.decompress(
        (ARCHIVE / "object/physical_program.json.gz").read_bytes()))
    commands = lower_softmax_program_from_ir(mlir, PROFILE,
                                            score_buffer="score", output_buffer="output")
    assert len(commands) == obj["command_count"] == 17
    assert [row["funct"] for row in physical["commands"]] == [cmd.funct for cmd in commands]
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert obj["object_sha256"] == summary["object_sha256"]
    assert _sha(ARCHIVE / "object/mx_issue.c") == obj["issuer_c_sha256"]
    driver = (ARCHIVE / "run/mx_driver.c").read_text()
    assert driver.count("mx_issue(S, P_hw);") == 1
    assert "gemmini_vpu_" not in driver
    log = (ARCHIVE / "run/spike.log").read_text()
    assert "model attention softmax 16x32: 0 mismatches vs ref" in log
    assert "vpu_softmax PASSED" in log
    assert summary["compared_bf16_values"] == 512
    assert _sha(ARCHIVE / "run/spike.log") == summary["spike_log_sha256"]
    line = re.search(r"^probability_head0_bf16:((?: [0-9a-f]{4}){64})$",
                     log, re.MULTILINE)
    assert line is not None
    words = np.asarray([int(word, 16) for word in line.group(1).split()], dtype="<u2")
    assert words.tobytes() == (ARCHIVE / "probability_head0_bf16.bin").read_bytes()


def test_typed_softmax_rejects_changed_order() -> None:
    mlir = (ARCHIVE / "softmax.profile_bound.mlir").read_text()
    with pytest.raises(ValueError, match="typed operation graph"):
        lower_softmax_program_from_ir(
            mlir.replace('site_id = "softmax:5"', 'site_id = "softmax:wrong"'), PROFILE)


def test_two_clean_spike_runs_reproduced_artifact_hashes() -> None:
    reproduction = json.loads((ARCHIVE / "reproduction.json").read_text())
    summary = json.loads((ARCHIVE / "summary.json").read_text())
    hashes = reproduction["files_sha256"]
    assert reproduction["status"] == "two_clean_spike_runs_byte_identical"
    assert reproduction["run_count"] == 2 and len(hashes) == 15
    for archived, source in (("summary.json", "summary.json"),
                             ("score_bf16.bin", "score_bf16.bin"),
                             ("probability_head0_bf16.bin", "probability_head0_bf16.bin"),
                             ("softmax.model2mlir.mlir", "softmax.model2mlir.mlir"),
                             ("softmax.profile_bound.mlir", "softmax.profile_bound.mlir"),
                             ("object/mx_issue.c", "object/mx_issue.c"),
                             ("run/spike.log", "build/spike.log"),
                             ("run/mx_driver.c", "build/mx_driver.c")):
        assert _sha(ARCHIVE / archived) == hashes[source]
    assert hashlib.sha256(gzip.decompress(
        (ARCHIVE / "object/physical_program.json.gz").read_bytes())).hexdigest() == (
            hashes["object/physical_program.json"])
    assert summary["object_sha256"] == hashes["object/mx_issue.o"]
    assert summary["elf_sha256"] == hashes["build/mx_program.elf"]
