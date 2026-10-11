"""The archived PV payload must consume the pinned QK/VPU probability bytes."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import (
    load_bundle, manifest_sha256, validate_model2mlir_attention_pv,
)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/model2mlir_tinyllama_connected_attention_head_a042643_266c593"
SOFTMAX = ROOT / "docs/evidence/model2mlir_tinyllama_attention_softmax_a042643_266c593"
QK = ROOT / "docs/evidence/model2mlir_four_model_fp8_qk_vpu_a042643_266c593/tinyllama"
PV = ROOT / "docs/evidence/model2mlir_four_model_fp8_pv_vpu_a042643_266c593/tinyllama"
FRONTEND = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984/tinyllama.mlir.gz"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_connected_pv_tracks_source_graph_and_exact_vpu_probability() -> None:
    summary = json.loads((ARCHIVE / "summary.json").read_text())
    softmax = json.loads((SOFTMAX / "summary.json").read_text())
    pv_receipt = json.loads((ARCHIVE / "source/pv_input_receipt.json").read_text())
    manifest, resources = load_bundle(ARCHIVE / "bundle")
    qk_manifest, qk_resources = load_bundle(QK / "bundle")
    frontend = gzip.decompress(FRONTEND.read_bytes()).decode()
    worklist = build_model2mlir_worklist(frontend, PROFILE)
    pair = [row for row in worklist["rank3_batch_matmuls"]
            if row["fqn"] == summary["site"]["fqn"]]
    producer = manifest["upstream_producer"]
    probability = np.load(ARCHIVE / "connected_activation.npy", allow_pickle=False)
    words = np.frombuffer((SOFTMAX / "probability_head0_bf16.bin").read_bytes(),
                          dtype="<u2")
    assert len(pair) == 2
    assert [row["region_id"] for row in pair] == ["matmul_4", "matmul_5"]
    assert summary["site"]["source_mlir_sha256"] == worklist["source_mlir_sha256"]
    assert summary["site"]["source_shape_bmnk"] == pair[1]["shape_bmnk"]
    assert qk_manifest["model2mlir_attention_qk"]["source_shape_bmnk"] == pair[0]["shape_bmnk"]
    assert producer == summary["upstream_producer"]
    assert producer["full_model_mlir_sha256"] == worklist["source_mlir_sha256"]
    assert producer["softmax_summary_sha256"] == _sha(SOFTMAX / "summary.json")
    assert producer["probability_bf16_sha256"] == _sha(SOFTMAX / "probability_head0_bf16.bin")
    assert producer["qk_golden_bf16_sha256"] == hashlib.sha256(
        qk_resources["golden_bf16"]).hexdigest()
    assert producer["head_shape_mk"] == list(probability.shape) == [8, 8]
    assert probability.dtype == np.float32
    assert np.array_equal(probability, (words.astype(np.uint32) << 16).view(
        np.float32).reshape(8, 8))
    assert _sha(ARCHIVE / "connected_activation.npy") != _sha(SOFTMAX / "probability.npy")
    assert summary["site"]["input_sha256"] == hashlib.sha256(
        probability.tobytes()).hexdigest()
    assert summary["site"]["weight_sha256"] == pv_receipt["weight_sha256"]
    assert _sha(ARCHIVE / "source/value.npy") == pv_receipt["weight_npy_sha256"]
    assert manifest["profile_sha256"] == profile_sha256(PROFILE)
    assert summary["scope"].endswith("on-device routing uncompiled")
    assert summary["padded_shape_mnk"] == [16, 64, 32]
    assert len(resources["golden_bf16"]) == 2 * 16 * 64


def test_connected_pv_lowering_and_spike_receipt() -> None:
    summary = json.loads((ARCHIVE / "summary.json").read_text())
    manifest, resources = load_bundle(ARCHIVE / "bundle")
    obj = json.loads((ARCHIVE / "object/object_manifest.json").read_text())
    run = json.loads((ARCHIVE / "run/artifact_manifest.json").read_text())
    bound = (ARCHIVE / "pv.payload_bound.mlir").read_text()
    physical = gzip.decompress((ARCHIVE / "object/physical_program.json.gz").read_bytes())
    assert verify_ir(bound, PROFILE)["contracts"] == 1
    assert lower_bound_source(bound, PROFILE, manifest, resources).receipt() == json.loads(
        physical)
    assert hashlib.sha256(physical).hexdigest() == obj["physical_program_sha256"]
    assert obj["payload_manifest_sha256"] == manifest_sha256(manifest)
    assert obj["object_sha256"] == summary["object_sha256"] == json.loads(
        (PV / "object/object_manifest.json").read_text())["object_sha256"]
    assert obj["allocated_data_section_bytes"] == obj["embedded_operand_bytes"] == 0
    assert run["status"] == "model_attention_pv_matched_on_pinned_spike"
    assert run["compared_bf16_outputs"] == summary["compared_bf16_values"] == 1024
    assert run["elf_sha256"] == summary["elf_sha256"]
    assert _sha(ARCHIVE / "run/artifact_manifest.json") == summary["run_receipt_sha256"]
    assert _sha(ARCHIVE / "run/spike.log") == run["spike_log_sha256"]
    assert "lowered MX 16x64x32: 0 BF16 mismatches" in (
        ARCHIVE / "run/spike.log").read_text()


def test_full_chain_quality_and_two_run_reproduction() -> None:
    quality = json.loads((ARCHIVE / "source_comparison.json").read_text())
    source_probability = np.load(ARCHIVE / "source/probability.npy", allow_pickle=False)
    value = np.load(ARCHIVE / "source/value.npy", allow_pickle=False)
    manifest, resources = load_bundle(ARCHIVE / "bundle")
    golden_words = np.frombuffer(resources["golden_bf16"], dtype="<u2").reshape(16, 64)
    result = (golden_words.astype(np.uint32) << 16).view(np.float32)[:8, :64]
    source = source_probability @ value
    error = np.abs(source - result)
    assert source_probability.shape == (8, 8) and value.shape == (8, 64)
    assert quality["scope"].endswith("separately executed QK, host scale/mask, VPU softmax, and PV")
    assert float(error.mean()) == pytest.approx(
        quality["source_f32_vs_compiled_chain_bf16_mean_absolute_error"])
    assert float(error.max()) == pytest.approx(
        quality["source_f32_vs_compiled_chain_bf16_max_absolute_error"])
    assert hashlib.sha256(resources["golden_bf16"]).hexdigest() == quality[
        "connected_pv_golden_bf16_sha256"]
    replay = json.loads((ARCHIVE / "reproduction.json").read_text())
    assert replay["run_count"] == 2 and len(replay["stable_files_sha256"]) == 15
    assert replay["status"] == "two_clean_spike_runs_stable_except_path_bearing_link_log"
    for relative, digest in replay["stable_files_sha256"].items():
        if relative == "object/mx_issue.o":
            assert digest == json.loads((ARCHIVE / "object/object_manifest.json").read_text())[
                "object_sha256"]
        elif relative == "object/physical_program.json":
            assert hashlib.sha256(gzip.decompress(
                (ARCHIVE / "object/physical_program.json.gz").read_bytes())).hexdigest() == digest
        elif relative == "run/mx_program.elf":
            assert digest == json.loads((ARCHIVE / "summary.json").read_text())["elf_sha256"]
        else:
            assert _sha(ARCHIVE / relative) == digest
    for relative, excluded in (("run/artifact_manifest.json", "link.log"),
                               ("summary.json", "run_receipt_sha256")):
        receipt = json.loads((ARCHIVE / relative).read_text())
        if relative.startswith("run/"):
            del receipt["build_log_sha256"][excluded]
        else:
            del receipt[excluded]
        canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(canonical).hexdigest() == replay[
            "normalized_receipts_sha256"][relative]


def test_connected_manifest_rejects_forged_producer() -> None:
    manifest, _ = load_bundle(ARCHIVE / "bundle")
    changed = json.loads(json.dumps(manifest))
    changed["upstream_producer"]["full_model_mlir_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="upstream producer differs"):
        validate_model2mlir_attention_pv(changed)
    changed = json.loads(json.dumps(manifest))
    changed["upstream_producer"]["head_shape_mk"] = [16, 32]
    with pytest.raises(ValueError, match="upstream producer differs"):
        validate_model2mlir_attention_pv(changed)
