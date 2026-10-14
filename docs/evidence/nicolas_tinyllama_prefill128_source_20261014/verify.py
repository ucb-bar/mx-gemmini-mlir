"""Verify archived 128-token source and compiler prefill evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_logits(log: str) -> np.ndarray:
    match = re.search(r"^last_logits_bf16 ([0-9a-f]+)$", log, re.M)
    assert match is not None
    words = match.group(1)
    assert len(words) == 32000 * 4
    bits = np.fromiter((int(words[i:i + 4], 16)
                        for i in range(0, len(words), 4)), dtype=np.uint32)
    return (bits << 16).view(np.float32)


def compiler_logits(log: str) -> np.ndarray:
    chunks = []
    for line in log.splitlines():
        if not line.startswith("OUT_CHUNK f32 "):
            continue
        words = line.split()
        assert int(words[2]) == sum(chunk.size for chunk in chunks)
        chunk = np.array([int(word) for word in words[4:]], dtype="<u4")
        assert chunk.size == int(words[3])
        chunks.append(chunk)
    bits = np.concatenate(chunks)
    assert bits.size == 32000
    assert digest(bits.tobytes()) == re.search(
        r"^OUT_SHA256 f32le 32000 128000 ([0-9a-f]+)$", log, re.M).group(1)
    return bits.view("<f4")


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    return {
        "cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "a_argmax": int(np.argmax(a)),
        "b_argmax": int(np.argmax(b)),
    }


def cycle_counts(source_log: str, compiler_log: str) -> dict[str, int]:
    return {
        "source": int(re.search(r"^stage_cycles64 0x([0-9a-f]+)$",
                                source_log, re.M).group(1), 16),
        "compiler": int(re.search(r"^METRIC cycles (\d+)$",
                                  compiler_log, re.M).group(1)),
    }


def verify() -> None:
    receipt = json.loads((HERE / "receipt.json").read_text())
    assert receipt["schema"] == "mx_gemmini.tinyllama_prefill128_source.v1"
    assert receipt["cycle_ratio_valid"] is False
    assert receipt["scope"] == "one 128-token prefill, 22 layers and final head"
    assert receipt["spike_isa"] == "rv64gc_zicntr"
    assert receipt["perf_config"] == "e4m3_vpu"
    assert receipt["perf_set"] == "mem.host_tracking=0"
    assert receipt["checkpoint_fp32_tensors_matched"] == 200
    assert receipt["compiler_mx_bindings"] == 200
    assert receipt["compiler_mx_precision"] == "mxfp8"
    assert receipt["compiler_vpu_add_regions"] == 88
    assert receipt["source_vpu_residual_add_calls"] == 44
    assert receipt["source_output_matches_generated_quantized_logits"] is True
    assert receipt["source_attention_valid_keys"] == 128
    assert receipt["compiler_attention_key_slots"] == 160

    for name, expected in receipt["artifact_sha256"].items():
        assert digest((HERE / name).read_bytes()) == expected, name
    counter_text = (HERE / "compiler_counter_disassembly.txt").read_text()
    assert len(re.findall(r"\bcsrrs\s+\w+,cycle,zero\b", counter_text)) == 2
    assert not re.search(r"\bcsrrs\s+\w+,mcycle,zero\b", counter_text)
    for name in ("source_perf_report.txt", "compiler_perf_report.txt"):
        assert (HERE / name).read_bytes().startswith(b"gemmini perf [e4m3_vpu]:")
    with np.load(HERE / "compiler_inputs.npz", allow_pickle=False) as data:
        compiler_ids = data["in0"].reshape(-1)
    with np.load(HERE / "source_capture_inputs.npz", allow_pickle=False) as data:
        source_ids = data["in0"].reshape(-1)
    assert compiler_ids.size == 128 and source_ids.size == 129
    assert np.array_equal(source_ids[:128], compiler_ids)
    assert source_ids[128:].tolist() == receipt["source_next_token_label"]
    weight_receipt = json.loads((HERE / "weight_receipt.json").read_text())
    assert weight_receipt["exact_fp32_checkpoint_tensors_matched"] == 200
    assert weight_receipt["input_ids_npz_sha256"] == digest(
        (HERE / "compiler_inputs.npz").read_bytes())
    assert weight_receipt["compiler_checkpoint_weights_sha256"] == (
        receipt["compiler_checkpoint_weights_sha256"])
    overrides = json.loads((HERE / "all_fp8_overrides.json").read_text())
    assert len(overrides["regions"]) == 155
    assert set(overrides["regions"].values()) == {"mxfp8"}
    assert overrides["profile_sha256"] == receipt["profile_sha256"]
    catalog = json.loads((HERE / "catalog_summary.json").read_text())
    assert catalog["schema"] == "mx_gemmini.tinyllama_prefill128_catalog_summary.v1"
    assert catalog["profile_sha256"] == receipt["profile_sha256"]
    assert catalog["checkpoint_weights_sha256"] == (
        receipt["compiler_checkpoint_weights_sha256"])
    assert catalog["covered_contractions"] == len(catalog["bindings"]) == 200
    assert len(catalog["vpu_add_source_regions"]) == 88
    assert catalog["precision_overrides"] == overrides["regions"]
    assert all(binding["target_dtypes"][:2] == ["f8E4M3FN", "f8E4M3FN"]
               for binding in catalog["bindings"])
    attention = next(binding for binding in catalog["bindings"]
                     if binding["region"] == "matmul_4")
    assert attention["tensor_types"][:2] == [
        "tensor<32x128x64xf32>", "tensor<32x64x160xf32>"]

    source_log = gzip.decompress((HERE / "source_spike.log.gz").read_bytes()).decode()
    compiler_log = gzip.decompress((HERE / "compiler_spike.log.gz").read_bytes()).decode()
    diagnostic = gzip.decompress((HERE / "one_layer_diagnostic.log.gz").read_bytes()).decode()
    assert "llama MODEL test PASSED (1 layer(s) + lm_head" in diagnostic
    assert "llama MODEL benchmark finished (22 layer(s) + lm_head)." in source_log
    assert re.search(r"^DONE$", compiler_log, re.M)
    assert "gemmini: GEMMINI_MODE=both" in source_log
    assert "gemmini: GEMMINI_MODE=both" in compiler_log
    assert cycle_counts(source_log, compiler_log) == receipt["stage_cycles"]
    source = source_logits(source_log)
    compiler = compiler_logits(compiler_log)
    generated_last = np.frombuffer(
        (HERE / "source_generated_last_bf16.bin").read_bytes(), dtype="<u2")
    assert generated_last.size == 32000
    assert np.array_equal((source.view("<u4") >> 16).astype("<u2"),
                          generated_last)
    reference = np.load(HERE / "pytorch_last_logits.npy", allow_pickle=False).reshape(-1)
    assert reference.size == 32000
    for name, observed in {
        "source_vs_compiler": compare(source, compiler),
        "source_vs_pytorch": compare(source, reference),
        "compiler_vs_pytorch": compare(compiler, reference),
    }.items():
        for key, value in observed.items():
            assert abs(value - receipt["comparisons"][name][key]) < 1e-9
    print("verified 128-token full-depth prefill outputs and modeled cycle counts")


if __name__ == "__main__":
    verify()
