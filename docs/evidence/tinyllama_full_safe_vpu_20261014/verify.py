"""Recheck the full-depth TinyLlama safe-VPU Spike result."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "nicolas_tinyllama_full_source_20261014"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    return {
        "cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "a_argmax": int(np.argmax(a)),
        "b_argmax": int(np.argmax(b)),
    }


def main() -> None:
    receipt = json.loads((HERE / "receipt.json").read_text())
    source_receipt_bytes = (SOURCE / "receipt.json").read_bytes()
    source_receipt = json.loads(source_receipt_bytes)
    assert digest(source_receipt_bytes) == receipt["source_evidence_receipt_sha256"]
    assert receipt["cycle_ratio_valid"] is False
    assert receipt["model_scope"].startswith("64-token prefill")
    for key in ("capture_model", "checkpoint_weights", "precision_overrides",
                "profile"):
        assert receipt[f"{key}_sha256"] == source_receipt[f"compiler_{key}_sha256"]
    assert receipt["spike_extlib_sha256"] == source_receipt["spike_extlib_sha256"]
    assert receipt["source_log_sha256"] == source_receipt["source_spike_log_sha256"]

    raw = gzip.decompress((HERE / "spike.log.gz").read_bytes())
    assert digest(raw) == receipt["spike_log_sha256"]
    text = raw.decode()
    assert re.search(r"^DONE$", text, re.M)
    assert int(re.search(r"^METRIC cycles (\d+)$", text, re.M).group(1)) == (
        receipt["stage_cycles"])
    chunks = []
    for line in text.splitlines():
        if line.startswith("OUT_CHUNK f32 "):
            words = line.split()
            assert int(words[2]) == sum(chunk.size for chunk in chunks)
            chunk = np.array([int(word) for word in words[4:]], dtype="<u4")
            assert chunk.size == int(words[3])
            chunks.append(chunk)
    compiler = np.concatenate(chunks).view("<f4")
    assert compiler.size == 32000
    assert digest(compiler.tobytes()) == receipt["logits_sha256"]
    assert digest(compiler.tobytes()) == re.search(
        r"^OUT_SHA256 f32le 32000 128000 ([0-9a-f]+)$", text, re.M).group(1)

    reference_bytes = (SOURCE / "pytorch_last_logits.npy").read_bytes()
    assert digest(reference_bytes) == receipt["reference_logits_sha256"]
    reference = np.load(SOURCE / "pytorch_last_logits.npy", allow_pickle=False)
    source_raw = gzip.decompress((SOURCE / "source_spike.log.gz").read_bytes())
    assert digest(source_raw) == receipt["source_log_sha256"]
    source_hex = re.search(r"^last_logits_bf16 ([0-9a-f]+)$",
                           source_raw.decode(), re.M).group(1)
    source_bits = np.fromiter((int(source_hex[i:i + 4], 16)
                               for i in range(0, len(source_hex), 4)),
                              dtype=np.uint32)
    source = (source_bits << 16).view(np.float32)
    for key, observed in {
        "compiler_vs_pytorch": compare(compiler, reference),
        "compiler_vs_source_quant": compare(compiler, source),
    }.items():
        for field, value in observed.items():
            assert abs(value - receipt["comparisons"][key][field]) < 1e-9
    print("verified full-depth stage output, cycles and comparison scope")


if __name__ == "__main__":
    main()
