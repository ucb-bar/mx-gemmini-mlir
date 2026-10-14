"""Recheck the archived TinyLlama Spike counters and complete logits."""

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
    expected = json.loads((HERE / "output_comparison.json").read_text())
    source = gzip.decompress((HERE / "source_spike.log.gz").read_bytes())
    compiler = gzip.decompress((HERE / "compiler_spike.log.gz").read_bytes())
    reference_file = HERE / "pytorch_last_logits.npy"
    assert digest(source) == receipt["source_spike_log_sha256"]
    assert digest(compiler) == receipt["compiler_spike_log_sha256"]
    assert digest(reference_file.read_bytes()) == receipt["pytorch_last_logits_npy_sha256"]

    source_text = source.decode()
    compiler_text = compiler.decode()
    logits_hex = re.search(r"^last_logits_bf16 ([0-9a-f]+)$", source_text, re.M).group(1)
    assert digest(logits_hex.encode()) == receipt["source_last_logits_hex_sha256"]
    source_bits = np.fromiter((int(logits_hex[i:i + 4], 16)
                               for i in range(0, len(logits_hex), 4)),
                              dtype=np.uint32)
    source_logits = (source_bits << 16).view(np.float32)

    chunks = []
    for line in compiler_text.splitlines():
        if line.startswith("OUT_CHUNK f32 "):
            words = line.split()
            offset, count = int(words[2]), int(words[3])
            assert offset == sum(chunk.size for chunk in chunks)
            chunk = np.array([int(word) for word in words[4:]], dtype="<u4")
            assert chunk.size == count
            chunks.append(chunk)
    compiler_bits = np.concatenate(chunks)
    compiler_logits = compiler_bits.view("<f4")
    reference = np.load(reference_file, allow_pickle=False).reshape(-1)
    assert compiler_logits.size == source_logits.size == reference.size == 32000
    wire_hash = re.search(r"^OUT_SHA256 f32le 32000 128000 ([0-9a-f]+)$",
                          compiler_text, re.M).group(1)
    assert digest(compiler_logits.tobytes()) == wire_hash

    for name, observed in {
        "compiler_vs_pytorch": compare(compiler_logits, reference),
        "compiler_vs_source_quant": compare(compiler_logits, source_logits),
        "source_quant_vs_pytorch": compare(source_logits, reference),
    }.items():
        recorded = expected["comparisons"][name]
        for key, value in observed.items():
            assert abs(value - recorded[key]) < 1e-9, (name, key, value, recorded[key])

    cycles = re.search(r"^cycles64 mesh 0x([0-9a-f]+) host 0x([0-9a-f]+)$",
                       source_text, re.M)
    mesh, host = (int(value, 16) for value in cycles.groups())
    assert (mesh, host) == (receipt["cycles"]["mx_mesh"],
                            receipt["cycles"]["scalar_host"])
    compiler_cycles = int(re.search(r"^METRIC cycles ([0-9]+)$",
                                    compiler_text, re.M).group(1))
    assert compiler_cycles == receipt["compiler_stage_cycles"]
    print("verified complete logits and raw cycle counters; no paired speedup")


if __name__ == "__main__":
    main()
