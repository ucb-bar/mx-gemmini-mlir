"""Recheck archived two-layer VPU class outputs and Spike cycles."""

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


def output(log: str) -> tuple[np.ndarray, str]:
    chunks = []
    offset = 0
    for line in log.splitlines():
        if not line.startswith("OUT_CHUNK f32 "):
            continue
        words = line.split()
        start, count = int(words[2]), int(words[3])
        assert start == offset
        chunk = np.array([int(word) for word in words[4:]], dtype="<u4")
        assert chunk.size == count
        chunks.append(chunk)
        offset += count
    assert offset == 32000
    bits = np.concatenate(chunks)
    wire_hash = re.search(r"^OUT_SHA256 f32le 32000 128000 ([0-9a-f]+)$",
                          log, re.M).group(1)
    assert digest(bits.tobytes()) == wire_hash
    return bits.view("<f4"), wire_hash


def main() -> None:
    summary = json.loads((HERE / "summary.json").read_text())
    ref_file = HERE / "pytorch_last_logits.npy"
    assert digest(ref_file.read_bytes()) == summary["golden_npy_sha256"]
    assert digest((HERE / "inputs.npz").read_bytes()) == summary["input_ids_npz_sha256"]
    reference = np.load(ref_file, allow_pickle=False).astype(np.float64).reshape(-1)
    assert reference.size == 32000
    assert int(np.argmax(reference)) == summary["pytorch_argmax"]
    for name, expected in summary["results"].items():
        raw = gzip.decompress((HERE / f"{name}.log.gz").read_bytes())
        assert digest(raw) == expected["spike_log_sha256"]
        log = raw.decode()
        values, wire_hash = output(log)
        assert wire_hash == expected["output_sha256"]["sha256"]
        values = values.astype(np.float64)
        cosine = float(np.dot(values, reference) /
                       (np.linalg.norm(values) * np.linalg.norm(reference)))
        cycles = int(re.search(r"^METRIC cycles ([0-9]+)$", log, re.M).group(1))
        assert abs(cosine - expected["cosine_to_pytorch"]) < 1e-9
        assert int(np.argmax(values)) == expected["argmax"]
        assert cycles == expected["cycles"]
    print(f"verified {len(summary['results'])} VPU class controls")


if __name__ == "__main__":
    main()
