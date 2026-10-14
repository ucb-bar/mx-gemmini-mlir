"""Verify the independent two-layer MX+VPU source and compiler control."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "nicolas_tinyllama_full_source_20261014"
COMPILER = HERE.parent / "tinyllama_l2_vpu_class_controls_20261014"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_logits(log: str) -> tuple[np.ndarray, str]:
    words = re.search(r"^last_logits_bf16 ([0-9a-f]+)$", log, re.M).group(1)
    bits = np.fromiter((int(words[i:i + 4], 16)
                        for i in range(0, len(words), 4)), dtype=np.uint32)
    return (bits << 16).view(np.float32), digest(words.encode())


def compiler_logits(log: str) -> np.ndarray:
    chunks = []
    for line in log.splitlines():
        if line.startswith("OUT_CHUNK f32 "):
            words = line.split()
            assert int(words[2]) == sum(chunk.size for chunk in chunks)
            chunk = np.array([int(word) for word in words[4:]], dtype="<u4")
            assert chunk.size == int(words[3])
            chunks.append(chunk)
    return np.concatenate(chunks).view("<f4")


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
    receipt = json.loads((HERE / "l2_receipt.json").read_text())
    source_receipt = json.loads((SOURCE / "receipt.json").read_text())
    compiler_summary = json.loads((COMPILER / "summary.json").read_text())
    assert receipt["cycle_ratio_valid"] is False
    assert receipt["source_internal_checks_passed"] is True
    assert receipt["source_spike_isa"] == "rv64gc_zicntr"
    assert receipt["compiler_spike_isa"] == "rv64gc"
    assert receipt["source_timing_mode"] == receipt["compiler_timing_mode"] == "func"
    assert receipt["nicolas_timing_model_active"] is False
    assert receipt["input_ids_npz_sha256"] == (
        source_receipt["input_ids_npz_sha256"])
    assert receipt["input_ids_npz_sha256"] == (
        compiler_summary["input_ids_npz_sha256"])
    assert receipt["spike_extlib_sha256"] == source_receipt["spike_extlib_sha256"]
    assert receipt["source_commit"] == source_receipt["source_commit"]
    assert receipt["rtl_commit"] == source_receipt["rtl_commit"]
    for name in ("vpu_residual.patch", "diagnostic_l2_wrapper.c", "timed_l2_wrapper.c"):
        key = ("vpu_patch_sha256" if name.endswith(".patch") else
               name + "_sha256")
        assert digest((HERE / name).read_bytes()) == receipt[key]

    logs = {}
    for kind in ("diagnostic", "timed"):
        raw = gzip.decompress((HERE / f"{kind}_l2head.log.gz").read_bytes())
        assert digest(raw) == receipt[f"{kind}_spike_log_sha256"]
        logs[kind] = raw.decode()
    assert re.search(r"^llama MODEL test PASSED ", logs["diagnostic"], re.M)
    assert re.search(r"^llama MODEL benchmark finished ", logs["timed"], re.M)
    diag, diag_hash = source_logits(logs["diagnostic"])
    timed, timed_hash = source_logits(logs["timed"])
    assert diag_hash == timed_hash == receipt["source_last_logits_hex_sha256"]
    assert np.array_equal(diag.view(np.uint32), timed.view(np.uint32))
    cycles = int(re.search(r"^stage_cycles64 0x([0-9a-f]+)$",
                           logs["timed"], re.M).group(1), 16)
    assert cycles == receipt["stage_cycles"]

    compiler_raw = gzip.decompress((COMPILER / "add.log.gz").read_bytes())
    assert digest(compiler_raw) == receipt["compiler_add_spike_log_sha256"] == (
        compiler_summary["results"]["add"]["spike_log_sha256"])
    compiler_log = compiler_raw.decode()
    compiler = compiler_logits(compiler_log)
    assert int(re.search(r"^METRIC cycles (\d+)$", compiler_log, re.M).group(1)) == (
        receipt["compiler_add_cycles"])
    reference_bytes = (COMPILER / "pytorch_last_logits.npy").read_bytes()
    assert digest(reference_bytes) == receipt["pytorch_logits_sha256"]
    reference = np.load(COMPILER / "pytorch_last_logits.npy", allow_pickle=False)
    assert diag.size == timed.size == compiler.size == reference.size == 32000
    for name, observed in {
        "source_vs_compiler_add": compare(timed, compiler),
        "source_vs_pytorch": compare(timed, reference),
        "compiler_add_vs_pytorch": compare(compiler, reference),
    }.items():
        for key, value in observed.items():
            assert abs(value - receipt["comparisons"][name][key]) < 1e-9
    print("verified two-layer source checks, timed output and compiler comparison")


if __name__ == "__main__":
    main()
