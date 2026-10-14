"""Verify the full-depth source VPU and compiler add-only Spike logs."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SCALAR_SOURCE = HERE.parent / "nicolas_tinyllama_full_source_20261014"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_logits(log: str) -> np.ndarray:
    words = re.search(r"^last_logits_bf16 ([0-9a-f]+)$", log, re.M).group(1)
    bits = np.fromiter((int(words[i:i + 4], 16)
                        for i in range(0, len(words), 4)), dtype=np.uint32)
    return (bits << 16).view(np.float32)


def compiler_logits(log: str) -> np.ndarray:
    chunks = []
    for line in log.splitlines():
        if line.startswith("OUT_CHUNK f32 "):
            words = line.split()
            assert int(words[2]) == sum(chunk.size for chunk in chunks)
            chunk = np.array([int(word) for word in words[4:]], dtype="<u4")
            assert chunk.size == int(words[3])
            chunks.append(chunk)
    bits = np.concatenate(chunks)
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


def main() -> None:
    receipt = json.loads((HERE / "full_receipt.json").read_text())
    scalar_receipt = json.loads((SCALAR_SOURCE / "receipt.json").read_text())
    assert receipt["schema"] == "mx_gemmini.nicolas_tinyllama_full_vpu_residual_spike.v1"
    assert receipt["cycle_ratio_valid"] is False
    assert receipt["source_timing_mode"] == receipt["compiler_timing_mode"] == "both"
    assert receipt["compiler_counter_csr"] == "cycle"
    assert receipt["compiler_counter_reads"] == 2
    counter_disassembly = (HERE / "compiler_counter_disassembly.txt").read_bytes()
    assert digest(counter_disassembly) == receipt["compiler_counter_disassembly_sha256"]
    assert len(re.findall(rb"\bcsrrs\s+\w+,cycle,zero\b", counter_disassembly)) == 2
    assert receipt["spike_isa"] == "rv64gc_zicntr"
    assert receipt["perf_config"] == "e4m3_vpu"
    assert receipt["perf_set"] == "mem.host_tracking=0"
    assert receipt["exact_fp32_checkpoint_tensors_matched"] == 200
    assert receipt["source_commit"] == scalar_receipt["source_commit"]
    assert receipt["rtl_commit"] == scalar_receipt["rtl_commit"]
    assert receipt["input_ids_npz_sha256"] == scalar_receipt["input_ids_npz_sha256"]
    assert receipt["spike_extlib_sha256"] == scalar_receipt["spike_extlib_sha256"]
    assert receipt["source_blob_sha256"] == scalar_receipt["source_blob_sha256"]
    for key in ("capture_model", "checkpoint_weights", "precision_overrides",
                "profile"):
        assert receipt[f"compiler_{key}_sha256"] == (
            scalar_receipt[f"compiler_{key}_sha256"])
    assert digest((HERE / "vpu_residual.patch").read_bytes()) == (
        receipt["vpu_patch_sha256"])

    source_raw = gzip.decompress((HERE / "source_full_vpu.log.gz").read_bytes())
    compiler_raw = gzip.decompress((HERE / "compiler_add_zicntr.log.gz").read_bytes())
    scalar_raw = gzip.decompress((SCALAR_SOURCE / "source_spike.log.gz").read_bytes())
    assert digest(source_raw) == receipt["source_vpu_spike_log_sha256"]
    assert digest(compiler_raw) == receipt["compiler_spike_log_sha256"]
    assert digest(scalar_raw) == scalar_receipt["source_spike_log_sha256"]
    source_text, compiler_text, scalar_text = (
        raw.decode() for raw in (source_raw, compiler_raw, scalar_raw))
    for log in (source_text, compiler_text):
        assert "gemmini: GEMMINI_MODE=both" in log
    for name in ("source_perf_report", "compiler_perf_report"):
        data = (HERE / f"{name}.txt").read_bytes()
        assert digest(data) == receipt[f"{name}_sha256"]
        assert data.startswith(b"gemmini perf [e4m3_vpu]:")
    assert re.search(r"^llama MODEL benchmark finished \(22 layer\(s\) \+ lm_head\).$",
                     source_text, re.M)
    assert re.search(r"^DONE$", compiler_text, re.M)
    assert re.search(r"^METRIC build_hash ([0-9a-f]+)$",
                     compiler_text, re.M).group(1) == receipt["compiler_build_hash"]
    source = source_logits(source_text)
    compiler = compiler_logits(compiler_text)
    scalar = source_logits(scalar_text)
    reference_bytes = (SCALAR_SOURCE / "pytorch_last_logits.npy").read_bytes()
    assert digest(reference_bytes) == scalar_receipt["pytorch_last_logits_npy_sha256"]
    reference = np.load(SCALAR_SOURCE / "pytorch_last_logits.npy",
                        allow_pickle=False)
    assert source.size == compiler.size == scalar.size == reference.size == 32000
    assert bool(np.array_equal(source.view(np.uint32), scalar.view(np.uint32))) == (
        receipt["source_vpu_matches_scalar_bits"])
    assert digest(source.tobytes()) == receipt["source_vpu_logits_sha256"]
    assert digest(compiler.tobytes()) == receipt["compiler_logits_sha256"]
    assert int(re.search(r"^stage_cycles64 0x([0-9a-f]+)$", source_text,
                         re.M).group(1), 16) == receipt["source_stage_cycles"]
    assert int(re.search(r"^METRIC cycles (\d+)$", compiler_text,
                         re.M).group(1)) == receipt["compiler_stage_cycles"]
    for name, observed in {
        "source_vpu_vs_compiler": compare(source, compiler),
        "source_vpu_vs_pytorch": compare(source, reference),
        "compiler_vs_pytorch": compare(compiler, reference),
    }.items():
        for key, value in observed.items():
            assert abs(value - receipt["comparisons"][name][key]) < 1e-9
    print("verified full-depth source VPU and compiler outputs and cycle counters")


if __name__ == "__main__":
    main()
