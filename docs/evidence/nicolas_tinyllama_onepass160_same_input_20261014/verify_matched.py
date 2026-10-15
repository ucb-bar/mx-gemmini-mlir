"""Verify the matched TinyLlama source/compiler result without the checkpoint."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np


HERE = Path(__file__).resolve().parent


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check() -> None:
    manifest = json.loads((HERE / "matched_archive_manifest.json").read_text())
    assert manifest["schema"] == "mx_gemmini.nicolas_tinyllama_onepass_matched_archive.v1"
    for name, digest in manifest["files"].items():
        assert sha((HERE / name).read_bytes()) == digest, name
    comparison = json.loads((HERE / "matched_onepass_comparison.json").read_text())
    identity = json.loads((HERE / "input_identity.json").read_text())
    run = json.loads((HERE / "compiler_run.json").read_text())
    build = json.loads((HERE / "compiler_build.json").read_text())
    catalog = json.loads((HERE / "compiler_catalog.json").read_text())
    source = gzip.decompress((HERE / "source_timed.log.gz").read_bytes()).decode()
    compiler = (HERE / "compiler_spike.log").read_text()
    assert comparison["schema"] == "mx_gemmini.nicolas_tinyllama_onepass_comparison.v1"
    assert comparison["cycle_ratio_valid"] is False
    assert identity["source_and_compiler_token_ids_exact"] is True
    assert identity["source_and_compiler_fp32_weights_exact"] is True
    assert identity["matched_fp32_weight_tensors"] == 200
    assert comparison["input_identity_receipt_sha256"] == sha(
        (HERE / "input_identity.json").read_bytes())
    assert comparison["sha256"]["source_log"] == sha(source.encode())
    assert comparison["sha256"]["compiler_log"] == sha(compiler.encode())
    assert comparison["sha256"]["source_header"] == sha(
        (HERE / "source_header.h").read_bytes())
    assert comparison["sha256"]["source_blob"] == manifest[
        "unarchived_source_blob_sha256"]
    assert comparison["sha256"]["compiler_run_receipt"] == sha(
        (HERE / "compiler_run.json").read_bytes())
    assert comparison["sha256"]["compiler_build_receipt"] == sha(
        (HERE / "compiler_build.json").read_bytes())
    assert run["done"] and run["exit_code"] == 0
    assert run["build_receipt_sha256"] == sha(
        (HERE / "compiler_build.json").read_bytes())
    assert run["log_sha256"] == sha(compiler.encode())
    assert run["spike_environment"]["GEMMINI_PERF_CONFIG"] == "e4m3_vpu"
    assert run["spike_environment"]["GEMMINI_PERF_SET"] == "mem.host_tracking=0"
    assert build["selected_operations"] == len(catalog["bindings"]) == 200
    assert all(row["target_dtypes"][:2] == ["f8E4M3FN", "f8E4M3FN"]
               for row in catalog["bindings"])
    for name, expected_tiles, expected_vpu, expected_load in (
            ("source", 40970240, 880, 2713452544),
            ("compiler", 40970280, 44872, 4407640384)):
        perf = (HERE / f"{name}_perf.txt").read_text()
        tiles, vpu = re.search(
            r"mesh: (\d+) tiles, \d+ busy cycles; VPU: (\d+) commands",
            perf).groups()
        loaded = re.search(r"load: (\d+) bytes", perf).group(1)
        assert (int(tiles), int(vpu), int(loaded)) == (
            expected_tiles, expected_vpu, expected_load)

    source_cycles = int(re.search(r"^stage_cycles64 0x([0-9a-f]{16})$",
                                  source, re.M).group(1), 16)
    compiler_cycles = int(re.search(r"^METRIC cycles (\d+)$",
                                    compiler, re.M).group(1))
    assert "llama MODEL benchmark finished (22 layer(s) + lm_head)." in source
    assert "\nDONE\n" in "\n" + compiler
    assert source_cycles == comparison["source_stage_cycles"]
    assert compiler_cycles == comparison["compiler_stage_cycles"]
    words = re.search(r"^ARGMAX 160((?: \d+){160})$", compiler, re.M)
    assert words is not None
    compiler_tops = np.fromstring(words.group(1), dtype=np.int64, sep=" ")
    assert compiler_tops.tolist() == comparison["compiler_argmax_160"]

    rows = re.findall(r"^session_logits_bf16 (\d+) ([0-9a-f]{128000})$",
                      source, re.M)
    assert len(rows) == 33 and [int(i) for i, _ in rows] == list(range(33))
    source_bits = np.stack([
        np.frombuffer(bytes.fromhex(encoded), dtype=">u2").astype("<u2")
        for _, encoded in rows])
    reference_bits = np.frombuffer(
        (HERE / "source_reference_late_bf16.bin").read_bytes(),
        dtype="<u2").reshape(33, 32000)
    assert np.array_equal(source_bits, reference_bits)
    source_logits = (source_bits.astype("<u4") << 16).view("<f4")
    fp32_late = np.load(HERE / "fp32_late_logits.npy", allow_pickle=False)
    fp32_tops = np.load(HERE / "fp32_argmax_160.npy", allow_pickle=False)
    assert fp32_late.shape == (33, 32000) and fp32_tops.shape == (160,)
    assert np.array_equal(np.argmax(fp32_late, axis=1), fp32_tops[127:160])
    source_tops = np.argmax(source_logits, axis=1)
    assert comparison["source_vs_compiler_top_matches_33"] == int(
        np.count_nonzero(source_tops == compiler_tops[127:160])) == 24
    assert comparison["source_vs_fp32_top_matches_33"] == int(
        np.count_nonzero(source_tops == fp32_tops[127:160])) == 27
    assert comparison["compiler_vs_fp32_top_matches_160"] == int(
        np.count_nonzero(compiler_tops == fp32_tops)) == 103
    assert comparison["compiler_vs_fp32_top_matches_33"] == int(
        np.count_nonzero(compiler_tops[127:160] == fp32_tops[127:160])) == 20
    cosines = []
    for actual, expected in zip(source_logits, fp32_late):
        a, b = actual.astype(np.float64), expected.astype(np.float64)
        cosines.append(float(np.dot(a, b) /
                             (np.linalg.norm(a) * np.linalg.norm(b))))
    assert abs(float(np.mean(cosines)) - comparison[
        "source_vs_fp32_late_cosine_mean"]) < 1e-12
    print("matched IDs, FP8 bindings, 33 source rows, 160 top tokens, and cycles verified")


if __name__ == "__main__":
    check()
