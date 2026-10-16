"""Recheck matched tokens and all 33 archived quality rows."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
COMPILER = EVIDENCE / "tinyllama_all_fp8_full_session_20261014"
SOURCE = EVIDENCE / "nicolas_tinyllama_onepass160_same_input_20261014"
sys.path.insert(0, str(HERE.parents[2]))
from tools.compare_mx_model_quality import _kl, compare_tinyllama  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    expected = json.loads((HERE / "receipt.json").read_text())
    for name, path in {
        "prefill": HERE / "prefill_inputs.npz",
        "decode": HERE / "decode_inputs.npz",
        "source_160": SOURCE / "input_ids.npz",
    }.items():
        assert digest(path) == expected["input_sha256"][name]
    with np.load(HERE / "prefill_inputs.npz", allow_pickle=False) as data:
        prefill = data["in0"].reshape(-1)
    with np.load(HERE / "decode_inputs.npz", allow_pickle=False) as data:
        decode = data["decode_tokens"].reshape(-1)
    with np.load(SOURCE / "input_ids.npz", allow_pickle=False) as data:
        source_ids = data["in0"].reshape(-1)
    assert prefill.size == 128 and decode.size == 32
    assert np.array_equal(np.concatenate((prefill, decode)), source_ids)

    files = {
        "compiler_run": COMPILER / "run.json",
        "source_timed_log_gz": SOURCE / "source_timed.log.gz",
        "compiler_prefill_fp32": COMPILER / "prefill_golden.npy",
        "compiler_decode_fp32": COMPILER / "decode_fp32.npz",
        "source_fp32_late": SOURCE / "fp32_late_logits.npy",
        "source_bf16_late": SOURCE / "source_reference_late_bf16.bin",
    }
    for name, path in files.items():
        assert digest(path) == expected["evidence_sha256"][name]
    run = json.loads(files["compiler_run"].read_text())
    assert run["complete"] is True
    dumps = {(item["program"], item["step"], item["output"]): item
             for item in run["output_dumps"]}
    guest = {(item["program"], item["step"], item["output"]): item
             for item in run["output_digests"]}
    keys = [(0, 0, 0)] + [(1, step, 0) for step in range(32)]
    rows = []
    for key in keys:
        item = dumps[key]
        path = COMPILER / item["path"]
        assert item["bytes"] == 128000
        assert digest(path) == item["sha256"] == guest[key]["sha256"]
        rows.append(np.frombuffer(path.read_bytes(), dtype="<f4"))
    compiler = np.stack(rows)
    assert hashlib.sha256(compiler.tobytes()).hexdigest() == (
        expected["compiler_logit_bytes_sha256"])

    raw = gzip.decompress(files["source_timed_log_gz"].read_bytes()).decode()
    source_rows = re.findall(r"^session_logits_bf16 (\d+) ([0-9a-f]{128000})$",
                             raw, re.MULTILINE)
    assert len(source_rows) == 33
    assert [int(index) for index, _ in source_rows] == list(range(33))
    source_bits = np.stack([
        np.frombuffer(bytes.fromhex(value), dtype=">u2").astype("<u2")
        for _, value in source_rows])
    assert source_bits.tobytes() == files["source_bf16_late"].read_bytes()
    source = (source_bits.astype("<u4") << 16).view("<f4")
    float_cached = np.concatenate((
        np.load(files["compiler_prefill_fp32"], allow_pickle=False),
        np.load(files["compiler_decode_fp32"], allow_pickle=False)["logits"][:, 0],
    ), axis=0)
    float_onepass = np.load(files["source_fp32_late"], allow_pickle=False)
    agreement = expected["fp32_reference_agreement"]
    assert float_cached.shape == float_onepass.shape == (33, 32000)
    assert abs(float(np.max(np.abs(float_cached - float_onepass))) -
               agreement["max_absolute_logit_difference"]) < 1e-8
    assert abs(float(np.mean(_kl(float_cached, float_onepass))) -
               agreement["mean_kl_nats"]) < 1e-8
    assert np.count_nonzero(np.argmax(float_cached, axis=1) ==
                            np.argmax(float_onepass, axis=1)) == 33
    report = compare_tinyllama(float_cached, source, compiler)
    for name in ("source", "compiler"):
        for metric in ("mean_kl_nats", "worst_kl_nats"):
            assert abs(report[name][metric] - expected[name][metric]) < 1e-8
        assert report[name]["top1_agreement"] == expected[name]["top1_agreement"]
    assert report["checks"] == expected["checks"]
    assert report["passed"] is expected["passed"] is False
    print("verified 33 aligned rows; compiler quality gate fails")


if __name__ == "__main__":
    main()
