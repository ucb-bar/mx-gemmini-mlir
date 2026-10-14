"""Check the pinned SmolVLA attention cutpoint against eager masked softmax."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parent
EXPECTED_SHA256 = "6b5b713660c819aa8f69a5174464b46258491bc81a86a8b4073b6c56cc9f122a"


def main() -> None:
    path = ROOT / "cutpoint.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256, digest
    with np.load(path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {"q", "kt", "p", "v", "mask"}
        q, kt, p, v, mask = (arrays[key] for key in ("q", "kt", "p", "v", "mask"))
    assert q.shape == (15, 50, 64)
    assert kt.shape == (15, 64, 163)
    assert p.shape == (15, 50, 163)
    assert v.shape == (15, 163, 64)
    assert mask.shape == (1, 1, 50, 163) and mask.dtype == np.bool_
    assert all(a.dtype == np.float32 for a in (q, kt, p, v))
    valid = np.broadcast_to(mask.reshape(50, 163), p.shape)
    assert np.all(valid.any(axis=-1))
    scores = np.where(valid, (q @ kt) / 8.0, np.finfo(np.float32).min)
    exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    expected = exp / exp.sum(axis=-1, keepdims=True)
    rmse = float(np.sqrt(np.mean((expected - p) ** 2)))
    max_abs = float(np.max(np.abs(expected - p)))
    assert rmse < 5e-9 and max_abs < 3e-7, (rmse, max_abs)
    assert np.all(p[~valid] == 0)
    receipt = json.loads((ROOT / "source_receipt.json").read_text())
    for name, key in (
        ("cutpoint.npz", "cutpoint_sha256"),
        ("attn_flash_smol.patch", "source_patch_sha256"),
        ("smol_head0.h", "generated_header_sha256"),
        ("driver.c", "driver_sha256"),
        ("smol_expect.h", "expected_hash_header_sha256"),
        ("source_spike.log", "source_log_sha256"),
    ):
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == receipt[key], name
    log = (ROOT / "source_spike.log").read_text()
    assert "attn_flash PASSED" in log
    assert "check O hash vs Spike: match (437d801250f000dd)" in log
    assert re.search(r"check O hash: serial 437d801250f000dd, pipelined 437d801250f000dd \(equal\)", log)
    assert re.search(r"PERF serial\s+33961 cycles", log)
    assert re.search(r"PERF pipelined\s+22447 cycles", log)
    assert re.search(r"accuracy O head 0 \(50 rows\): rel_fro 34425 ppm", log)
    all_heads = json.loads((ROOT / "source_all_heads.json").read_text())
    rows = all_heads["rows"]
    assert [row["head"] for row in rows] == list(range(15))
    assert all_heads["cutpoint_sha256"] == digest
    assert all_heads["sum_of_separately_timed_head_cycles"] == 336705
    for row in rows:
        head_log = ROOT / "source_all_heads" / f"head_{row['head']:02d}.log"
        assert hashlib.sha256(head_log.read_bytes()).hexdigest() == row["source_log_sha256"]
        content = head_log.read_text()
        assert "attn_flash PASSED" in content
        assert (f"check O hash: serial {row['output_hash']}, pipelined "
                f"{row['output_hash']} (equal)") in content
        assert re.search(rf"PERF serial\s+{row['serial_cycles']} cycles", content)
        assert re.search(rf"PERF pipelined\s+{row['pipelined_cycles']} cycles", content)
        assert (f"rel_fro {row['relative_frobenius_error_ppm_vs_captured_fp32']} ppm"
                in content)
        assert row["pipelined_cycles"] == 22447
    short_path = ROOT / "cutpoint113.npz"
    short_digest = hashlib.sha256(short_path.read_bytes()).hexdigest()
    assert short_digest == "19fcd4c7ec2e39870632ad0b97074b240029afd77c78b98bf8c148eae338f356"
    with np.load(short_path, allow_pickle=False) as arrays:
        sq, skt, sp, sv, short_mask = (
            arrays[key] for key in ("q", "kt", "p", "v", "mask"))
    assert (sq.shape, skt.shape, sp.shape, sv.shape, short_mask.shape) == (
        (15, 50, 64), (15, 64, 113), (15, 50, 113),
        (15, 113, 64), (1, 1, 50, 113))
    assert short_mask.dtype == np.bool_ and np.all(short_mask)
    short_scores = (sq @ skt) / 8.0
    short_exp = np.exp(short_scores - short_scores.max(axis=-1, keepdims=True))
    short_expected = short_exp / short_exp.sum(axis=-1, keepdims=True)
    short_rmse = float(np.sqrt(np.mean((short_expected - sp) ** 2)))
    assert short_rmse < 3e-8, short_rmse
    short_heads = json.loads((ROOT / "source_113_all_heads.json").read_text())
    short_rows = short_heads["rows"]
    assert short_heads["cutpoint_sha256"] == short_digest
    assert [row["head"] for row in short_rows] == list(range(15))
    assert short_heads["sum_of_separately_timed_head_cycles"] == 251625
    assert (hashlib.sha256((ROOT / "smol_head113.h").read_bytes()).hexdigest()
            == short_rows[0]["generated_header_sha256"])
    for row in short_rows:
        head_log = ROOT / "source_113_all_heads" / f"head_{row['head']:02d}.log"
        assert hashlib.sha256(head_log.read_bytes()).hexdigest() == row["source_log_sha256"]
        content = head_log.read_text()
        assert "attn_flash PASSED" in content
        assert (f"check O hash: serial {row['output_hash']}, pipelined "
                f"{row['output_hash']} (equal)") in content
        assert re.search(rf"PERF serial\s+{row['serial_cycles']} cycles", content)
        assert re.search(rf"PERF pipelined\s+{row['pipelined_cycles']} cycles", content)
        assert (f"rel_fro {row['relative_frobenius_error_ppm_vs_captured_fp32']} ppm"
                in content)
        assert row["pipelined_cycles"] == 16775
    result = {
        "cutpoint_sha256": digest,
        "heads": 15,
        "queries": 50,
        "keys": 163,
        "head_width": 64,
        "valid_mask_fraction": float(valid.mean()),
        "probability_rmse": rmse,
        "probability_max_abs": max_abs,
        "attention_output_shape": list((p @ v).shape),
        "source_first_head_pipelined_cycles": receipt["pipelined_cycles"],
        "source_first_head_relative_error_ppm": receipt["relative_frobenius_error_ppm_vs_captured_fp32"],
        "source_verified_heads": len(rows),
        "source_113_key_verified_heads": len(short_rows),
        "source_113_key_pipelined_cycles_each": 16775,
        "source_113_key_error_ppm_range": [
            min(row["relative_frobenius_error_ppm_vs_captured_fp32"]
                for row in short_rows),
            max(row["relative_frobenius_error_ppm_vs_captured_fp32"]
                for row in short_rows),
        ],
        "source_113_key_probability_rmse": short_rmse,
        "source_head_error_ppm_range": [
            min(row["relative_frobenius_error_ppm_vs_captured_fp32"] for row in rows),
            max(row["relative_frobenius_error_ppm_vs_captured_fp32"] for row in rows),
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
