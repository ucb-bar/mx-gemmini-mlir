"""Compare dumped TinyLlama logits with captured FP32 references.

The captured tokens are synthetic. These metrics diagnose numerical error;
they are not a model quality acceptance threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    if (actual.shape != expected.shape or actual.shape != (1, 32000)
            or actual.dtype != expected.dtype or actual.dtype != np.float32
            or not np.isfinite(actual).all() or not np.isfinite(expected).all()):
        raise ValueError("TinyLlama logits need finite, equal 1×32000 f32 arrays")
    lhs, rhs = actual.astype(np.float64), expected.astype(np.float64)
    difference = lhs - rhs
    rmse = float(np.sqrt(np.mean(difference * difference)))
    reference_rms = float(np.sqrt(np.mean(rhs * rhs)))
    actual_top = int(np.argmax(actual))
    reference_top = int(np.argmax(expected))
    reference_top_rank = 1 + int(np.count_nonzero(actual > actual[0, reference_top]))
    return {
        "cosine": float(np.clip(np.sum(lhs * rhs) /
                                max(float(np.linalg.norm(lhs) * np.linalg.norm(rhs)),
                                    1e-300), -1.0, 1.0)),
        "rmse": rmse,
        "normalized_rmse": rmse / max(reference_rms, 1e-300),
        "actual_top_token": actual_top,
        "reference_top_token": reference_top,
        "top_token_equal": actual_top == reference_top,
        "reference_top_rank_in_actual": reference_top_rank,
    }


def compare_session(run_receipt: Path, link_receipt: Path,
                    session_root: Path, fast_run_receipt: Path,
                    fast_link_receipt: Path) -> dict:
    run = json.loads(run_receipt.read_text())
    link = json.loads(link_receipt.read_text())
    fast = json.loads(fast_run_receipt.read_text())
    fast_link = json.loads(fast_link_receipt.read_text())
    if (run.get("status") != "complete_numeric_qualification_pending"
            or run.get("complete") is not True or run.get("exit_code") != 0
            or run.get("link_receipt_sha256") != sha(link_receipt)
            or Path(link["session_root"]).resolve() != session_root.resolve()
            or [row["name"] for row in link["programs"]] != ["prefill", "decode"]
            or [row["steps"] for row in link["programs"]] != [1, 32]
            or link.get("dump_outputs") != [
                {"program": 0, "name": "prefill", "output": 0, "bytes": 128000},
                {"program": 1, "name": "decode", "output": 0, "bytes": 128000}]
            or link.get("dump_limit_bytes") != 128000):
        raise ValueError("TinyLlama run, link, and capture identities differ")
    if (link["stage_object_sha256"] != fast_link["stage_object_sha256"]
            or link["shared_weight_sha256"] != fast_link["shared_weight_sha256"]
            or [row["capture_model_sha256"] for row in link["programs"]]
            != [row["capture_model_sha256"] for row in fast_link["programs"]]
            or run["output_digests"] != fast["output_digests"]
            or len(run["output_digests"]) != 99):
        raise ValueError("dumped outputs differ from the existing full compiler run")

    prefill_path = session_root / "stages/prefill/golden.npy"
    decode_path = session_root / "stages/decode/session_quality_fp32.npz"
    prefill = np.load(prefill_path, allow_pickle=False)
    with np.load(decode_path, allow_pickle=False) as archive:
        decode = archive["logits"]
    if (prefill.shape != (1, 32000) or decode.shape != (32, 1, 32000)
            or prefill.dtype != decode.dtype or prefill.dtype != np.float32):
        raise ValueError("captured FP32 references have unexpected shapes")
    expected_keys = {(0, 0, 0)} | {(1, step, 0) for step in range(32)}
    selected = {(row["program"], row["step"], row["output"]): row
                for row in run.get("output_dumps", [])}
    if len(run.get("output_dumps", [])) != 33 or set(selected) != expected_keys:
        raise ValueError("full TinyLlama comparison needs all 33 logit dumps")
    observed = {}
    for key, row in selected.items():
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("logit dump path escapes the run directory")
        path = run_receipt.parent / relative
        if row["bytes"] != 128000 or path.stat().st_size != 128000 or sha(path) != row["sha256"]:
            raise ValueError("logit dump differs from its run receipt")
        observed[key] = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(1, 32000).copy()
    prefill_metrics = metrics(observed[0, 0, 0], prefill)
    decode_metrics = [
        {"step": step, **metrics(observed[1, step, 0], decode[step])}
        for step in range(32)
    ]
    cosines = [row["cosine"] for row in decode_metrics]
    return {
        "schema": "mx_gemmini.tinyllama_session_fp32_comparison.v1",
        "reference_kind": "captured_unquantized_fp32",
        "numerical_qualification": "not_established",
        "run_receipt_sha256": sha(run_receipt),
        "link_receipt_sha256": sha(link_receipt),
        "fast_run_receipt_sha256": sha(fast_run_receipt),
        "fast_link_receipt_sha256": sha(fast_link_receipt),
        "prefill_reference_sha256": sha(prefill_path),
        "decode_reference_sha256": sha(decode_path),
        "output_digests_match_fast_compiler": True,
        "fast_compiler_cycles": fast["total_performance"]["cycles"],
        "dumped_run_cycles": run["total_performance"]["cycles"],
        "prefill_metrics": prefill_metrics,
        "decode_step_metrics": decode_metrics,
        "decode_summary": {
            "cosine_min": min(cosines),
            "cosine_mean": float(np.mean(cosines)),
            "top_token_matches": sum(row["top_token_equal"] for row in decode_metrics),
            "steps": 32,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-receipt", "link-receipt", "session-root",
                 "fast-run-receipt", "fast-link-receipt", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new file")
    report = compare_session(args.run_receipt, args.link_receipt,
                             args.session_root, args.fast_run_receipt,
                             args.fast_link_receipt)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["decode_summary"], sort_keys=True))


if __name__ == "__main__":
    main()
