"""Compare a complete dumped SmolVLA session against its captured FP32 path.

This is an unquantized reference diagnostic. It does not certify MX arithmetic
or define an acceptable action-quality threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape or actual.dtype != np.float32 or expected.dtype != np.float32:
        raise ValueError("SmolVLA action arrays must have equal f32 shapes")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("SmolVLA action comparison contains nonfinite values")
    lhs, rhs = actual.astype(np.float64), expected.astype(np.float64)
    difference = lhs - rhs
    denominator = max(float(np.linalg.norm(lhs) * np.linalg.norm(rhs)), 1e-300)
    reference_rms = float(np.sqrt(np.mean(rhs * rhs)))
    rmse = float(np.sqrt(np.mean(difference * difference)))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "normalized_rmse": rmse / max(reference_rms, 1e-300),
        "max_abs": float(np.max(np.abs(difference))),
        "cosine": float(np.clip(np.sum(lhs * rhs) / denominator, -1.0, 1.0)),
    }


def compare_session(run_receipt: Path, link_receipt: Path, session_root: Path) -> dict:
    run = json.loads(run_receipt.read_text(encoding="utf-8"))
    link = json.loads(link_receipt.read_text(encoding="utf-8"))
    if (run.get("status") != "complete_numeric_qualification_pending"
            or run.get("link_receipt_sha256") != _sha(link_receipt)
            or Path(link["session_root"]).resolve() != session_root.resolve()):
        raise ValueError("SmolVLA run, link, and capture identities differ")
    names = [program["name"] for program in link["programs"]]
    if names != ["prefix_encode", "flow_denoise", "action_decode"] or [
        program["steps"] for program in link["programs"]] != [1, 10, 1]:
        raise ValueError("comparison expects the captured 1+10+1 SmolVLA session")
    reference_path = session_root / "stages/flow_denoise/session_goldens.npz"
    with np.load(reference_path, allow_pickle=False) as archive:
        expected = archive["actions"]
    if expected.shape != (10, 1, 50, 32) or expected.dtype != np.float32:
        raise ValueError("captured SmolVLA flow reference has unexpected shape")

    selected = {(row["program"], row["step"], row["output"]): row
                for row in run.get("output_dumps", [])}
    required = {(1, step, 0) for step in range(10)} | {(2, 0, 0)}
    if set(selected) != required:
        raise ValueError("SmolVLA comparison needs all flow and final action dumps")
    values = {}
    for key, row in selected.items():
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("output dump path escapes the run directory")
        path = run_receipt.parent / relative
        if row["bytes"] != 6400 or path.stat().st_size != 6400 or _sha(path) != row["sha256"]:
            raise ValueError("SmolVLA output dump differs from its run receipt")
        values[key] = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(1, 50, 32).copy()
    if not np.array_equal(values[1, 9, 0], values[2, 0, 0]):
        raise ValueError("action_decode did not preserve the final flow state")
    steps = [{"step": step, **_metrics(values[1, step, 0], expected[step])}
             for step in range(10)]
    return {
        "schema": "mx_gemmini.smolvla_session_fp32_comparison.v1",
        "reference_kind": "captured_unquantized_fp32",
        "numerical_qualification": "not_established",
        "run_receipt_sha256": _sha(run_receipt),
        "link_receipt_sha256": _sha(link_receipt),
        "reference_sha256": _sha(reference_path),
        "final_action_matches_flow_bytes": True,
        "final_action_metrics": steps[-1],
        "flow_step_metrics": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-receipt", type=Path, required=True)
    parser.add_argument("--link-receipt", type=Path, required=True)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new file")
    report = compare_session(args.run_receipt, args.link_receipt, args.session_root)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps(report["final_action_metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
