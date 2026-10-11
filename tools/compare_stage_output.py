"""Compare one complete MX Spike stage output with an explicit reference.

The reference kind is recorded, not inferred. In particular, a TorchAO/MX
operand fake-quant run is a diagnostic and cannot certify mesh arithmetic.
The comparison covers the first result only; recurrent state results need
their own independently captured arrays and indexed-digest checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REFERENCE_KINDS = (
    "unquantized_pytorch",
    "mx_operand_fake_quant",
    "independent_mx_arithmetic",
)


def compare_output(log: str, reference: np.ndarray, *, kind: str) -> dict:
    """Require a complete, byte-verified first output before reporting errors."""
    if kind not in REFERENCE_KINDS:
        raise ValueError(f"unknown reference kind {kind!r}")
    from merlin.runtime.backends.spike_model import parse_console
    from merlin.runtime.output_digest import verify_output_sha256

    observed = parse_console(log)
    target = np.asarray(observed["outputs"])
    expected = np.asarray(reference)
    if target.size != expected.size or target.dtype != expected.dtype:
        raise ValueError(
            "Spike first output is incomplete or has the wrong dtype: "
            f"observed={target.size} {target.dtype}, reference={expected.size} {expected.dtype}")
    digest = verify_output_sha256(log, target)
    target = target.reshape(expected.shape)
    report: dict = {
        "schema": "mx_gemmini.stage_output_comparison.v1",
        "scope": "first_result_only",
        "reference_kind": kind,
        "target_digest": digest,
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "numerical_qualification": "not_established",
    }
    if expected.dtype == np.dtype("float32"):
        if not np.isfinite(target).all() or not np.isfinite(expected).all():
            raise ValueError("nonfinite values require a separate comparison policy")
        lhs = target.astype(np.float64)
        rhs = expected.astype(np.float64)
        difference = lhs - rhs
        denominator = max(float(np.linalg.norm(lhs) * np.linalg.norm(rhs)), 1e-300)
        report["metrics"] = {
            "mae": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "max_abs": float(np.max(np.abs(difference))),
            "max_rel": float(np.max(np.abs(difference) / np.maximum(np.abs(rhs), 1e-12))),
            "cosine": float(np.sum(lhs * rhs) / denominator),
            "target_argmax": np.argmax(target, axis=-1).tolist(),
            "reference_argmax": np.argmax(expected, axis=-1).tolist(),
            "argmax_equal": bool(np.array_equal(
                np.argmax(target, axis=-1), np.argmax(expected, axis=-1))),
        }
    elif expected.dtype in (np.dtype("int64"), np.dtype("bool")):
        report["metrics"] = {
            "mismatched_elements": int(np.count_nonzero(target != expected)),
            "exact_equal": bool(np.array_equal(target, expected)),
        }
    else:
        raise ValueError("comparison supports f32, i64 and bool first outputs")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path,
                        help="complete first-output .npy reference")
    parser.add_argument("--kind", choices=REFERENCE_KINDS, required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("--out must be a new file")
    for path in (args.log, args.reference):
        if path.is_symlink() or not path.is_file():
            parser.error(f"input must be a regular file: {path}")
    log_bytes = args.log.read_bytes()
    reference = np.load(args.reference, allow_pickle=False)
    try:
        report = compare_output(log_bytes.decode("utf-8", errors="replace"),
                                reference, kind=args.kind)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    report["log_sha256"] = hashlib.sha256(log_bytes).hexdigest()
    report["reference_sha256"] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "metrics": report["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
