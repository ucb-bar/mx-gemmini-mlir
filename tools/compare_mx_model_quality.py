"""Check MX numerical arrays against reference using the same float reference.

The caller must separately establish program, input, and Spike identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


_MARGIN = 1.05
_KL_FLOOR = 1e-6
_ERROR_FLOOR = 2 ** -20


def _finite_triplet(reference: np.ndarray, source: np.ndarray,
                    compiler: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(item, dtype=np.float64)
                   for item in (reference, source, compiler))
    if not arrays[0].size or any(item.shape != arrays[0].shape or
                                 not np.isfinite(item).all() for item in arrays):
        raise ValueError("MX model quality needs equal, nonempty, finite outputs")
    return arrays


def _kl(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    def log_softmax(value: np.ndarray) -> np.ndarray:
        shifted = value - np.max(value, axis=-1, keepdims=True)
        return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))

    log_ref, log_candidate = log_softmax(reference), log_softmax(candidate)
    probabilities = np.exp(log_ref)
    return np.maximum(0.0, np.sum(
        probabilities * (log_ref - log_candidate), axis=-1))


def compare_tinyllama(reference: np.ndarray, source: np.ndarray,
                      compiler: np.ndarray) -> dict:
    """Compare matched token-logit rows, including cached-session rows."""
    reference, source, compiler = _finite_triplet(reference, source, compiler)
    if reference.ndim != 2 or reference.shape[1] < 2:
        raise ValueError("TinyLlama quality expects [token, vocabulary] logits")
    reference_top = np.argmax(reference, axis=1)

    def metrics(candidate: np.ndarray) -> dict:
        kl = _kl(reference, candidate)
        return {"mean_kl_nats": float(np.mean(kl)),
                "worst_kl_nats": float(np.max(kl)),
                "top1_agreement": int(np.count_nonzero(
                    np.argmax(candidate, axis=1) == reference_top)),
                "positions": reference.shape[0]}

    src, got = metrics(source), metrics(compiler)
    checks = {key: got[key] <= _MARGIN * max(src[key], _KL_FLOOR)
              for key in ("mean_kl_nats", "worst_kl_nats")}
    checks["top1_agreement"] = got["top1_agreement"] >= src["top1_agreement"]
    return {"schema": "mx_gemmini.model_quality.v1", "model": "tinyllama",
            "qualification_scope": "numerical_arrays_only",
            "margin": _MARGIN, "source": src, "compiler": got,
            "checks": checks, "passed": all(checks.values())}


def compare_smolvla(reference: np.ndarray, source: np.ndarray,
                    compiler: np.ndarray) -> dict:
    """Gate every action dimension against the independent source."""
    reference, source, compiler = _finite_triplet(reference, source, compiler)
    if reference.shape != (50, 32):
        raise ValueError("SmolVLA quality expects a [50, 32] action")
    scale = np.maximum(np.sqrt(np.mean(reference * reference, axis=0)),
                       _ERROR_FLOOR)

    def metrics(candidate: np.ndarray) -> dict:
        delta = candidate - reference
        return {"rel_rmse_by_dim": (np.sqrt(np.mean(delta * delta, axis=0)) /
                                    scale).tolist(),
                "worst_normalized_error_by_dim": (np.max(np.abs(delta), axis=0) /
                                                  scale).tolist()}

    src, got = metrics(source), metrics(compiler)
    checks = {key: [got[key][i] <= _MARGIN * max(src[key][i], _ERROR_FLOOR)
                    for i in range(reference.shape[1])]
              for key in src}
    return {"schema": "mx_gemmini.model_quality.v1", "model": "smolvla",
            "qualification_scope": "numerical_arrays_only",
            "margin": _MARGIN, "source": src, "compiler": got,
            "checks_by_dimension": checks,
            "passed": all(all(values) for values in checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tinyllama", "smolvla"), required=True)
    parser.add_argument("--float", dest="float_path", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must name a new receipt")
    paths = (args.float_path, args.source, args.compiler)
    if any(path.is_symlink() or not path.is_file() for path in paths):
        parser.error("MX quality inputs must be regular NumPy array files")
    arrays = [np.load(path, allow_pickle=False) for path in paths]
    result = (compare_tinyllama(*arrays) if args.model == "tinyllama" else
              compare_smolvla(*arrays))
    import hashlib

    result["inputs_sha256"] = {
        key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in
        zip(("float", "source", "compiler"), paths, strict=True)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(args.out), "passed": result["passed"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
