"""Compare receipt-bound TinyLlama operand proxies on one captured input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


_IDENTITY = ("capture_model_sha256", "capture_weights_sha256",
             "capture_inputs_sha256", "baseline_logits_sha256",
             "stage_kind", "layers", "capacity", "fp8_attention_calls",
             "fp8_rotary_bmm_calls")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _run(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    receipt_path = path / "result.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "mx_gemmini.tinyllama_operand_fake_quant.v1"
            or receipt.get("status") != "diagnostic_not_mesh_oracle"):
        raise ValueError("TinyLlama operand proxy has an unsupported receipt")
    baseline = np.load(path / "baseline.npy", allow_pickle=False)
    proxy = np.load(path / "proxy.npy", allow_pickle=False)
    if (baseline.shape != (1, 32000) or proxy.shape != baseline.shape
            or baseline.dtype != np.float32 or proxy.dtype != np.float32
            or not np.isfinite(baseline).all() or not np.isfinite(proxy).all()):
        raise ValueError("TinyLlama proxy logits have an unexpected shape or value")
    if (hashlib.sha256(baseline.tobytes()).hexdigest() != receipt["baseline_logits_sha256"]
            or hashlib.sha256(proxy.tobytes()).hexdigest() != receipt["proxy_logits_sha256"]
            or int(baseline.argmax()) != receipt["baseline_top_token"]
            or int(proxy.argmax()) != receipt["proxy_top_token"]):
        raise ValueError("TinyLlama proxy arrays differ from the run receipt")
    return receipt | {"result_sha256": _sha(receipt_path)}, baseline, proxy


def compare(runs: dict[str, Path]) -> dict:
    if len(runs) < 2:
        raise ValueError("compare at least two TinyLlama precision runs")
    loaded = {name: _run(path) for name, path in sorted(runs.items())}
    first = next(iter(loaded.values()))
    reference, baseline = first[0], first[1]
    total_dense = reference["fp4_linear_sites"] + reference["fp8_linear_sites"]
    rows = []
    for name, (receipt, current_baseline, proxy) in loaded.items():
        if (any(receipt[field] != reference[field] for field in _IDENTITY)
                or receipt["fp4_linear_sites"] + receipt["fp8_linear_sites"] != total_dense
                or not np.array_equal(current_baseline, baseline)):
            raise ValueError("TinyLlama precision runs use different captured work")
        actual = proxy.astype(np.float64)
        expected = baseline.astype(np.float64)
        difference = actual - expected
        rows.append({
            "name": name,
            "result_sha256": receipt["result_sha256"],
            "precision_overrides_sha256": receipt["precision_overrides_sha256"],
            "fp4_linear_sites": receipt["fp4_linear_sites"],
            "fp8_linear_sites": receipt["fp8_linear_sites"],
            "top_token": receipt["proxy_top_token"],
            "top_token_matches_reference": receipt["proxy_top_token"] == reference["baseline_top_token"],
            "mae": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference * difference))),
            "max_abs": float(np.max(np.abs(difference))),
            "cosine": float(np.sum(actual * expected) /
                            (np.linalg.norm(actual) * np.linalg.norm(expected))),
        })
    return {
        "schema": "mx_gemmini.tinyllama_precision_comparison.v1",
        "reference_kind": "captured_unquantized_pytorch",
        "numeric_status": "operand_proxy_only_not_mesh_oracle",
        "identity": {field: reference[field] for field in _IDENTITY},
        "reference_top_token": reference["baseline_top_token"],
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="name=/path/to/diagnostic-output; repeat for each policy")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new file")
    runs = {}
    for item in args.run:
        name, separator, path = item.partition("=")
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in runs:
            parser.error("--run needs a distinct name=/path pair")
        runs[name] = Path(path)
    report = compare(runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({"out": str(args.out), "compared_runs": len(runs)}, sort_keys=True))


if __name__ == "__main__":
    main()
