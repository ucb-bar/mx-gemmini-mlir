"""Rewrite exact VPU sites selected by an npu-exploration model run plan."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

from mx_gemmini_support.npu_vpu_stage_bridge import rewrite_selected_vpu_sites
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _regular(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} needs a regular file")
    return path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True,
                        help="npu model_quantization_plan.json with VPU choices")
    parser.add_argument("--run", type=Path, required=True,
                        help="the exact npu model run recipe JSON")
    parser.add_argument("--profile", type=Path, required=True,
                        help="the exact MX target profile JSON")
    parser.add_argument("--mlir", type=Path, required=True,
                        help="captured source model .mlir or .mlir.gz")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="new directory for rewritten MLIR and receipts")
    args = parser.parse_args()
    try:
        plan_bytes = _regular(args.plan, "model plan")
        run = json.loads(_regular(args.run, "model run recipe"),
                         object_pairs_hook=_unique_pairs)
        _regular(args.profile, "target profile")
        profile = load_profile(args.profile)
        raw = _regular(args.mlir, "captured MLIR")
        source = (gzip.decompress(raw) if args.mlir.name.endswith(".gz") else raw).decode()
        os.environ.setdefault("MERLIN_TARGET_PATH", str(ROOT))
        receipt = rewrite_selected_vpu_sites(
            json.loads(plan_bytes, object_pairs_hook=_unique_pairs), plan_bytes,
            source, profile, run, args.out_dir)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": receipt["status"],
                      "vpu_selected": sum(map(len, receipt["selected_vpu_regions"].values())),
                      "rewritten_mlir_sha256": receipt["rewritten_mlir_sha256"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
