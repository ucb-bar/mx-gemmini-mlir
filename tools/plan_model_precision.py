"""Emit a graph- and profile-bound precision proposal for model2MLIR MLIR."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.merlin_inventory import build_mx_worklist_from_merlin
from mx_gemmini_support.model_precision_plan import build_model_precision_plan
from mx_gemmini_support.target_profile import load_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-mlir", type=Path, required=True)
    parser.add_argument("--reader", choices=("local", "merlin"), default="local",
                        help="use Merlin's shared graph inventory for whole-model planning")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.model_mlir.read_bytes()
    mlir = (gzip.decompress(raw) if args.model_mlir.name.endswith(".gz") else raw).decode()
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    reader = build_mx_worklist_from_merlin if args.reader == "merlin" else build_model2mlir_worklist
    plan = build_model_precision_plan(reader(mlir, profile), profile)
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: plan[key] for key in ("profile_name", "profile_has_vpu",
                                           "selected_format_counts", "status")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
