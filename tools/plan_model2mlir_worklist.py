"""Emit an MX contraction worklist from a complete model2MLIR frontend graph."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.target_profile import load_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    raw = args.mlir.read_bytes()
    text = (gzip.decompress(raw) if args.mlir.name.endswith(".mlir.gz") else raw).decode()
    worklist = build_model2mlir_worklist(text, load_profile(args.profile))
    args.out.write_text(json.dumps(worklist, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
