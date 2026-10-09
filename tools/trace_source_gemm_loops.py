"""Trace source MX GEMM loop-FSM commands from a profile-bound MLIR contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_loop_trace import trace_bound_source_gemm_loops
from mx_gemmini_support.target_profile import load_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    result = trace_bound_source_gemm_loops(
        args.mlir.read_text(), read_source_gemm(args.driver), profile=profile,
        source_root=args.source_root, rtl_root=args.rtl_root)
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"traced {result['wave_count']} waves and {len(result['packets'])} MX loop packets")


if __name__ == "__main__":
    main()
