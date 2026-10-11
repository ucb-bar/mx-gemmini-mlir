"""Build the experimental source-bound MX catalog for a prepared Merlin graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mx_gemmini_support.merlin_catalog_builder import build_merlin_catalog
from mx_gemmini_support.target_profile import load_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path,
                        help="verify the profile against this pinned RTL checkout")
    parser.add_argument("--riscv-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("output directory already exists")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    manifest, obj = build_merlin_catalog(
        args.prepared, args.out_dir, profile=profile, riscv_root=args.riscv_root)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps({
        "status": data["status"], "manifest": str(manifest), "object": str(obj),
        "selected_operations": data["covered_contractions"],
        "unique_kernels": data["unique_kernels"],
        "shared_workspace_bytes": data["compilation"]["shared_workspace_bytes"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
