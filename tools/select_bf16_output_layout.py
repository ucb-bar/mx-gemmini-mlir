"""Set a source-bound MX BF16 readout's explicit physical memory layout."""

from __future__ import annotations

import argparse
from pathlib import Path

from mx_gemmini_support.bind_payload import select_bf16_output_layout
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlir", "bundle", "profile", "rtl-root", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--layout", choices=("row_major_bf16",
                                             "output_tile_major_bf16"),
                        default="row_major_bf16")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    manifest, _ = load_bundle(args.bundle)
    selected = select_bf16_output_layout(args.mlir.read_text(), profile,
                                         manifest, layout=args.layout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(selected)
    print(f"selected {args.layout}: {args.out}")


if __name__ == "__main__":
    main()
