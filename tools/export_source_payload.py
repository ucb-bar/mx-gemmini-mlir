"""Export a source MX GEMM's packed data as a checked compiler payload bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from mx_gemmini_support.source_gemm import plan_source_gemm, read_source_gemm
from mx_gemmini_support.source_payload import manifest_sha256, write_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--site-id", default="functional:matmul")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    kernel = read_source_gemm(args.driver)
    plan_source_gemm(kernel, scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
                     profile=profile)
    manifest = write_bundle(args.out_dir, kernel, site_id=args.site_id,
                            profile_sha256=profile_sha256(profile))
    print(f"exported {kernel.datatype} {kernel.shape} source payload "
          f"{manifest_sha256(manifest)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
