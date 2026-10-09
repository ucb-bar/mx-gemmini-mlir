"""One-command source specialization, MX lowering, RV64 build, and Spike run."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.bind_payload import bind_payload
from mx_gemmini_support.source_gemm import plan_source_gemm, read_source_gemm
from mx_gemmini_support.source_payload import write_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", required=True, type=Path,
                        help="profile-bound model2MLIR handoff")
    parser.add_argument("--driver", required=True, type=Path,
                        help="Radiance source driver and adjacent data header")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--riscv-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--site-id", default="functional:matmul")
    parser.add_argument("--fp6-quantized-specialization", action="store_true",
                        help="project the checked-in FP6 fullout BF16 result onto its C LUT")
    args = parser.parse_args()
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    kernel = read_source_gemm(args.driver)
    plan_source_gemm(kernel, scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
                     profile=profile)
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    bundle = args.out_dir / "bundle"
    manifest = write_bundle(bundle, kernel, site_id=args.site_id,
                            profile_sha256=profile_sha256(profile),
                            fp6_quantized_specialization=args.fp6_quantized_specialization)
    mlir = args.out_dir / "payload_bound.mlir"
    mlir.write_text(bind_payload(args.mlir.read_text(), profile, manifest))
    subprocess.run([sys.executable, "-m", "tools.compile_mx",
                    "--mlir", str(mlir.resolve()), "--bundle", str(bundle.resolve()),
                    "--profile", str(args.profile.resolve()),
                    "--rtl-root", str(args.rtl_root.resolve()),
                    "--riscv-root", str(args.riscv_root.resolve()),
                    "--out-dir", str((args.out_dir / "build").resolve()), "--run-spike"],
                   cwd=Path(__file__).resolve().parents[1], check=True)


if __name__ == "__main__":
    main()
