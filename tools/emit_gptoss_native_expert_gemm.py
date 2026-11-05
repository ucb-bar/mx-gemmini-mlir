"""Emit one unlinked native-MXFP4 gpt-oss expert GEMM stage with live user A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mx_gemmini_support.native_expert_gemm import emit_native_expert_gemm_stage
from mx_gemmini_support.target_profile import load_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--extractor-receipt", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True,
                        help="user model_run.v2 recipe selecting native FP4 B and host_c_v1 A")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--site-id", required=True,
                        help="explicit tile-local site ID; no model2MLIR source region is claimed")
    parser.add_argument("--callback-c", type=Path, required=True)
    parser.add_argument("--abi-header", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = emit_native_expert_gemm_stage(
        tile_dir=args.tile_dir, fixture_dir=args.fixture_dir,
        extractor_receipt_path=args.extractor_receipt,
        run_recipe_path=args.run, profile=load_profile(args.profile),
        site_id=args.site_id, callback_source_path=args.callback_c,
        abi_header_path=args.abi_header, out_dir=args.out_dir)
    print(json.dumps({"status": receipt["status"],
                      "logical_shape_mnk": receipt["logical_shape_mnk"],
                      "physical_shape_mnk": receipt["physical_shape_mnk"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
