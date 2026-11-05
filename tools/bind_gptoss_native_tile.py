#!/usr/bin/env python3
"""Admit a range-extracted gpt-oss expert tile and explicit BF16 A operand."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mx_gemmini_support.gptoss_native_tile import bind_native_expert_tile
from mx_gemmini_support.target_profile import load_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--extractor-receipt", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True,
                        help="full pinned checkpoint revision from the model recipe")
    parser.add_argument("--activation-bf16", type=Path, required=True,
                        help="explicit row-major BF16 [M, selected K] operand")
    parser.add_argument("--activation-m", type=int, required=True)
    parser.add_argument("--projection", choices=("gate_up", "down"), required=True)
    parser.add_argument("--expert-index", type=int, required=True)
    parser.add_argument("--output-start", type=int, required=True)
    parser.add_argument("--output-count", type=int, required=True)
    parser.add_argument("--block-start", type=int, required=True)
    parser.add_argument("--block-count", type=int, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    receipt_bytes = args.extractor_receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    profile = load_profile(args.profile) if args.profile else None
    activation_bf16 = args.activation_bf16.read_bytes()
    codes, scales, manifest = bind_native_expert_tile(
        receipt, receipt_bytes, args.fixture_dir,
        projection=args.projection, expert_index=args.expert_index,
        output_start=args.output_start, output_count=args.output_count,
        block_start=args.block_start, block_count=args.block_count,
        activation_bf16=activation_bf16, activation_m=args.activation_m,
        expected_revision=args.expected_revision, profile=profile)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "b_codes.bin").write_bytes(codes)
    (args.output_dir / "b_scales.bin").write_bytes(scales)
    (args.output_dir / "a_bf16.bin").write_bytes(activation_bf16)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir),
                      "status": manifest["status"],
                      "execution_refusal": manifest["execution_refusal"]}, sort_keys=True))


if __name__ == "__main__":
    main()
