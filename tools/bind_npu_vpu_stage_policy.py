"""Bind an admitted npu VPU rewrite to Merlin's capture-bound stage policy.

The output is a structural site policy. The current Merlin catalog does not
consume the npu recipe's quantizer or produce a complete model executable.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mx_gemmini_support.npu_vpu_stage_bridge import bind_rewrite_to_stage_policy  # noqa: E402
from mx_gemmini_support.target_profile import load_profile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mlir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rewrite-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    raw_source = args.mlir.read_bytes()
    source = (gzip.decompress(raw_source) if args.mlir.name.endswith(".gz")
              else raw_source).decode("utf-8")
    receipt = bind_rewrite_to_stage_policy(
        json.loads(plan_bytes), plan_bytes, source, load_profile(args.profile),
        json.loads(args.run.read_bytes()), args.rewrite_dir, args.out_dir)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
