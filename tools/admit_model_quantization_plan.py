"""Structurally recheck an npu-exploration model quantization plan in MX.

The result is a source/profile/site admission receipt, not a compiled model,
numeric parity result, or permission to infer a quantization algorithm.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from mx_gemmini_support.model_quantization_admission import validate_model_quantization_plan
from mx_gemmini_support.target_profile import load_profile


def _plain_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True,
                        help="npu-exploration model_quantization_plan.json")
    parser.add_argument("--profile", type=Path, required=True,
                        help="selected MX target profile v2 JSON")
    parser.add_argument("--mlir", type=Path, required=True,
                        help="captured model2MLIR source (.mlir or .mlir.gz)")
    parser.add_argument("--run", type=Path,
                        help="original npu model run JSON, to recheck the declared recipe digest")
    parser.add_argument("--rtl-root", type=Path,
                        help="also verify the profile against this Gemmini/MxGen source closure")
    parser.add_argument("--out", type=Path, help="new path for the admission JSON receipt")
    args = parser.parse_args()
    plan = json.loads(_plain_file(args.plan, "model plan").read_text(encoding="utf-8"))
    profile = load_profile(_plain_file(args.profile, "MX profile"))
    source_path = _plain_file(args.mlir, "captured MLIR")
    source_bytes = source_path.read_bytes()
    source = (gzip.decompress(source_bytes) if source_path.name.endswith(".gz")
              else source_bytes).decode("utf-8")
    run_recipe = (json.loads(_plain_file(args.run, "model run recipe").read_text(encoding="utf-8"))
                  if args.run is not None else None)
    admission = validate_model_quantization_plan(plan, source, profile,
                                                 rtl_root=args.rtl_root,
                                                 run_recipe=run_recipe)
    rendered = json.dumps(admission, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        if args.out.exists() or args.out.is_symlink():
            parser.error("--out must name a new receipt file")
        args.out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
