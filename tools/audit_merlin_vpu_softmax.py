"""Write a path-free MX VPU softmax boundary summary for captured MLIR stages."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from mx_gemmini_support.merlin_vpu_inventory import audit_softmax_boundaries
from mx_gemmini_support.target_profile import load_profile, profile_sha256


def summarize(models: dict[str, Path], profile: dict) -> dict:
    if not models:
        raise ValueError("VPU audit needs at least one captured MLIR model")
    rows = {}
    for name, path in models.items():
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"VPU audit model {name} must be a regular MLIR file")
        inventory = audit_softmax_boundaries(path.read_text(encoding="utf-8"), profile)
        shapes = Counter((tuple(row["input_shape"]), row["input_dtype"])
                         for row in inventory["regions"])
        rows[name] = {
            "capture_model_sha256": inventory["source_sha256"],
            "softmax_regions": inventory["softmax_regions"],
            "direct_profile_matches_unlowered": inventory["direct_profile_matches_unlowered"],
            "host_owned_regions": inventory["host_owned_regions"],
            "boundary_shapes": [{"shape": list(shape), "dtype": dtype,
                                 "regions": count}
                                for (shape, dtype), count in sorted(shapes.items())],
        }
    direct = sum(row["direct_profile_matches_unlowered"] for row in rows.values())
    total = sum(row["softmax_regions"] for row in rows.values())
    return {
        "schema": "mx_gemmini.model_vpu_softmax_inventory.v1",
        "status": ("unlowered_direct_vpu_softmax_present" if direct else
                   "no_softmax_regions" if not total else
                   "all_model_softmax_host_owned_conversion_required"),
        "profile_sha256": profile_sha256(profile), "models": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--model", action="append", required=True,
                        help="name=/path/to/captured/model.mlir; repeat per model")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    models = {}
    for item in args.model:
        name, separator, path = item.partition("=")
        if not separator or not name.isidentifier() or not path or name in models:
            parser.error(f"invalid or duplicate --model {item!r}")
        models[name] = Path(path)
    if args.out.exists() or args.out.is_symlink():
        parser.error("--out must not already exist")
    result = summarize(models, load_profile(args.profile))
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({"out": str(args.out), "models": len(models),
                      "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
