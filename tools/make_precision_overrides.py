"""Create a capture- and profile-bound FP8 dense-site experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.merlin_inventory import build_mx_worklist_from_merlin
from mx_gemmini_support.model_precision_plan import build_model_precision_plan
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.build_merlin_stage import _capture_model


def make_manifest(source: Path, profile: dict, regions: list[str]) -> dict:
    if not regions or len(regions) != len(set(regions)):
        raise ValueError("select at least one distinct FP4 dense source region")
    raw = source.read_bytes()
    worklist = build_mx_worklist_from_merlin(raw.decode("utf-8"), profile)
    overrides = {region: "mxfp8" for region in regions}
    build_model_precision_plan(worklist, profile, precision_overrides=overrides)
    return {
        "schema": "mx_gemmini.precision_overrides.v1",
        "source_mlir_sha256": hashlib.sha256(raw).hexdigest(),
        "profile_sha256": profile_sha256(profile),
        "regions": dict(sorted(overrides.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--promote-region", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new file")
    source = _capture_model(args.stage.resolve(strict=True))
    profile = load_profile(args.profile.resolve(strict=True))
    manifest = make_manifest(source, profile, args.promote_region)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({"out": str(args.out),
                      "promoted_regions": len(manifest["regions"])}, sort_keys=True))


if __name__ == "__main__":
    main()
