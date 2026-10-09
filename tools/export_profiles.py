"""Export source-bound structural MX profiles from a pinned Gemmini checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mx_gemmini_support.target_profile import export_profiles, profile_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemmini", required=True, type=Path)
    parser.add_argument("--config", help="Gemmini fragment or Chipyard config class; default exports all")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    profiles = export_profiles(args.gemmini)
    if args.config:
        if args.config not in profiles:
            parser.error(f"unknown MX config {args.config}")
        profiles = {args.config: profiles[args.config]}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, profile in profiles.items():
        path = args.out / f"{name}.json"
        if path.exists():
            parser.error(f"refusing to overwrite {path}")
        path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
        print(f"{name} {profile_sha256(profile)} {path}")


if __name__ == "__main__":
    main()
