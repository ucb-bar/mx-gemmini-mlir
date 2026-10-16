"""Generate an editable, capture-bound MX precision and VPU placement policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mx_gemmini_support.execution_policy import make_policy
from mx_gemmini_support.target_profile import load_profile
from tools.build_merlin_stage import _capture_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must name a new file")
    source = _capture_model(args.stage.resolve(strict=True)).read_text(encoding="utf-8")
    profile = load_profile(args.profile.resolve(strict=True))
    policy = make_policy(source, profile)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    routes = policy["vpu_placements"]
    print(json.dumps({"out": str(args.out),
                      "matrix_sites": len(policy["matrix_formats"]),
                      "eligible_vpu_sites": sum(len(sites) for sites in
                                                routes.values()),
                      "enabled_vpu_sites": sum(
                          route == "vpu" for sites in routes.values()
                          for route in sites.values())},
                     sort_keys=True))


if __name__ == "__main__":
    main()
