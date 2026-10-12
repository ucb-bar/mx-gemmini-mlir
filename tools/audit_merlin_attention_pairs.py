"""Audit source-backed MX attention selection from a Merlin model graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mx_gemmini_support.merlin_inventory import build_mx_worklist_from_merlin
from mx_gemmini_support.model_precision_plan import build_model_precision_plan
from mx_gemmini_support.target_profile import load_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    worklist = build_mx_worklist_from_merlin(args.mlir.read_text(), profile)
    plan = build_model_precision_plan(worklist, profile)
    selected = {site["region_id"]: site for site in plan["sites"]}
    pairs = [{**pair,
              "qk_shape_bmnk": selected[pair["qk_region_id"]].get("shape_bmnk"),
              "pv_shape_bmnk": selected[pair["pv_region_id"]].get("shape_bmnk"),
              "selected_formats": [selected[pair[key]]["selected_format"]
                                   for key in ("qk_region_id", "pv_region_id")]}
             for pair in worklist["attention_pairs"]]
    result = {
        "schema": "mx_gemmini.merlin_ssa_attention_audit.v1",
        "source_mlir_sha256": worklist["source_mlir_sha256"],
        "profile_sha256": worklist["profile_sha256"],
        "ssa_pair_count": len(pairs),
        "selected_attention_regions": sum(site["role"] in {"qk", "pv"}
                                          and site["selected_format"] == "mxfp8"
                                          for site in plan["sites"]),
        "selected_without_module_fqn": sum(site["role"] in {"qk", "pv"}
                                           and site["selected_format"] == "mxfp8"
                                           and not site["fqn"]
                                           for site in plan["sites"]),
        "pairs": pairs,
    }
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
