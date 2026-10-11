"""Report MX candidates and remaining operation families in a captured session.

Requires model2MLIR for bundle integrity and Merlin for the shared MLIR parser.
This is a structural audit, never an executable placement or Spike result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from mx_gemmini_support.merlin_inventory import build_mx_worklist_from_merlin
from mx_gemmini_support.dense_contraction import MAX_UNROLLED_COMMANDS, plan_dense_shape
from mx_gemmini_support.model_precision_plan import build_model_precision_plan
from mx_gemmini_support.target_profile import load_profile, profile_sha256


def summarize_dense_shape_plans(proposal: dict, profile: dict) -> dict:
    """Count source-independent matrix schedules without claiming placement."""
    counts: Counter[str] = Counter()
    failures: list[dict] = []
    max_tiles = max_waves = max_commands = 0
    for site in proposal["sites"]:
        selected = site["selected_format"]
        if selected not in {"mxfp4", "mxfp8"}:
            continue
        batched = "shape_bmnk" in site
        shape = site.get("shape_bmnk") if batched else site.get("shape_mnk")
        if not isinstance(shape, list) or len(shape) != (4 if batched else 3):
            counts["missing_static_shape"] += 1
            continue
        matrix_shape = tuple(shape[1:] if batched else shape)
        try:
            planned = plan_dense_shape(
                profile, matrix_shape, precision="FP4" if selected == "mxfp4" else "FP8")
        except (KeyError, ValueError) as error:
            counts["unplanned_shape"] += 1
            if len(failures) < 8:
                failures.append({"region_id": site["region_id"], "reason": str(error)})
            continue
        if batched:
            counts["rank3_slice_shape_plan"] += 1
        elif site.get("element_type") == "f32":
            counts["rank2_f32_dense_plan"] += 1
        else:
            counts["rank2_plan_needs_source_cast"] += 1
        max_tiles = max(max_tiles, planned.output_tiles)
        max_waves = max(max_waves, planned.k_waves)
        max_commands = max(max_commands, planned.estimated_commands)
        if planned.estimated_commands > MAX_UNROLLED_COMMANDS:
            counts["uses_compact_command_issuer"] += 1
    return {
        "status": "shape_schedule_only_no_executable_placement",
        "counts": dict(sorted(counts.items())),
        "max_output_tiles_per_call": max_tiles,
        "max_k_waves_per_tile": max_waves,
        "max_estimated_commands_per_call": max_commands,
        "unrolled_command_limit": MAX_UNROLLED_COMMANDS,
        "first_failures": failures,
    }


def audit_session(root: Path, profile: dict) -> dict:
    from m2m.capture.bundle_integrity import verify_bundle_integrity
    from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir

    if root.is_symlink() or not root.is_dir():
        raise ValueError("session must be a plain directory")
    root = root.resolve(strict=True)
    verify_bundle_integrity(root)
    contract = yaml.safe_load((root / "session_contract.yaml").read_text())
    programs = contract.get("programs") if isinstance(contract, dict) else None
    if not isinstance(programs, list) or not programs:
        raise ValueError("captured session has no program roster")
    if [row.get("name") for row in programs] != contract.get("stages"):
        raise ValueError("captured session has inconsistent stage order")
    stages = []
    for row in programs:
        relative = Path(str(row.get("bundle", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("stage bundle path escapes the session")
        stage = root / relative
        if stage.is_symlink() or not stage.is_dir():
            raise ValueError("stage bundle is not a plain directory")
        mlir = (stage / "model.mlir").read_text()
        inventory = parse_linalg_mlir(mlir)
        worklist = build_mx_worklist_from_merlin(mlir, profile)
        proposal = build_model_precision_plan(worklist, profile)
        families = Counter((op.get("prov") or {}).get("family") or "unattributed"
                           for op in inventory["ops"])
        stages.append({
            "name": row["name"], "source_mlir_sha256": worklist["source_mlir_sha256"],
            "operation_count": len(inventory["ops"]),
            "operation_families": dict(sorted(families.items())),
            "contraction_regions": worklist["contraction_region_count"],
            "precision_proposal": proposal["selected_format_counts"],
            "dense_shape_plans": summarize_dense_shape_plans(proposal, profile),
            "unlowered_operation_count": len(inventory["ops"]),
        })
    return {
        "schema": "mx_gemmini.session_structural_audit.v1",
        "status": "capture_inventory_only_no_executable_placement",
        "session": str(root), "profile_name": profile["name"],
        "profile_sha256": profile_sha256(profile),
        "full_checkpoint": bool((contract.get("provenance") or {}).get("full_checkpoint")),
        "paper_ready": contract.get("paper_ready") is True,
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    audit = audit_session(args.session, load_profile(args.profile))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"session": audit["session"], "full_checkpoint": audit["full_checkpoint"],
                      "stages": [{"name": row["name"],
                                  "precision_proposal": row["precision_proposal"],
                                  "operation_count": row["operation_count"]}
                                 for row in audit["stages"]]}, sort_keys=True))


if __name__ == "__main__":
    main()
