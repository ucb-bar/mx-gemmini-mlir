"""Apply source-bound npu model-run VPU placements to MX's existing lowerer.

This is a rewrite probe. It does not bind the model run's matrix quantization
policy, compile a catalog or ELF, or establish Spike/output parity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .merlin_vpu_add import MXVPUPostOffload
from .model_quantization_admission import validate_model_quantization_plan
from .target_profile import profile_sha256


VPU_CLASSES = ("softmax", "add", "elementwise", "unary", "reduce")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selected_vpu_regions(plan: dict, source: str, profile: dict,
                         *, run_recipe: dict) -> tuple[dict[str, frozenset[str]], dict]:
    """Admit the complete plan, then return only its exact selected VPU sites."""
    admission = validate_model_quantization_plan(
        plan, source, profile, run_recipe=run_recipe)
    if "vpu_sites" not in plan:
        raise ValueError("npu model run has no selected VPU placement policy")
    regions = {name: frozenset(row["region_id"] for row in plan["vpu_sites"]
                               if row["operation_class"] == name and
                               row["placement"] == "vpu")
               for name in VPU_CLASSES}
    if sum(map(len, regions.values())) != admission["vpu_site_status_counts"]["vpu_selected"]:
        raise ValueError("admitted VPU selected-region count changed")
    return regions, admission


def rewrite_selected_vpu_sites(plan: dict, plan_bytes: bytes,
                               source: str, profile: dict, run_recipe: dict,
                               out_dir: Path) -> dict:
    """Write an inspectable raw-source rewrite and verify every selected site."""
    if json.loads(plan_bytes) != plan:
        raise ValueError("model plan bytes differ from the admitted plan object")
    regions, admission = selected_vpu_regions(
        plan, source, profile, run_recipe=run_recipe)
    if out_dir.exists():
        raise ValueError("VPU rewrite output directory must be new")
    out_dir.mkdir(parents=True)
    input_path = out_dir / "source.mlir"
    input_path.write_text(source, encoding="utf-8")
    sidecar_path = out_dir / "rewrite_probe_sidecar.json"
    sidecar_path.write_text(json.dumps({
        "schema": "mx_gemmini.npu_vpu_rewrite_probe_sidecar.v1",
        "source_mlir_sha256": _sha(source.encode()),
        "plan_sha256": _sha(plan_bytes),
    }, sort_keys=True) + "\n")
    rewritten = MXVPUPostOffload(profile, enabled_site_regions=regions)(
        input_path, sidecar_path, out_dir / "rewrites")
    rewrite_receipts = {}
    for name in VPU_CLASSES:
        path = out_dir / "rewrites" / f"mx_vpu_{name}_rewrite.json"
        if not regions[name]:
            if path.exists():
                raise ValueError(f"unselected {name} VPU rewrite unexpectedly ran")
            continue
        if not path.is_file():
            raise ValueError(f"selected {name} VPU rewrite has no receipt")
        raw = path.read_bytes()
        receipt = json.loads(raw)
        observed = [row["region"] for row in receipt["regions"]]
        if len(observed) != len(set(observed)) or set(observed) != regions[name]:
            raise ValueError(f"selected {name} VPU rewrite lost source sites")
        rewrite_receipts[name] = {"sha256": _sha(raw),
                                  "regions": sorted(observed)}
    if not rewritten.is_file():
        raise ValueError("selected VPU rewrite produced no MLIR")
    receipt = {
        "schema": "mx_gemmini.npu_vpu_stage_bridge.v1",
        "status": "source_bound_vpu_rewrite_only_no_model_elf_or_spike_parity",
        "source_mlir_sha256": _sha(source.encode()),
        "profile_sha256": profile_sha256(profile),
        "plan_sha256": _sha(plan_bytes),
        "run_recipe_sha256": plan["run_recipe_sha256"],
        "mx_admission_sha256": _sha((json.dumps(admission, sort_keys=True,
                                                separators=(",", ":")) + "\n").encode()),
        "selected_vpu_regions": {name: sorted(regions[name]) for name in VPU_CLASSES},
        "host_vpu_sites": [row for row in plan["vpu_sites"] if row["placement"] == "host"],
        "rewritten_mlir": str(rewritten.relative_to(out_dir)),
        "rewritten_mlir_sha256": _sha(rewritten.read_bytes()),
        "rewrite_receipts": rewrite_receipts,
        "matrix_quantization_status": "model_run_matrix_choices_not_applied_by_this_rewrite",
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
