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
from .execution_policy import eligible_sites, validate_policy
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


def bind_rewrite_to_stage_policy(plan: dict, plan_bytes: bytes, source: str,
                                 profile: dict, run_recipe: dict,
                                 rewrite_dir: Path, out_dir: Path) -> dict:
    """Map an exact npu rewrite to Merlin's original-capture stage policy.

    Merlin builds the matrix catalog before its post-offload VPU pass. The
    rewritten raw capture is therefore evidence of coverage, not the input to
    catalog planning. This adapter emits the existing execution-policy format
    for the original capture and explicitly leaves recipe quantization unbound.
    """
    if json.loads(plan_bytes) != plan:
        raise ValueError("model plan bytes differ from the admitted plan object")
    regions, _ = selected_vpu_regions(plan, source, profile, run_recipe=run_recipe)
    if rewrite_dir.is_symlink() or not rewrite_dir.is_dir():
        raise ValueError("VPU rewrite directory must be a real directory")
    receipt_path = rewrite_dir / "receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("VPU rewrite receipt must be a regular file")
    rewrite = json.loads(receipt_path.read_bytes())
    expected = {name: sorted(regions[name]) for name in VPU_CLASSES}
    if (rewrite.get("schema") != "mx_gemmini.npu_vpu_stage_bridge.v1" or
            rewrite.get("source_mlir_sha256") != _sha(source.encode()) or
            rewrite.get("profile_sha256") != profile_sha256(profile) or
            rewrite.get("plan_sha256") != _sha(plan_bytes) or
            rewrite.get("run_recipe_sha256") != plan["run_recipe_sha256"] or
            rewrite.get("selected_vpu_regions") != expected):
        raise ValueError("VPU rewrite receipt differs from admitted source sites")
    records = rewrite.get("rewrite_receipts")
    if not isinstance(records, dict) or set(records) != {
            name for name in VPU_CLASSES if regions[name]}:
        raise ValueError("VPU rewrite class receipts differ from admitted source sites")
    for name, record in records.items():
        path = rewrite_dir / "rewrites" / f"mx_vpu_{name}_rewrite.json"
        if (path.is_symlink() or not path.is_file() or
                not isinstance(record, dict) or
                record.get("sha256") != _sha(path.read_bytes()) or
                record.get("regions") != expected[name]):
            raise ValueError(f"VPU {name} rewrite receipt differs from selected sites")
        observed = json.loads(path.read_bytes()).get("regions")
        if (not isinstance(observed, list) or
                any(not isinstance(row, dict) or
                    not isinstance(row.get("region"), str) for row in observed) or
                sorted(row.get("region") for row in observed) != expected[name]):
            raise ValueError(f"VPU {name} rewrite operation census changed")
    relative = Path(rewrite.get("rewritten_mlir", ""))
    if (relative.is_absolute() or not relative.parts or
            any(part in {".", ".."} for part in relative.parts)):
        raise ValueError("VPU rewrite MLIR path is not relative to its receipt")
    rewritten = rewrite_dir / relative
    if rewritten.is_symlink() or not rewritten.is_file() or (
            _sha(rewritten.read_bytes()) != rewrite.get("rewritten_mlir_sha256")):
        raise ValueError("VPU rewritten MLIR differs from its receipt")
    matrix, vpu, _ = eligible_sites(source, profile)
    rows = {row["region_id"]: row for row in plan["sites"]}
    if set(rows) != set(matrix):
        raise ValueError("npu matrix sites differ from stage catalog candidates")
    formats = {}
    fp6_groups = {}
    for region, row in rows.items():
        choice = row["requested"]
        if choice["placement"] == "host":
            raise ValueError(f"stage policy cannot represent host matrix site {region}")
        pair = (choice["activation_format"], choice["weight_format"],
                choice["activation_projection"], choice["weight_projection"],
                choice["weight_source"])
        if pair == ("fp8_e4m3", "fp8_e4m3", "direct", "direct", "quantize_float"):
            formats[region] = "mxfp8"
        elif pair == ("fp4_e2m1", "fp4_e2m1", "direct", "direct", "quantize_float"):
            formats[region] = "mxfp4"
        elif pair == ("fp6_e3m2", "fp6_e3m2", "lut", "lut", "quantize_float"):
            formats[region] = "mxfp6"
            fp6_groups[region] = run_recipe["quantizer"]["lut"]["group"]
        else:
            raise ValueError(f"stage policy cannot represent npu matrix format/source at {region}")
    policy = {
        "schema": ("mx_gemmini.execution_policy.v3" if fp6_groups
                   else "mx_gemmini.execution_policy.v1"),
        "source_mlir_sha256": _sha(source.encode()),
        "profile_sha256": profile_sha256(profile),
        "matrix_formats": {region: formats[region] for region in sorted(formats)},
        "vpu_placements": {
            name: {region: ("vpu" if region in regions[name] else "host")
                   for region in sorted(vpu[name])} for name in VPU_CLASSES},
    }
    if fp6_groups:
        policy["matrix_lut_groups"] = dict(sorted(fp6_groups.items()))
        policy["run_recipe_sha256"] = plan["run_recipe_sha256"]
    resolved = validate_policy(policy, source, profile)
    if (resolved["enabled_vpu_regions"] != regions or
            resolved["annotated_fp6_regions"] != frozenset(fp6_groups)):
        raise ValueError("stage policy changed admitted VPU placements")
    if out_dir.exists() or out_dir.is_symlink():
        raise ValueError("stage policy output directory must be new")
    out_dir.mkdir(parents=True)
    policy_bytes = (json.dumps(policy, indent=2, sort_keys=True) + "\n").encode()
    (out_dir / "execution_policy.json").write_bytes(policy_bytes)
    receipt = {
        "schema": "mx_gemmini.npu_stage_policy_bridge.v1",
        "status": ("source_site_fp6_annotation_only_no_merlin_stage_elf_or_spike_parity"
                   if fp6_groups else
                   "source_site_policy_only_no_recipe_quantizer_binding_or_elf_or_spike_parity"),
        "source_mlir_sha256": _sha(source.encode()),
        "rewritten_mlir_sha256": rewrite["rewritten_mlir_sha256"],
        "rewrite_receipt_sha256": _sha(receipt_path.read_bytes()),
        "plan_sha256": _sha(plan_bytes),
        "execution_policy_sha256": _sha(policy_bytes),
        "stage_source": "original_capture_before_post_offload_vpu_rewrite",
        "matrix_sites": len(formats),
        "vpu_sites": {name: len(regions[name]) for name in VPU_CLASSES},
        "fp6_lut_groups": dict(sorted(fp6_groups.items())),
        "quantizer_binding": ("recipe_sha_only_lut_values_not_in_merlin_stage"
                              if fp6_groups else "not_applied_by_merlin_execution_policy"),
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
