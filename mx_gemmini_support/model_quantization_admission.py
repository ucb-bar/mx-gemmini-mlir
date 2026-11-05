"""Structural admission of npu-exploration's source-bound model run receipt.

This reopens the captured MLIR with Merlin and checks the selected MX profile,
site census, wire layouts, and legal PE modes. The quantizer is deliberately
opaque here: this gate neither assumes a numerical rule nor produces operand
bytes, an executable, or a Spike correctness claim. The original run recipe
must be verified by its owner, npu-exploration, against the digest in the plan.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .merlin_inventory import build_mx_worklist_from_merlin
from .target_profile import (SCHEMA as PROFILE_SCHEMA, profile_sha256,
                             require_compute, verify_profile_source)


PLAN_SCHEMA = "npu_exploration.model_quantization_plan.v2"
ADMISSION_SCHEMA = "mx_gemmini.model_quantization_admission.v1"
_HEX = frozenset("0123456789abcdef")
_MODE_KEYS = frozenset({"activation_format", "weight_format", "activation_projection",
                        "weight_projection", "pe_mode"})


def _sha(value: object, label: str, length: int = 64) -> str:
    if (not isinstance(value, str) or len(value) != length or
            any(char not in _HEX for char in value)):
        raise ValueError(f"{label} must be a lowercase {length}-digit SHA")
    return value


def _encoder(fmt: str, projection: str) -> str:
    """Physical operand encoding, independent of how a user made the codes."""
    if fmt == "fp8_e4m3" and projection in {"direct", "lut"}:
        return "fp8_e4m3_quad" if projection == "lut" else fmt
    if (fmt == "fp4_e2m1" and projection == "direct") or (
            fmt in {"fp6_e3m2", "fp6_e2m3", "fp8_e5m2"} and projection == "lut"):
        return fmt
    raise ValueError(f"no MX wire encoding for {fmt}/{projection}")


def _sites(worklist: dict) -> dict[str, tuple[dict, str | None]]:
    shaped: dict[str, tuple[dict, str]] = {}
    for collection, shape_key in (("rank2_matmuls", "shape_mnk"),
                                  ("rank3_batch_matmuls", "shape_bmnk")):
        rows = worklist.get(collection)
        if not isinstance(rows, list):
            raise ValueError(f"Merlin worklist lacks {collection}")
        for site in rows:
            region = site.get("region_id") if isinstance(site, dict) else None
            if not isinstance(region, str) or not region or region in shaped:
                raise ValueError(f"Merlin worklist has ambiguous {collection} region")
            shape = site.get(shape_key)
            if (not isinstance(shape, list) or len(shape) != (3 if shape_key == "shape_mnk" else 4)
                    or any(type(dim) is not int or dim <= 0 for dim in shape)):
                raise ValueError(f"Merlin worklist has malformed {region} shape")
            shaped[region] = (site, shape_key)
    other = {}
    for site in worklist.get("other_contraction_regions", []):
        region = site.get("region_id") if isinstance(site, dict) else None
        if not isinstance(region, str) or not region or region in other:
            raise ValueError("Merlin worklist has ambiguous other contraction region")
        if region in shaped and any(site.get(key) != shaped[region][0].get(key)
                                    for key in ("fqn", "source_op")):
            raise ValueError(f"Merlin worklist has conflicting {region} provenance")
        other[region] = site
    result: dict[str, tuple[dict, str | None]] = {
        region: (site, None) for region, site in other.items() if region not in shaped}
    result.update(shaped)
    if len(result) != worklist.get("contraction_region_count"):
        raise ValueError("Merlin worklist contraction census is incomplete")
    return result


def _padded_shape(shape: list[int], decision: dict, profile: dict) -> list[int]:
    geometry = profile["geometry"]
    m, n, k = shape[-3:]
    a_width = 2 if (decision["activation_format"].startswith(("fp4", "fp6")) or
                    decision["activation_projection"] == "lut") else 1
    b_width = 2 if (decision["weight_format"].startswith(("fp4", "fp6")) or
                    decision["weight_projection"] == "lut") else 1
    tiles = (geometry["mesh_rows"] * a_width,
             geometry["mesh_columns"] * b_width, 32)
    if any(type(tile) is not int or tile <= 0 for tile in tiles):
        raise ValueError("selected MX profile has invalid mesh geometry")
    return [*shape[:-3], *((value + tile - 1) // tile * tile
                            for value, tile in zip((m, n, k), tiles, strict=True))]


def validate_model_quantization_plan(plan: dict, source: str, profile: dict, *,
                                     rtl_root: Path | None = None,
                                     run_recipe: dict | None = None) -> dict:
    """Return a structural receipt, or refuse a stale/changed/illegal model plan."""
    if not isinstance(source, str) or not source:
        raise ValueError("MX model admission needs captured MLIR text")
    if (not isinstance(profile, dict) or profile.get("schema") != PROFILE_SCHEMA or
            profile.get("qualification") != "structural_unqualified"):
        raise ValueError("MX model admission needs a structural target profile v2")
    profile_source = profile.get("source")
    if (not isinstance(profile.get("geometry"), dict) or
            not isinstance(profile.get("resources"), dict) or
            not isinstance(profile.get("legal_compute"), list) or
            not isinstance(profile_source, dict) or
            not isinstance(profile_source.get("sha256"), dict) or
            not profile_source["sha256"]):
        raise ValueError("MX model admission needs a complete source-bound profile")
    for revision in ("gemmini_revision", "mxgen_revision"):
        _sha(profile_source.get(revision), f"profile {revision}", 40)
    for name, digest in profile_source["sha256"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("MX profile source file name is malformed")
        _sha(digest, f"profile source {name}")
    if rtl_root is not None:
        verify_profile_source(profile, Path(rtl_root))
    if (not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or
            plan.get("status") != "planning_only_no_executable"):
        raise ValueError("MX model admission needs a npu model quantization plan v2")
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    profile_sha = profile_sha256(profile)
    if (plan.get("source_mlir_sha256") != source_sha or
            plan.get("profile_sha256") != profile_sha or
            plan.get("profile_name") != profile.get("name") or
            plan.get("profile_source") != profile.get("source")):
        raise ValueError("MX model plan differs from captured source or profile")
    run_id = _sha(plan.get("run_id"), "run ID", 16)
    run_sha = _sha(plan.get("run_recipe_sha256"), "run recipe digest")
    if run_recipe is not None:
        if not isinstance(run_recipe, dict):
            raise ValueError("selected run recipe must be a JSON object")
        run_bytes = (json.dumps(run_recipe, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if hashlib.sha256(run_bytes).hexdigest() != run_sha:
            raise ValueError("selected run recipe differs from model plan digest")
    quantizer = plan.get("quantizer")
    if not isinstance(quantizer, dict) or not quantizer:
        raise ValueError("MX model plan needs a declared user quantizer")
    worklist = build_mx_worklist_from_merlin(source, profile)
    if (worklist.get("source_mlir_sha256") != source_sha or
            worklist.get("profile_sha256") != profile_sha or
            worklist.get("legal_mx_compute_modes") != profile.get("legal_compute")):
        raise ValueError("regenerated Merlin worklist differs from captured source or profile")
    expected = _sites(worklist)
    rows = plan.get("sites")
    if not isinstance(rows, list):
        raise ValueError("MX model plan lacks a site list")
    observed = {}
    for row in rows:
        region = row.get("region_id") if isinstance(row, dict) else None
        if not isinstance(region, str) or region in observed:
            raise ValueError("MX model plan has duplicate or malformed source sites")
        observed[region] = row
    if set(observed) != set(expected):
        raise ValueError(f"MX model plan site census differs: missing={sorted(set(expected)-set(observed))}, "
                         f"extra={sorted(set(observed)-set(expected))}")
    if profile.get("resources", {}).get("scale_block_elements") != 32:
        raise ValueError("MX model plan needs source-bound 32-element scale blocks")
    counts = {"structural_candidate_not_executable": 0,
              "host_selected": 0, "host_unclassified_contraction": 0}
    for region, (site, shape_key) in expected.items():
        row = observed[region]
        if (row.get("fqn") != site.get("fqn") or
                row.get("source_op") != site.get("source_op") or
                type(row.get("explicit_override")) is not bool):
            raise ValueError(f"MX model plan source provenance differs at {region}")
        decision = row.get("requested")
        if not isinstance(decision, dict) or decision.get("placement") not in {"host", "mesh"}:
            raise ValueError(f"MX model plan has no explicit placement at {region}")
        if shape_key is not None and row.get(shape_key) != site[shape_key]:
            raise ValueError(f"MX model plan shape differs at {region}")
        if shape_key is None and any(key in row for key in ("shape_mnk", "shape_bmnk")):
            raise ValueError(f"MX model plan invents a matrix shape at {region}")
        if decision["placement"] == "host" or shape_key is None:
            if shape_key is None and decision["placement"] == "mesh" and row["explicit_override"]:
                raise ValueError(f"MX model plan explicitly selects an unshaped mesh site at {region}")
            status = ("host_selected" if decision["placement"] == "host" else
                      "host_unclassified_contraction")
            if (row.get("status") != status or row.get("hardware_mode") is not None or
                    any(key in row for key in ("operand_encoders", "padding_required",
                                               "padded_shape_mnk", "padded_shape_bmnk"))):
                raise ValueError(f"MX model plan host route differs at {region}")
            counts[status] += 1
            continue
        mode = row.get("hardware_mode")
        if (not isinstance(mode, dict) or set(mode) != _MODE_KEYS or
                type(mode.get("pe_mode")) is not int or
                any(mode[key] != decision.get(key) for key in _MODE_KEYS - {"pe_mode"})):
            raise ValueError(f"MX model plan has an invalid compute mode at {region}")
        require_compute(profile, mode["activation_format"], mode["weight_format"],
                        pe_mode=mode["pe_mode"],
                        activation_projection=mode["activation_projection"],
                        weight_projection=mode["weight_projection"])
        if profile["legal_compute"].count(mode) != 1:
            raise ValueError(f"MX model plan has ambiguous hardware mode at {region}")
        encoders = {
            "activation": _encoder(mode["activation_format"], mode["activation_projection"]),
            "weight": _encoder(mode["weight_format"], mode["weight_projection"]),
        }
        if row.get("operand_encoders") != encoders:
            raise ValueError(f"MX model plan wire encoders differ at {region}")
        if "lut" in (mode["activation_projection"], mode["weight_projection"]):
            config = profile["resources"].get("lut_config")
            entry_bits = max((6 if mode[side + "_format"].startswith("fp6") else 8)
                             for side in ("activation", "weight")
                             if mode[side + "_projection"] == "lut")
            if (not profile["resources"].get("lut") or not isinstance(config, dict) or
                    config.get("address_bits") != 4 or
                    type(config.get("read_data_bits")) is not int or
                    config["read_data_bits"] < entry_bits):
                raise ValueError(f"MX model plan lacks LUT hardware resources at {region}")
        padded = _padded_shape(site[shape_key], decision, profile)
        if (row.get("status") != "structural_candidate_not_executable" or
                row.get("padded_" + shape_key) != padded or
                row.get("padding_required") is not (padded[-3:] != site[shape_key][-3:])):
            raise ValueError(f"MX model plan physical padding differs at {region}")
        counts["structural_candidate_not_executable"] += 1
    canonical_plan = (json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return {
        "schema": ADMISSION_SCHEMA,
        "status": "structural_admission_only_no_executable",
        "source_mlir_sha256": source_sha,
        "profile_sha256": profile_sha,
        "run_id": run_id,
        "run_recipe_sha256": run_sha,
        "run_recipe_digest_status": ("recomputed_from_selected_recipe" if run_recipe is not None
                                     else "declared_not_recomputed"),
        "canonical_plan_sha256": hashlib.sha256(canonical_plan).hexdigest(),
        "site_count": len(expected),
        "site_status_counts": counts,
        "rtl_source_checked_against_checkout": rtl_root is not None,
    }
