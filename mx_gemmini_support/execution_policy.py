"""Capture-bound MX matrix precision and VPU placement decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .merlin_inventory import build_mx_worklist_from_merlin
from .model_precision_plan import build_model_precision_plan
from .target_profile import profile_sha256


_CLASSES = ("softmax", "add", "elementwise", "unary", "reduce")


def eligible_sites(source: str, profile: dict) -> tuple[
        dict[str, str], dict[str, set[str]], set[str]]:
    """Return exact executable MX and VPU site IDs in an imported graph."""
    from .merlin_vpu_softmax import selected_softmax_sites
    from .merlin_vpu_add import selected_add_sites
    from .merlin_vpu_elementwise import selected_elementwise_sites
    from .merlin_vpu_unary import selected_unary_sites
    from .merlin_vpu_reduce import selected_row_reduce_sites

    worklist = build_mx_worklist_from_merlin(source, profile)
    proposal = build_model_precision_plan(worklist, profile)
    matrix = {site["region_id"]: site["selected_format"]
              for site in proposal["sites"]
              if site["selected_format"] in {"mxfp4", "mxfp8"}}
    selectors = (selected_softmax_sites, selected_add_sites,
                 selected_elementwise_sites, selected_unary_sites,
                 selected_row_reduce_sites)
    selected = {name: selector(source, profile)
                for name, selector in zip(_CLASSES, selectors, strict=True)}
    vpu = {name: {site["region"] for site in sites}
           for name, sites in selected.items()}
    if len(matrix) != sum(site["selected_format"] in {"mxfp4", "mxfp8"}
                          for site in proposal["sites"]):
        raise ValueError("MX execution policy has duplicate matrix source sites")
    for name, sites in selected.items():
        if len(vpu[name]) != len(sites):
            raise ValueError(f"MX execution policy has duplicate {name} source sites")
    if len(set().union(*vpu.values())) != sum(map(len, vpu.values())):
        raise ValueError("MX VPU source site appears in multiple classes")
    # Merlin folds a one-element add into scalar broadcast during preparation.
    # It remains an identified source site, but has no useful VPU route.
    singleton_adds = {site["region"] for site in selected["add"]
                      if site["elements"] == 1}
    return matrix, vpu, singleton_adds


def make_policy(source: str, profile: dict) -> dict:
    """Start from all FP8 and retain VPU except sensitive float paths."""
    matrix, vpu, singleton_adds = eligible_sites(source, profile)
    return {
        "schema": "mx_gemmini.execution_policy.v1",
        "source_mlir_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "profile_sha256": profile_sha256(profile),
        "matrix_formats": {region: "mxfp8" for region in sorted(matrix)},
        "vpu_placements": {
            name: {region: ("host" if name == "softmax" or
                            (name == "add" and region in singleton_adds) or
                            (name == "elementwise" and "#add" in region)
                            else "vpu") for region in sorted(sites)}
            for name, sites in vpu.items()},
    }


def validate_policy(data: dict, source: str, profile: dict) -> dict:
    """Reject stale, partial, unsupported, or silently ignored decisions."""
    matrix, vpu, _ = eligible_sites(source, profile)
    if (not isinstance(data, dict) or
            set(data) != {"schema", "source_mlir_sha256", "profile_sha256",
                         "matrix_formats", "vpu_placements"} or
            data["schema"] != "mx_gemmini.execution_policy.v1" or
            data["source_mlir_sha256"] != hashlib.sha256(source.encode()).hexdigest() or
            data["profile_sha256"] != profile_sha256(profile)):
        raise ValueError("MX execution policy differs from the captured source or profile")
    formats = data["matrix_formats"]
    if not isinstance(formats, dict) or set(formats) != set(matrix) or any(
            value not in ({"mxfp4", "mxfp8"} if matrix[region] == "mxfp4"
                          else {"mxfp8"}) for region, value in formats.items()):
        raise ValueError("MX execution policy has invalid matrix site formats")
    placements = data["vpu_placements"]
    if (not isinstance(placements, dict) or set(placements) != set(_CLASSES) or
            any(not isinstance(placements[name], dict) or
                set(placements[name]) != sites or
                any(route not in {"host", "vpu"}
                    for route in placements[name].values())
                for name, sites in vpu.items())):
        raise ValueError("MX execution policy has invalid VPU site placements")
    return {
        "precision_overrides": {region: "mxfp8" for region, fmt in formats.items()
                                if matrix[region] == "mxfp4" and fmt == "mxfp8"},
        "enabled_vpu_regions": {name: frozenset(
            region for region, route in placements[name].items() if route == "vpu")
            for name in _CLASSES},
    }


def load_policy(path: Path, source: str, profile: dict) -> tuple[dict, dict]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("MX execution policy needs a regular JSON file")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, validate_policy(data, source, profile)


def reconcile_vpu_region_sets(
        available: dict[str, set[str]], enabled: dict[str, frozenset[str]]
        ) -> tuple[dict[str, frozenset[str]], dict[str, str]]:
    """Follow Merlin's verified scalar-add canonicalization across the boundary."""
    if set(available) != set(_CLASSES) or set(enabled) != set(_CLASSES):
        raise ValueError("MX VPU policy lacks an operation class")
    routed = {name: set(regions) for name, regions in enabled.items()}
    aliases = {}
    for region in sorted(routed["add"] - available["add"]):
        alias = region + "#add0"
        if alias not in available["elementwise"] or alias in routed["elementwise"]:
            raise ValueError(f"MX VPU add {region} has no unique prepared equivalent")
        routed["add"].remove(region)
        routed["elementwise"].add(alias)
        aliases[region] = alias
    if any(not sites <= available[name] for name, sites in routed.items()):
        raise ValueError("MX VPU policy differs from prepared source")
    return {name: frozenset(sites) for name, sites in routed.items()}, aliases


def reconcile_vpu_regions(
        prepared: str, profile: dict,
        enabled: dict[str, frozenset[str]]
        ) -> tuple[dict[str, frozenset[str]], dict[str, str]]:
    from .merlin_vpu_softmax import selected_softmax_sites
    from .merlin_vpu_add import selected_add_sites
    from .merlin_vpu_elementwise import selected_elementwise_sites
    from .merlin_vpu_unary import selected_unary_sites
    from .merlin_vpu_reduce import selected_row_reduce_sites

    selectors = (selected_softmax_sites, selected_add_sites,
                 selected_elementwise_sites, selected_unary_sites,
                 selected_row_reduce_sites)
    available = {name: {site["region"] for site in selector(prepared, profile)}
                 for name, selector in zip(_CLASSES, selectors, strict=True)}
    return reconcile_vpu_region_sets(available, enabled)
