"""Catalog mode-class probes separately from named-profile execution evidence.

The all-asymmetric Spike matrices exercise every legal PE mode at each mesh
size. A result from one of those configurations is useful when selecting a
different profile, but it does not qualify that profile's memory system, VPU,
or command schedule. This report preserves that distinction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "profiles/gemmini-mx-cleanup-266c593"
EVIDENCE = ROOT / "docs/evidence"
OUTPUT = EVIDENCE / "mx_profile_qualification_catalog_266c593/index.json"
DIRECT_E4M3 = {
    "activation_format": "fp8_e4m3", "activation_projection": "direct",
    "weight_format": "fp8_e4m3", "weight_projection": "direct", "pe_mode": 8,
}


def _read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def _digest(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _key(cell: dict) -> str:
    return json.dumps(cell, sort_keys=True, separators=(",", ":"))


def _stock_modes() -> tuple[dict[int, dict[str, dict]], dict[int, str]]:
    paths = {
        8: "docs/evidence/nicolas_generated_mesh_dim8_266c593/qualification.json",
        16: "docs/evidence/nicolas_generated_modes_266c593/qualification.json",
        32: "docs/evidence/nicolas_generated_mesh_dim32_266c593/qualification.json",
    }
    result: dict[int, dict[str, dict]] = {}
    for dim in (8, 32):
        index = _read(paths[dim])
        if index["mesh_dim"] != dim or index["legal_mode_count"] != 36:
            raise ValueError(f"DIM{dim} stock mode index changed")
        rows = {}
        for row in index["rows"]:
            key = _key(row["compute"])
            if key in rows:
                raise ValueError(f"duplicate DIM{dim} mode {key}")
            rows[key] = {"status": row["stock_status"], "source_suffix": row["mode"]}
        result[dim] = rows
    matrix_path = "docs/evidence/nicolas_asym_matrix_dim16_all_plus_fp4_266c593/matrix_first.json"
    matrix = _read(matrix_path)
    generated = _read(paths[16])
    if (matrix["mesh_dim"], matrix["passed_modes"],
            generated["distinct_stock_spike_passes"]) != (16, 30, 35):
        raise ValueError("DIM16 stock mode index changed")
    rows = {}
    for row in matrix["rows"]:
        if row["status"] != "passed":
            raise ValueError("DIM16 checked-in source mode failed")
        rows[_key(row["compute"])] = {"status": "passed", "source_suffix": row["source_suffix"]}
    for row in generated["generated_rows"]:
        key = _key(row["compute"])
        if key in rows:
            raise ValueError(f"duplicate DIM16 mode {key}")
        rows[key] = {
            "status": "passed" if row["status"] == "source_golden_matched_on_pinned_spike"
            else "failed", "source_suffix": row["mode"],
        }
    direct = _key(DIRECT_E4M3)
    if direct in rows:
        raise ValueError("DIM16 direct E4M3 mode is duplicated")
    rows[direct] = {"status": "passed", "source_suffix": "radiance_direct_e4m3"}
    result[16] = rows
    for dim, modes in result.items():
        profile_name = ("MxAllAsymGemminiRocketConfig" if dim == 16 else
                        f"MxDim{dim}AllAsymGemminiRocketConfig")
        profile = load_profile(PROFILE_DIR / f"{profile_name}.json")
        if (set(modes) != {_key(cell) for cell in profile["legal_compute"]} or
                len(modes) != 36 or
                sum(row["status"] == "passed" for row in modes.values()) != 35):
            raise ValueError(f"DIM{dim} stock mode coverage is incomplete")
    return result, paths


def build_report() -> dict:
    modes, stock_paths = _stock_modes()
    candidate_path = (
        "docs/evidence/nicolas_spike_weight_lut_candidate_all_modes_266c593/qualification.json")
    candidate = _read(candidate_path)
    if (candidate["candidate_passes"], candidate["stock_passes_unchanged"],
            candidate["stock_failures_repaired_in_candidate"]) != (108, 105, 3):
        raise ValueError("isolated Spike candidate matrix changed")
    candidate_seen: set[tuple[int, str]] = set()
    for row in candidate["rows"]:
        identity = (row["mesh_dim"], row["source_suffix"])
        if identity in candidate_seen:
            raise ValueError(f"duplicate candidate mode: {identity}")
        candidate_seen.add(identity)
        stock = modes[row["mesh_dim"]]
        matches = [item for item in stock.values()
                   if item["source_suffix"] == row["source_suffix"]]
        if row["source_suffix"] == "direct_e4m3" and row["mesh_dim"] == 16:
            matches = [stock[_key(DIRECT_E4M3)]]
        if len(matches) != 1:
            raise ValueError(f"candidate mode has no unique stock cell: {row}")
        expected = ("stock_pass_unchanged" if matches[0]["status"] == "passed"
                    else "stock_failure_repaired")
        if row["comparison"] != expected:
            raise ValueError(f"candidate and stock result differ: {row}")
    for dim in (8, 16, 32):
        expected = {(dim, item["source_suffix"] if item["source_suffix"] !=
                     "radiance_direct_e4m3" else "direct_e4m3")
                    for item in modes[dim].values()}
        if {item for item in candidate_seen if item[0] == dim} != expected:
            raise ValueError(f"candidate mode-class coverage is incomplete at DIM{dim}")

    selected_path = "docs/evidence/radiance_selected_mx_profiles_266c593/index.json"
    selected = _read(selected_path)
    vpu_path = "docs/evidence/radiance_mx_vpu_legal_roster_266c593/index.json"
    vpu = _read(vpu_path)
    dedicated_path = "docs/evidence/nicolas_asym_matrix_dim16_266c593/matrix_first.json"
    dedicated = _read(dedicated_path)
    plain_path = "docs/evidence/radiance_plain_mx_profile_trio_266c593/index.json"
    plain = _read(plain_path)
    vector_paths = [
        f"docs/evidence/nicolas_vpu_{kind}_compiled_266c593/index.json"
        for kind in ("elementwise", "fused", "variants", "ordering")]
    direct_receipts: dict[str, list[dict]] = {}
    for dim in (8, 16, 32):
        name = ("MxAllAsymGemminiRocketConfig" if dim == 16 else
                f"MxDim{dim}AllAsymGemminiRocketConfig")
        direct_receipts[name] = [{
            "kind": "all_mode_stock_spike_matrix", "evidence": stock_paths[dim],
            "passing_modes": 35, "failing_modes": 1,
        }]
    if selected["profile_count"] != 6:
        raise ValueError("selected-profile evidence changed")
    for row in selected["profiles"]:
        named = load_profile(PROFILE_DIR / f"{row['profile_name']}.json")
        if profile_sha256(named) != row["profile_sha256"]:
            raise ValueError(f"selected profile hash changed: {row['profile_name']}")
        direct_receipts.setdefault(row["profile_name"], []).append({
            "kind": "selected_radiance_gemm_spike", "evidence": selected_path,
            "precisions": row["cases"], "cases": len(row["cases"]),
            "compared_bf16_outputs": row["compared_bf16_outputs"],
        })
    if (dedicated["selected_profiles"], dedicated["selected_modes"],
            dedicated["passed_modes"]) != (20, 26, 26):
        raise ValueError("dedicated asymmetric profile matrix changed")
    dedicated_rows: dict[str, list[dict]] = {}
    for row in dedicated["rows"]:
        if row["status"] != "passed" or row["matched_bf16_outputs"] != 4096:
            raise ValueError("dedicated asymmetric profile has a failed mode")
        dedicated_rows.setdefault(row["profile_name"], []).append(row["compute"])
    if len(dedicated_rows) != 20:
        raise ValueError("dedicated asymmetric profile census changed")
    for name, cells in dedicated_rows.items():
        profile = load_profile(PROFILE_DIR / f"{name}.json")
        if {_key(cell) for cell in cells} != {_key(cell) for cell in profile["legal_compute"]}:
            raise ValueError(f"dedicated profile mode coverage changed: {name}")
        direct_receipts.setdefault(name, []).append({
            "kind": "complete_dedicated_asymmetric_spike_matrix",
            "evidence": dedicated_path, "passing_modes": len(cells),
            "compared_bf16_outputs": 4096 * len(cells),
        })
    plain_profile = load_profile(PROFILE_DIR / "MxGemminiRocketConfig.json")
    if (plain["profile_sha256"] != profile_sha256(plain_profile) or
            len(plain["rows"]) != 3):
        raise ValueError("plain MX profile evidence changed")
    direct_receipts.setdefault(plain_profile["name"], []).append({
        "kind": "plain_mx_radiance_gemm_spike", "evidence": plain_path,
        "cases": len(plain["rows"]),
        "compared_bf16_outputs": plain["compared_bf16_outputs"],
    })
    base_vpu = load_profile(PROFILE_DIR / "MxE4M3VpuGemminiRocketConfig.json")
    for path in vector_paths:
        index = _read(path)
        if (index["profile_sha256"] != profile_sha256(base_vpu) or
                not index["rows"] or
                any(row["status"] != "source_vpu_reference_matched_on_pinned_spike"
                    for row in index["rows"])):
            raise ValueError(f"base VPU profile evidence changed: {path}")
        direct_receipts.setdefault(base_vpu["name"], []).append({
            "kind": "compiled_vpu_source_spike", "evidence": path,
            "cases": len(index["rows"]),
        })
    if vpu["profile_name"] != "MxE4M3Fp4VpuGemminiRocketConfig":
        raise ValueError("VPU roster profile changed")
    vpu_profile = load_profile(PROFILE_DIR / f"{vpu['profile_name']}.json")
    if profile_sha256(vpu_profile) != vpu["profile_sha256"]:
        raise ValueError("VPU roster profile hash changed")
    direct_receipts.setdefault(vpu["profile_name"], []).append({
        "kind": "radiance_gemm_roster_spike", "evidence": vpu_path,
        "precisions": vpu["legal_precisions"],
        "cases_per_run": vpu["case_count_per_run"], "runs": 2,
        "issues_vpu_commands": False,
    })
    profiles = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        profile = load_profile(path)
        dim = profile["geometry"]["mesh_rows"]
        if dim != profile["geometry"]["mesh_columns"] or dim not in modes:
            raise ValueError(f"no mode-class matrix for {path.name}")
        cells = {_key(cell) for cell in profile["legal_compute"]}
        if not cells <= modes[dim].keys():
            raise ValueError(f"profile has unprobed mode class: {path.name}")
        named = direct_receipts.get(profile["name"], [])
        profiles.append({
            "name": profile["name"], "profile_sha256": profile_sha256(profile),
            "mesh_dim": dim, "vpu": profile["resources"]["vpu"],
            "legal_mode_count": len(cells),
            "mode_class_stock_spike_passes": sum(
                modes[dim][cell]["status"] == "passed" for cell in cells),
            "mode_class_stock_spike_failures": [json.loads(cell) for cell in sorted(cells)
                                                if modes[dim][cell]["status"] == "failed"],
            "mode_class_candidate_spike_passes": len(cells),
            "named_profile_spike_evidence": named,
            "profile_qualification": profile["qualification"],
        })
    if len(profiles) != 81 or set(direct_receipts) - {row["name"] for row in profiles}:
        raise ValueError("named RTL profile census changed")
    return {
        "schema": "mx_gemmini.profile_qualification_catalog.v1",
        "scope": "selected indexed evidence; mode-class probes do not qualify other named profiles",
        "rtl_revision": candidate["rtl_revision"],
        "sources_sha256": {path: _digest(path) for path in sorted(
            set(stock_paths.values()) | {candidate_path, selected_path, vpu_path,
                                         dedicated_path, plain_path, *vector_paths})},
        "profile_count": len(profiles),
        "named_profiles_with_indexed_spike_evidence": sum(
            bool(row["named_profile_spike_evidence"]) for row in profiles),
        "profiles": profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare with checked-in catalog")
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rendered = json.dumps(build_report(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if args.out.read_text() != rendered:
            raise SystemExit(f"qualification catalog is stale: {args.out}")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(args.out)


if __name__ == "__main__":
    main()
