"""Catalog mode-class probes separately from named-profile execution evidence.

The all-asymmetric Spike matrices exercise every legal PE mode at each mesh
size. A result from one of those configurations is useful when selecting a
different profile, but it does not qualify that profile's memory system, VPU,
or command schedule. This report preserves that distinction.
"""

from __future__ import annotations

import argparse
import gzip
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


def _narrow_vpu_receipt(relative: str, profile: dict) -> dict:
    """Index only the named VPU profile that produced this full-output run."""
    directory = ROOT / relative.rsplit("/", 1)[0]
    index = _read(relative)
    capture = json.loads((directory / "capture/receipt.json").read_text())
    binding = json.loads((directory / "bound/binding_manifest.json").read_text())
    obj = json.loads((directory / "object/object_manifest.json").read_text())
    spike = json.loads((directory / "spike/qualification_manifest.json").read_text())
    expected = {"c1_bf16_values": 4096, "c1_fp8_codes": 4096,
                "c1_e8m0_scales": 128, "c2_fp8_codes": 2048,
                "c2_e8m0_scales": 64}
    digest = profile_sha256(profile)
    if (index["status"] != "source_mx_vpu_and_narrow_mm2_matched_on_pinned_spike" or
            index["compared"] != expected or
            index["first_shape_mnk"] != [64, 64, 64] or
            index["second_shape_mnk"] != [64, 32, 64] or
            index["model2mlir_revision"] != capture["model2mlir_revision"] or
            capture["profile_sha256"] != digest or
            binding["profile_sha256"] != digest or
            obj["profile_sha256"] != digest or
            spike["profile_sha256"] != digest or
            binding["bound_mlir_sha256"] != obj["bound_mlir_sha256"] or
            obj["bound_mlir_sha256"] != spike["bound_mlir_sha256"] or
            obj["shape_mnk"] != [64, 32, 64] or
            spike["object_sha256"] != obj["object_sha256"] or
            spike["spike_exit_code"] != 0 or
            any(obj.get(name) != 0 for name in
                ("allocated_data_section_bytes", "embedded_operand_bytes",
                 "embedded_golden_bytes")) or
            not profile["resources"]["vpu"] or
            not profile["resources"]["spad_requant"] or
            _key(DIRECT_E4M3) not in {_key(cell) for cell in profile["legal_compute"]}):
        raise ValueError(f"narrow VPU qualification differs from selected profile: {relative}")
    if "profile_sha256" in index and index["profile_sha256"] != digest:
        raise ValueError(f"narrow VPU index profile differs: {relative}")
    for name, descriptor in index["files"].items():
        raw = (directory / name).read_bytes()
        data = gzip.decompress(raw) if name.endswith(".gz") else raw
        if (len(data) != descriptor["bytes"] or
                hashlib.sha256(data).hexdigest() != descriptor["sha256"]):
            raise ValueError(f"narrow VPU archive file differs: {relative}/{name}")
    if (hashlib.sha256(gzip.decompress(
            (directory / "object/mx_issue.o.gz").read_bytes())).hexdigest() !=
            obj["object_sha256"]):
        raise ValueError(f"narrow VPU object differs from manifest: {relative}")
    log = (directory / "spike/spike.log").read_text()
    if ("lowered narrow MX/VPU: C1 BF16 0, C1 0 codes 0 scales, C2 0 codes 0 scales" not in log or
            hashlib.sha256(log.encode()).hexdigest() != spike["spike_log_sha256"]):
        raise ValueError(f"narrow VPU Spike output differs: {relative}")
    return {"kind": "connected_mx_vpu_narrow_spike", "evidence": relative,
            "first_shape_mnk": [64, 64, 64],
            "second_shape_mnk": [64, 32, 64],
            "compared": expected,
            "issues_vpu_commands": True}


def _two_tile_vpu_receipt(capture_path: str, compiled_path: str,
                          source_path: str, source_log_path: str,
                          profile: dict) -> dict:
    """Check the full three-site program under this exact named VPU profile."""
    capture, obj, source = (_read(path) for path in
                            (capture_path, compiled_path, source_path))
    base = ROOT / compiled_path.rsplit("/", 1)[0]
    digest = profile_sha256(profile)
    spike = obj.get("spike_qualification", {})
    physical = json.loads((base / "physical_program.json").read_text())
    commands = [item for item in physical["commands"] if item["kind"] == "command"]
    if (capture.get("schema") !=
            "mx_gemmini.nicolas_chain_pipelined_model2mlir_capture.v1" or
            capture.get("status") != "three_site_frontend_handoff_only" or
            capture.get("model2mlir_revision") !=
            "e9ded36eb85abf2d9097ac4dc11457c825853388" or
            capture.get("profile_sha256") != digest or
            obj.get("schema") != "mx_gemmini.full_chain_pipelined_linkable_object.v1" or
            obj.get("status") != "rv64_rocc_two_tile_object_built" or
            obj.get("profile_sha256") != digest or
            obj.get("capture_receipt_sha256") != _digest(capture_path) or
            obj.get("bound_mlir_sha256") != _digest(
                str(base.relative_to(ROOT) / "connected.mlir")) or
            obj.get("preloaded_mlir_sha256") != _digest(
                str(base.relative_to(ROOT) / "preloaded.mlir")) or
            obj.get("physical_program_sha256") != _digest(
                str(base.relative_to(ROOT) / "physical_program.json")) or
            obj.get("issuer_c_sha256") != _digest(
                str(base.relative_to(ROOT) / "mx_issue.c")) or
            obj.get("object_sha256") != _digest(
                str(base.relative_to(ROOT) / "mx_issue.o")) or
            spike.get("elf_sha256") != _digest(
                str(base.relative_to(ROOT) / "mx_program.elf")) or
            spike.get("spike_log_sha256") != _digest(
                str(base.relative_to(ROOT) / "spike.log")) or
            spike.get("status") != "full_three_site_chain_matched_on_pinned_spike" or
            spike.get("spike_exit_code") != 0 or
            (spike.get("compared_c1_bf16_values"),
             spike.get("compared_fp8_codes"),
             spike.get("compared_e8m0_scales")) != (4096, 16384, 512) or
            any(obj.get(name) != 0 for name in
                ("allocated_data_section_bytes", "embedded_operand_bytes",
                 "embedded_golden_bytes")) or
            obj.get("issue_schedule") != "program_order_with_dependency_fences" or
            physical.get("profile_sha256") != digest or
            source.get("profile_sha256") != digest or
            source.get("status") != "four_source_checks_matched_on_pinned_spike" or
            source.get("spike_log_sha256") != _digest(source_log_path) or
            [item.get("schedule") for item in source.get("checks", [])] !=
            ["warmup", "fenced", "program", "pipelined"] or
            len(commands) != obj.get("command_count") or
            sum(item["funct"] == 8 for item in commands) != 3 or
            sum(item["funct"] == 33 for item in commands) != 2 or
            sum(item["funct"] == 34 for item in commands) != 2 or
            sum(item["rs1"].get("buffer") == "b2_weight" for item in commands) != 16 or
            "C1 BF16 0, C1 0 codes 0 scales, C2 0 codes 0 scales" not in
            (base / "spike.log").read_text() or
            not profile["resources"]["vpu"] or
            not profile["resources"]["spad_requant"] or
            _key(DIRECT_E4M3) not in {_key(cell) for cell in profile["legal_compute"]}):
        raise ValueError(f"full two-tile VPU qualification differs from profile: {compiled_path}")
    return {
        "kind": "connected_mx_vpu_two_tile_spike",
        "evidence": compiled_path,
        "capture": capture_path,
        "source": source_path,
        "first_shape_mnk": [64, 64, 64],
        "second_shape_mnk": [64, 64, 64],
        "second_tile_count": 2,
        "compared_c1_bf16_values": 4096,
        "compared_fp8_codes": 16384,
        "compared_e8m0_scales": 512,
        "issues_vpu_commands": True,
        "issues_captured_mm1": True,
        "schedule": "program_order_with_dependency_fences",
    }


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
    wrapper_path = "docs/evidence/nicolas_rocket_wrapper_matrix_266c593/index.json"
    wrappers = _read(wrapper_path)
    requant_path = "docs/evidence/nicolas_requantizer_wrapper_266c593/index.json"
    requant = _read(requant_path)
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
    resident_path = "docs/evidence/nicolas_resident_mm2_128_266c593/index.json"
    resident = _read(resident_path)
    if (resident["profile_sha256"] != profile_sha256(plain_profile) or
            resident["rtl_revision"] != candidate["rtl_revision"] or
            resident["compared_fp8_codes"] != 16384 or
            resident["compared_e8m0_scales"] != 512 or
            resident["scope"] !=
            "source C1 codes/scales preloaded; typed resident MM2 lowered; excludes MM1"):
        raise ValueError("plain MX resident MM2 receipt differs from selected profile")
    direct_receipts[plain_profile["name"]].append({
        "kind": "source_bound_resident_mm2_spike", "evidence": resident_path,
        "compared_fp8_codes": resident["compared_fp8_codes"],
        "compared_e8m0_scales": resident["compared_e8m0_scales"],
        "excludes_mm1": True,
    })
    connected_path = "docs/evidence/nicolas_connected_plain_chain_128_266c593/index.json"
    connected = _read(connected_path)
    if (connected["profile_sha256"] != profile_sha256(plain_profile) or
            connected["rtl_revision"] != candidate["rtl_revision"] or
            connected["compared_c1_fp8_codes"] != 16384 or
            connected["compared_c1_e8m0_scales"] != 512 or
            connected["compared_fp8_codes"] != 16384 or
            connected["compared_e8m0_scales"] != 512 or
            connected["scope"] !=
            "typed MM1 quantized C1 and scales remain resident for typed MM2"):
        raise ValueError("plain MX connected-chain receipt differs from selected profile")
    direct_receipts[plain_profile["name"]].append({
        "kind": "source_bound_connected_mm1_mm2_spike", "evidence": connected_path,
        "compared_c1_fp8_codes": 16384,
        "compared_c1_e8m0_scales": 512,
        "compared_c2_fp8_codes": 16384,
        "compared_c2_e8m0_scales": 512,
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
    if (wrappers["profile_count"], wrappers["compared_bf16_outputs_per_run"],
            len(wrappers["rows"])) != (8, 32768, 8):
        raise ValueError("Rocket wrapper Spike matrix changed")
    for row in wrappers["rows"]:
        name = row["profile_name"]
        profile = load_profile(PROFILE_DIR / f"{name}.json")
        recipe_path = ROOT / "docs/evidence/nicolas_rocket_wrapper_matrix_266c593" / (
            row["slug"] + "/recipe.json.gz")
        recipe_bytes = gzip.decompress(recipe_path.read_bytes())
        recipe = json.loads(recipe_bytes)
        if (profile["chipyard_config"] is None or
                profile_sha256(profile) != row["profile_sha256"] or
                hashlib.sha256(recipe_bytes).hexdigest() != row["artifact_sha256"]["recipe.json"] or
                recipe["profile_sha256"] != row["profile_sha256"] or
                recipe["source_header_sha256"] != row["source_header_sha256"] or
                _key(recipe["compute"]) not in {
                    _key(cell) for cell in profile["legal_compute"]}):
            raise ValueError(f"Rocket wrapper evidence differs from profile: {name}")
        direct_receipts.setdefault(name, []).append({
            "kind": "source_mode_spike_reproduced", "evidence": wrapper_path,
            "source_selection": row["source_selection"],
            "runs": 2, "compared_bf16_outputs_per_run": 4096,
        })
    requant_profile = load_profile(
        PROFILE_DIR / "TestRequantizerLutMxGemminiRocketConfig.json")
    if (requant["profile_name"] != requant_profile["name"] or
            requant["profile_sha256"] != profile_sha256(requant_profile) or
            requant["rtl_revision"] != candidate["rtl_revision"] or
            {row["precision"] for row in requant["rows"]} != {"fp8", "fp4", "fp6"} or
            any(row["profile_sha256"] != requant["profile_sha256"] or
                row["status"] != "nicolas_oracle_matched_on_pinned_spike" or
                row["fp6_quantized_readout_derived_from_fullout"] !=
                (row["precision"] == "fp6") for row in requant["rows"])):
        raise ValueError("Nicolas requantizer wrapper evidence differs from selected profile")
    direct_receipts.setdefault(requant_profile["name"], []).append({
        "kind": "hardware_requantized_output_spike", "evidence": requant_path,
        "output_precisions": [row["precision"] for row in requant["rows"]],
        "cases": len(requant["rows"]),
        "fp6_terminal_readout_derived_from_fullout": True,
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
    narrow_vpu_paths = {
        "MxE4M3Fp4VpuGemminiRocketConfig":
        "docs/evidence/nicolas_narrow_vpu_pair_64x64x64_64x32x64_9cd918c/index.json",
        "MxE4M3VpuGemminiRocketConfig":
        "docs/evidence/nicolas_narrow_vpu_pair_e4m3_only_103acdc/index.json",
    }
    for name, path in narrow_vpu_paths.items():
        profile = load_profile(PROFILE_DIR / f"{name}.json")
        direct_receipts.setdefault(name, []).append(_narrow_vpu_receipt(path, profile))
    two_tile_paths = {
        "MxE4M3Fp4VpuGemminiRocketConfig": (
            "docs/evidence/nicolas_chain_pipelined_266c593/receipt.json",
            "docs/evidence/nicolas_chain_pipelined_full_266c593/object_manifest.json",
            "docs/evidence/nicolas_chain_pipelined_266c593/source_spike_receipt.json",
            "docs/evidence/nicolas_chain_pipelined_266c593/spike.log"),
        "MxE4M3VpuGemminiRocketConfig": (
            "docs/evidence/nicolas_chain_pipelined_e4m3_only_266c593/capture/receipt.json",
            "docs/evidence/nicolas_chain_pipelined_e4m3_only_266c593/compiled/object_manifest.json",
            "docs/evidence/nicolas_chain_pipelined_e4m3_only_266c593/source_spike_receipt.json",
            "docs/evidence/nicolas_chain_pipelined_e4m3_only_266c593/source_spike.log"),
    }
    for name, paths in two_tile_paths.items():
        profile = load_profile(PROFILE_DIR / f"{name}.json")
        direct_receipts.setdefault(name, []).append(
            _two_tile_vpu_receipt(*paths, profile))
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
            "chipyard_wrapper": profile["chipyard_config"] is not None,
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
    wrappers_with_receipts = [row for row in profiles if row["chipyard_wrapper"] and
                              row["named_profile_spike_evidence"]]
    if sum(row["chipyard_wrapper"] for row in profiles) != 40 or len(
            wrappers_with_receipts) != 40:
        raise ValueError("a Chipyard Rocket wrapper lacks direct Spike evidence")
    return {
        "schema": "mx_gemmini.profile_qualification_catalog.v1",
        "scope": "selected indexed evidence; mode-class probes do not qualify other named profiles",
        "rtl_revision": candidate["rtl_revision"],
        "sources_sha256": {path: _digest(path) for path in sorted(
            set(stock_paths.values()) | {candidate_path, selected_path, vpu_path,
                                         dedicated_path, plain_path, resident_path,
                                         connected_path,
                                         wrapper_path,
                                         requant_path,
                                         *narrow_vpu_paths.values(),
                                         *(path for paths in two_tile_paths.values()
                                           for path in paths),
                                         *vector_paths})},
        "profile_count": len(profiles),
        "chipyard_wrapper_count": 40,
        "chipyard_wrappers_with_indexed_spike_evidence": len(wrappers_with_receipts),
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
