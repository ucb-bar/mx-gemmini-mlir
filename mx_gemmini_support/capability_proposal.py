"""Project a reviewable MX capability proposal from selected facts and policy.

This is not a Merlin target contract. In particular, an elaborated multiplier
hierarchy does not license a compute unit in the active provider contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from .contract import compile_contract


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def derive_proposal(facts_bytes: bytes, software_spec_bytes: bytes) -> dict:
    facts = json.loads(facts_bytes)
    spec = yaml.safe_load(software_spec_bytes)
    contract = compile_contract(software_spec_bytes)
    if not isinstance(spec, dict) or contract["status"] != "unreviewed":
        raise ValueError("capability proposal requires an unreviewed software candidate")
    inputs = facts.get("inputs") or {}
    body = facts.get("facts") or {}
    consistency = facts.get("source_consistency") or {}
    provenance = facts.get("structural_observation_provenance") or {}
    fir_sha = inputs.get("fir_sha256")
    observations = body.get("structural_observations") or []
    if (inputs.get("target") != contract["target"] or body.get("target") != contract["target"]
            or not _digest(fir_sha)
            or consistency.get("status") != "verified"
            or consistency.get("config") != contract["rtl_config_class"]
            or (consistency.get("production") or {}).get("firrtl_sha256") != fir_sha
            or (body.get("source") or {}).get("fir_sha256") != fir_sha
            or (body.get("source") or {}).get("config") != contract["rtl_config_class"]
            or provenance.get("schema") != "mx_gemmini.structural_observation.v1"
            or not _digest(provenance.get("build_receipt_sha256"))
            or not _digest(provenance.get("generic_facts_sha256"))
            or provenance.get("software_spec_sha256") != _sha(software_spec_bytes)
            or provenance.get("phase0_admitted") is not False
            or body.get("arrays")
            or len(observations) != 1):
        raise ValueError("software candidate and structural facts are not one selected source")
    mesh = observations[0]
    rows, cols, instances = (mesh.get(key) for key in ("rows", "cols", "instances"))
    hierarchy = mesh.get("multiplier_hierarchy") or {}
    wiring = hierarchy.get("port_wiring") or {}
    if (mesh.get("kind") != "mesh_tile_grid" or mesh.get("source") != "selected_firrtl"
            or mesh.get("firrtl_sha256") != fir_sha
            or mesh.get("compute_engine_established") is not False
            or any(type(value) is not int or value < 1 for value in (rows, cols, instances))
            or rows * cols != instances
            or hierarchy.get("tiles_with_hierarchy") != instances
            or type(hierarchy.get("fused_units_per_tile")) is not int
            or hierarchy["fused_units_per_tile"] < 1
            or not _digest(hierarchy.get("path_manifest_sha256"))
            or wiring.get("tiles_with_witnesses") != instances
            or not _digest(wiring.get("witness_manifest_sha256"))
            or hierarchy.get("connected_arithmetic_verified") is not False):
        raise ValueError("selected structural observation is not an uncorroborated mesh grid")
    features = mesh.get("selected_config_features") or {}
    expected_features = {"has_nonlinear_activations", "has_normalizations", "has_max_pool",
                         "enable_lut", "lut_present"}
    if set(features) != expected_features or any(
        not isinstance(row, dict) or type(row.get("value")) is not bool
        or row.get("origin") not in {"default", "standalone"}
        for row in features.values()
    ):
        raise ValueError("selected RTL configuration features are incomplete")

    formats = []
    host_families = set()
    operations = spec.get("operations") or {}
    if not isinstance(operations, dict):
        raise ValueError("software candidate has no operation declarations")
    for name, row in sorted(operations.items()):
        if not isinstance(row, dict):
            raise ValueError(f"malformed operation {name}")
        families = row.get("families") or [name]
        if not isinstance(families, list) or not all(isinstance(family, str) for family in families):
            raise ValueError(f"{name}: operation families must be strings")
        if row.get("placement") == "host":
            host_families.update(families)
            continue
        if row.get("placement") != "accelerator" or "contraction" not in families:
            continue
        dtypes = row.get("operand_dtypes") or []
        if len(dtypes) != 1 or dtypes[0] not in contract["formats"]:
            raise ValueError(f"{name}: no unique compiled MX operand format")
        if any(not isinstance(row.get(key), list) or not row[key]
               for key in ("ranks", "layouts", "tails", "broadcasting", "aliasing")):
            raise ValueError(f"{name}: accelerator legality declaration is incomplete")
        if not isinstance(row.get("accumulator_dtype"), str) or not row["accumulator_dtype"]:
            raise ValueError(f"{name}: accumulator dtype is absent")
        formats.append({
            "format": dtypes[0],
            "operation": name,
            "accumulator_dtype": row.get("accumulator_dtype"),
            "ranks": row.get("ranks") or [],
            "layouts": row.get("layouts") or [],
            "tails": row["tails"],
            "broadcasting": row["broadcasting"],
            "aliasing": row["aliasing"],
            "shape_bounds": row.get("shape_bounds") or {},
            "site_modes": contract["formats"][dtypes[0]]["site_modes"],
            "numeric_review": (row.get("numerical_contract") or {}).get("status") or "unknown",
            "basis": "authored software spec; source cross-check and execution are separate",
        })
    if {row["format"] for row in formats} != set(contract["formats"]) or len(formats) != len(contract["formats"]):
        raise ValueError("accelerator contraction declarations differ from compiled MX formats")
    return {
        "schema": "mx_gemmini.capability_proposal.v1",
        "status": "candidate_not_admitted",
        "target": contract["target"],
        "source": {
            "gemmini_commit": contract["rtl_commit"],
            "mxgen_commit": contract["mxgen_commit"],
            "config": contract["rtl_config_class"],
            "firrtl_sha256": fir_sha,
            "facts_sha256": _sha(facts_bytes),
            "build_receipt_sha256": provenance["build_receipt_sha256"],
            "generic_facts_sha256": provenance["generic_facts_sha256"],
            "software_spec_sha256": _sha(software_spec_bytes),
        },
        "rtl_geometry": {"rows": rows, "cols": cols, "tiles": instances,
                         "multiplier_hierarchy": hierarchy,
                         "basis": "selected FIRRTL tile grid; not compute corroboration"},
        "rtl_builtin_features": features,
        "authored_accelerator_intent": formats,
        "authored_host_families": sorted(host_families),
        "review_obligations": [
            "connected arithmetic and compute-unit capability",
            "per-format numerical semantics and FP6 codebooks",
            "host operation and transfer semantics",
            "source-build input closure and independent target execution",
        ],
        "merlin_target_contract_selectable": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Project an unreviewed MX capability proposal")
    parser.add_argument("facts", type=Path)
    parser.add_argument("software_spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    proposal = derive_proposal(args.facts.read_bytes(), args.software_spec.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(proposal, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
