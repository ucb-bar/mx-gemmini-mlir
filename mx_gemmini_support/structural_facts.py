"""Attach selected RTL structure to Merlin facts without licensing compute.

The elaborated tile grid establishes geometry. It does not by itself establish
the multiply-accumulate datapath, so it stays outside ``facts.arrays``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .build_receipt import verify_build_receipt
from .contract import compile_contract


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def attach_mesh_observation(
    facts_bytes: bytes, audit: dict, receipt_bytes: bytes, contract_bytes: bytes,
    *, sources: dict[str, Path] | None = None,
) -> dict:
    """Bind a source-checked mesh grid to the exact generic FIRRTL fact bundle.

    ``audit`` must be the result of ``verify_build_receipt`` for the supplied
    receipt and software contract. This pure step checks the independent Merlin
    FIRRTL receipt before adding the bounded OOT observation.
    """
    facts = json.loads(facts_bytes)
    contract = compile_contract(contract_bytes)
    inputs = facts.get("inputs") or {}
    body = facts.get("facts") or {}
    source = body.get("source") or {}
    consistency = facts.get("source_consistency") or {}
    production = consistency.get("production") or {}
    receipt = json.loads(receipt_bytes)
    fir_sha = (audit.get("artifacts") or {}).get("firrtl", {}).get("sha256")
    if audit.get("schema") != "mx_gemmini.build_receipt_audit.v1" or audit.get("status") != "observed_consistency":
        raise ValueError("selected RTL build audit is not consistent")
    if audit.get("phase0_admitted") is not False or audit.get("historical_build_provenance_verified") is not False:
        raise ValueError("unexpected build audit qualification state")
    if (audit.get("gemmini_commit") != contract["rtl_commit"] or
        audit.get("mxgen_commit") != contract["mxgen_commit"] or
        receipt.get("artifacts", {}).get("firrtl", {}).get("sha256") != fir_sha):
        raise ValueError("build audit and selected contract or receipt differ")
    if (not isinstance(body, dict) or inputs.get("target") != contract["target"] or
        body.get("target") != contract["target"] or
        consistency.get("status") != "verified" or
        consistency.get("config") != contract["rtl_config_class"] or
        source.get("config") != contract["rtl_config_class"] or
        source.get("fir_sha256") != fir_sha or
        inputs.get("fir_sha256") != fir_sha or
        production.get("firrtl_sha256") != fir_sha):
        raise ValueError("generic RTL facts do not match the selected FIRRTL and configuration")
    if body.get("structural_observations") or body.get("arrays"):
        raise ValueError("generic facts already contain an array or structural observation")
    mesh = audit.get("elaborated_mesh") or {}
    rows, columns, tiles = mesh.get("rows"), mesh.get("columns"), mesh.get("tiles")
    if (type(rows) is not int or type(columns) is not int or type(tiles) is not int or
        rows < 1 or columns < 1 or rows * columns != tiles):
        raise ValueError("build audit has no complete selected mesh grid")
    hierarchy = audit.get("mesh_compute_hierarchy") or {}
    if (hierarchy.get("tiles_with_hierarchy") != tiles or
        type(hierarchy.get("fused_units_per_tile")) is not int or
        hierarchy["fused_units_per_tile"] < 1 or
        not isinstance(hierarchy.get("path_manifest_sha256"), str) or
        len(hierarchy["path_manifest_sha256"]) != 64 or
        hierarchy.get("connected_arithmetic_verified") is not False):
        raise ValueError("build audit has no bounded selected multiplier hierarchy")
    config_features = audit.get("selected_config_features") or {}
    expected_features = {"has_nonlinear_activations", "has_normalizations", "has_max_pool",
                         "enable_lut", "lut_present"}
    if (set(config_features) != expected_features or any(
        not isinstance(row, dict) or type(row.get("value")) is not bool
        or row.get("origin") not in {"default", "standalone"}
        for row in config_features.values()
    )):
        raise ValueError("build audit has no selected built-in feature switches")
    body["structural_observations"] = [{
        "kind": "mesh_tile_grid",
        "name": "mesh",
        "container": "Mesh",
        "element": "Tile",
        "rows": rows,
        "cols": columns,
        "instances": tiles,
        "source": "selected_firrtl",
        "firrtl_sha256": fir_sha,
        "multiplier_hierarchy": hierarchy,
        "selected_config_features": config_features,
        "corroborated": False,
        "compute_engine_established": False,
    }]
    facts["structural_observation_provenance"] = {
        "schema": "mx_gemmini.structural_observation.v1",
        "build_receipt_sha256": _sha(receipt_bytes),
        "software_spec_sha256": _sha(contract_bytes),
        "generic_facts_sha256": _sha(facts_bytes),
        "phase0_admitted": False,
    }
    if sources:
        readers = inputs.setdefault("reader_sources", [])
        for path in sorted(sources.values(), key=str):
            path = path.resolve()
            readers.append({"path": str(path), "sha256": _sha(path.read_bytes())})
    return facts


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach an uncorroborated MX mesh observation to selected Merlin facts")
    parser.add_argument("chipyard_root", type=Path)
    parser.add_argument("build_receipt", type=Path)
    parser.add_argument("software_spec", type=Path)
    parser.add_argument("generic_facts", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt_bytes = args.build_receipt.read_bytes()
    contract_bytes = args.software_spec.read_bytes()
    audit = verify_build_receipt(args.chipyard_root, receipt_bytes, contract_bytes)
    sources = {
        "build_receipt": args.build_receipt,
        "software_spec": args.software_spec,
        "reader": Path(__file__),
        "audit_reader": Path(__file__).with_name("build_receipt.py"),
        "config_reader": Path(__file__).with_name("config_facts.py"),
        "contract_reader": Path(__file__).with_name("contract.py"),
    }
    selected = attach_mesh_observation(
        args.generic_facts.read_bytes(), audit, receipt_bytes, contract_bytes, sources=sources,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(selected, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
