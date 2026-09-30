"""Bind an operator's per-site MX choices to frozen candidate capture inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from examples import iteration_workloads
from mx_gemmini_support.m2m_adapter import derive_site_inventory
from mx_gemmini_support.policy import load_policy
from mx_gemmini_support.rtl_check import check_sources


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_roster_case(roster_root: Path, contract: Path, item: dict):
    """Reconstruct only declared case bytes; refuse a changed export or inventory."""
    name = item["name"]
    if name not in iteration_workloads.CASES:
        raise ValueError(f"unknown iteration case {name!r}")
    artifacts = item["artifacts"]
    if set(artifacts) != {"state.pt", "inputs.pt", "original.pt2", "site-inventory.json"}:
        raise ValueError(f"{name}: incomplete frozen artifacts")
    case_root = roster_root / name
    for filename, expected in artifacts.items():
        path = case_root / filename
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{name}/{filename}: frozen digest differs")
    model, _ = iteration_workloads.make_case(name)
    if type(model).__name__ != item["model_class"]:
        raise ValueError(f"{name}: model class differs")
    model.load_state_dict(torch.load(case_root / "state.pt", map_location="cpu", weights_only=True))
    inputs = torch.load(case_root / "inputs.pt", map_location="cpu", weights_only=True)
    if not isinstance(inputs, tuple) or [list(value.shape) for value in inputs] != item["input_shapes"]:
        raise ValueError(f"{name}: input shapes differ")
    archived = torch.export.load(case_root / "original.pt2")
    torch.testing.assert_close(archived.module()(*inputs), model(*inputs))
    inventory = json.loads((case_root / "site-inventory.json").read_text())
    derived = derive_site_inventory(model, inputs, contract_bytes=contract.read_bytes())
    if inventory != derived:
        raise ValueError(f"{name}: frozen site inventory differs from current export")
    return model, inputs, inventory


def verify_roster(roster_root: Path, contract: Path, *, rtl_root: Path, source_record: Path) -> dict:
    source = roster_root / "roster.json"
    if source.is_symlink() or not source.is_file():
        raise ValueError("frozen roster is missing or symlinked")
    roster = json.loads(source.read_text())
    if (roster.get("schema") != "mx_gemmini.iteration_roster_candidate.v2"
            or roster.get("status") != "candidate_not_admitted"
            or roster.get("source_sha256") != sha256(Path(iteration_workloads.__file__))
            or roster.get("contract_sha256") != sha256(contract)
            or roster.get("rtl_source_record_sha256") != sha256(source_record)
            or roster.get("rtl_source_check") !=
               check_sources(rtl_root, contract.read_bytes(), source_record.read_bytes())):
        raise ValueError("roster source or contract identity differs")
    cases = roster.get("cases")
    if (not isinstance(cases, list) or len(cases) != len(iteration_workloads.CASES)
            or {item.get("name") for item in cases} != set(iteration_workloads.CASES)):
        raise ValueError("roster case membership differs")
    return roster


def check_policy(inventory: dict, raw: bytes) -> dict:
    """Require a graph-bound explicit choice at every derived contraction site."""
    policy = load_policy(raw)
    if policy.default_format != "host" or policy.graph_sha256 != inventory["source_graph_sha256"]:
        raise ValueError("selection needs host default and the exact source graph digest")
    sites = {row["site_id"]: row for row in inventory["sites"]}
    chosen = {f"module:{name}": fmt for name, fmt in policy.module_overrides.items()}
    chosen.update(policy.functional_overrides)
    if set(chosen) != set(sites):
        raise ValueError(f"selection must assign every site exactly: missing={sorted(set(sites) - set(chosen))}, "
                         f"unknown={sorted(set(chosen) - set(sites))}")
    if any(sites[site_id]["kind"] != "linear" for site_id in
           (f"module:{name}" for name in policy.module_overrides)):
        raise ValueError("module override names a non-Linear site")
    if any(sites[site_id]["kind"] != "functional" for site_id in policy.functional_overrides):
        raise ValueError("functional override names a non-functional site")
    for site_id, fmt in chosen.items():
        if fmt == "host":
            continue
        if fmt not in sites[site_id]["eligible_formats"]:
            raise ValueError(f"{site_id}: selected {fmt} is structurally ineligible")
        if fmt == "mxfp6":
            policy.codebooks(site_id)
    if policy.output_chains:
        raise ValueError("iteration selection does not yet admit resident output chains")
    return {"policy_sha256": policy.source_sha256, "source_graph_sha256": policy.graph_sha256,
            "sites": chosen}


def select(roster_root: Path, contract: Path, policy_dir: Path, output: Path,
           *, rtl_root: Path, source_record: Path) -> Path:
    """Freeze choices after independently rederiving the exact roster inventories."""
    roster = verify_roster(roster_root, contract, rtl_root=rtl_root, source_record=source_record)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    result = {
        "schema": "mx_gemmini.iteration_selection.v1",
        "status": "candidate_not_admitted",
        "roster_sha256": sha256(roster_root / "roster.json"),
        "contract_sha256": sha256(contract),
        "rtl_source_record_sha256": sha256(source_record),
        "cases": [],
    }
    for item in roster["cases"]:
        name = item["name"]
        _, _, inventory = load_roster_case(roster_root, contract, item)
        path = policy_dir / f"{name}.yaml"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{name}: selected policy is missing or symlinked")
        choice = check_policy(inventory, path.read_bytes())
        result["cases"].append({"name": name,
                                "inventory_sha256": item["artifacts"]["site-inventory.json"],
                                **choice})
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(select(args.roster_root, args.contract, args.policy_dir, args.output,
                 rtl_root=args.rtl_root, source_record=args.sources))
