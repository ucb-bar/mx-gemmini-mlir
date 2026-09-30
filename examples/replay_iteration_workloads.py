"""Replay a frozen MX iteration roster through model2MLIR's OOT adapter."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import torch

from examples import iteration_workloads
from mx_gemmini_support.handoff import render_handoff, validate_handoff


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(roster_root: Path, output_root: Path, contract: Path, policy: Path,
           *, mx_opt: Path | None = None) -> Path:
    import m2m
    from m2m import convert
    from m2m.capture.external_quantization import ExternalQuantizationConfig
    from m2m.capture.trace import capture_frontend_snapshot
    from m2m.coverage import opaque_report

    roster = json.loads((roster_root / "roster.json").read_text())
    if (roster.get("schema") != "mx_gemmini.iteration_roster_candidate.v1" or
            roster.get("status") != "candidate_not_admitted" or
            roster.get("source_sha256") != _sha(Path(iteration_workloads.__file__)) or
            roster.get("contract_sha256") != _sha(contract) or
            roster.get("policy_sha256") != _sha(policy)):
        raise ValueError("roster source, contract, or policy identity differs")
    if (len(roster["cases"]) != len(iteration_workloads.CASES) or
            {item["name"] for item in roster["cases"]} != set(iteration_workloads.CASES) or
            any(set(item["artifacts"]) != {"state.pt", "inputs.pt", "original.pt2"}
                for item in roster["cases"])):
        raise ValueError("roster cases or artifacts differ from the candidate declaration")
    m2m_root = Path(m2m.__file__).resolve().parents[1]
    git = lambda *args: subprocess.check_output(
        ["git", "-C", str(m2m_root), *args], text=True).strip()
    if git("status", "--porcelain"):
        raise ValueError("model2MLIR source checkout has local changes")
    support_root = Path(__file__).resolve().parents[1]
    support_git = lambda *args: subprocess.check_output(
        ["git", "-C", str(support_root), *args], text=True).strip()
    if support_git("status", "--porcelain", "--", "mx_gemmini_support", "examples"):
        raise ValueError("MX support source checkout has local changes")
    output_root.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "mx_gemmini.iteration_capture_candidate.v1",
        "status": "capture_only_not_admitted",
        "roster_sha256": _sha(roster_root / "roster.json"),
        "contract_sha256": _sha(contract),
        "policy_sha256": _sha(policy),
        "replay_source_sha256": _sha(Path(__file__)),
        "support_commit": support_git("rev-parse", "HEAD"),
        "model2mlir_commit": git("rev-parse", "HEAD"),
        "torchao_version": version("torchao"),
        "cases": [],
    }
    if mx_opt is not None:
        report["mx_opt_sha256"] = _sha(mx_opt)
    contract_bytes, policy_bytes = contract.read_bytes(), policy.read_bytes()
    for item in roster["cases"]:
        name = item["name"]
        source_dir = roster_root / name
        for filename, expected in item["artifacts"].items():
            if _sha(source_dir / filename) != expected:
                raise ValueError(f"frozen {name}/{filename} digest differs from roster")
        model, _ = iteration_workloads.make_case(name)
        if type(model).__name__ != item["model_class"]:
            raise ValueError(f"frozen {name} model class differs from roster")
        model.load_state_dict(torch.load(source_dir / "state.pt", map_location="cpu", weights_only=True))
        inputs = torch.load(source_dir / "inputs.pt", map_location="cpu", weights_only=True)
        if [list(value.shape) for value in inputs] != item["input_shapes"]:
            raise ValueError(f"frozen {name} input shapes differ from roster")
        archived = torch.export.load(source_dir / "original.pt2")
        torch.testing.assert_close(archived.module()(*inputs), model(*inputs))
        original = capture_frontend_snapshot(model, inputs)
        if original["status"] != "complete":
            raise ValueError(f"frozen {name} original frontend snapshot is incomplete")
        result = convert(
            model, inputs,
            quantization=ExternalQuantizationConfig("mx_gemmini", contract, policy),
            backend="fx_importer", capture_trace=True,
            original_frontend_snapshot=original,
        )
        if not result.ok:
            raise ValueError(f"frozen {name} capture failed: {result.diagnostics}")
        if result.capture_trace["status"] != "complete":
            raise ValueError(f"frozen {name} capture trace is incomplete")
        opaque = opaque_report(result.mlir_text)
        if opaque:
            raise ValueError(f"frozen {name} has opaque operations: {opaque}")
        validate_handoff(result, contract_bytes, policy_bytes)
        case_dir = output_root / name
        case_dir.mkdir()
        (case_dir / "source.mlir").write_text(result.mlir_text)
        (case_dir / "manifest.json").write_text(
            json.dumps(result.quantization_manifest, indent=2, sort_keys=True) + "\n")
        (case_dir / "handoff.mlir").write_text(
            render_handoff(result, contract_bytes, policy_bytes))
        if mx_opt is not None:
            subprocess.run([str(mx_opt.resolve()), str(case_dir / "handoff.mlir"),
                            "-o", "/dev/null"], check=True)
        report["cases"].append({
            "name": name,
            "source_graph_sha256": original["sha256"],
            "capture_trace_status": "complete",
            "opaque_operations": 0,
            "dialect_verifier": "passed" if mx_opt is not None else "not_run",
            "sites": len(result.quantization_manifest["sites"]),
            "quantized_sites": sum(site["status"] == "quantized"
                                   for site in result.quantization_manifest["sites"]),
            "outputs": {path.name: _sha(path) for path in (
                case_dir / "source.mlir", case_dir / "manifest.json", case_dir / "handoff.mlir")},
        })
    receipt = output_root / "capture-receipt.json"
    receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--mx-opt", type=Path)
    args = parser.parse_args()
    print(replay(args.roster_root, args.output_root, args.contract, args.policy,
                 mx_opt=args.mx_opt))
