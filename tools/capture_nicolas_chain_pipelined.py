"""Capture the shared-MM1, two-branch PyTorch graph for Nicolas's chain test.

The source performance driver preloads its C1 BF16 tile. This capture includes
the upstream MM1 so a later compiler program can produce that tile, but source
parity of the three issue schedules is a separate qualification gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def validate_original_branch_trace(trace: dict) -> None:
    """Require two scalar branches from one MM1 and one shared B2 input."""
    try:
        nodes = trace["graphs"]["original"]["nodes"]
        targets = [node["target"] for node in nodes]
        edges = [[arg.get("node_id") if isinstance(arg, dict) else arg
                  for arg in node["args"]] for node in nodes]
        ids = [node["id"] for node in nodes]
    except (KeyError, TypeError) as error:
        raise ValueError("model2MLIR original graph trace is incomplete") from error
    if (targets != ["a", "b1", "b2", "aten.matmul.default",
                    "aten.mul.Tensor", "aten.matmul.default",
                    "aten.mul.Tensor", "aten.matmul.default", "output"] or
            edges[3] != ids[:2] or edges[4] != [ids[3], 2.0] or
            edges[5] != [ids[4], ids[2]] or edges[6] != [ids[3], 4.0] or
            edges[7] != [ids[6], ids[2]] or edges[8] != [[
                {"node_id": ids[5], "value_id": f"{ids[5]}:v0"},
                {"node_id": ids[7], "value_id": f"{ids[7]}:v0"}]]):
        raise ValueError("model2MLIR original graph lost the shared two-branch chain")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "rtl-root", "profile", "mx-opt",
                 "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    root = Path(__file__).resolve().parents[1]
    m2m_root, mxq_root = args.model2mlir_root.resolve(), args.mxq_root.resolve()
    sys.path[:0] = [str(root), str(m2m_root), str(mxq_root)]

    import m2m
    import mxq
    import torch
    from m2m.capture.external_quantization import ExternalQuantizationConfig
    from m2m.coverage import opaque_report
    from mx_gemmini_support.bind_profile import bind_handoff
    from mx_gemmini_support.handoff import render_handoff, validate_handoff
    from mx_gemmini_support.target_profile import load_profile, profile_sha256
    from mx_gemmini_support.verify_profile_ir import verify_ir

    if (Path(m2m.__file__).resolve().parents[1] != m2m_root or
            Path(mxq.__file__).resolve().parents[1] != mxq_root):
        raise ValueError("model2MLIR or MXQuant resolved to another checkout")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if (profile["transport"] != "rocket_rocc" or
            profile["geometry"]["mesh_columns"] != 16 or
            not profile["resources"].get("vpu") or
            not profile["resources"].get("spad_requant") or
            "fp8_e4m3" not in profile["candidate_output_modes"]):
        raise ValueError("chain_pipelined needs Nicolas's DIM16 FP8 MX+VPU profile")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    source = software / "bareMetalC/chain_pipelined.c"
    header = software / "include/matmul_fp8_64x64_chain.h"
    if not source.is_file() or not header.is_file():
        raise ValueError("Nicolas chain_pipelined source or header is absent")

    class Chain(torch.nn.Module):
        def forward(self, a: torch.Tensor, b1: torch.Tensor,
                    b2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            c1 = torch.matmul(a, b1)
            return torch.matmul(c1 * 2.0, b2), torch.matmul(c1 * 4.0, b2)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        example = tuple(torch.randn((64, 64), dtype=torch.float32) for _ in range(3))
    contract = root / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
    policy = root / "examples/default-policy.yaml"
    result = m2m.convert(
        Chain().eval(), example,
        quantization=ExternalQuantizationConfig("mx_gemmini", contract, policy),
        backend="fx_importer", capture_trace=True)
    if not result.ok:
        raise RuntimeError(f"model2MLIR three-site capture failed: {result.diagnostics}")
    opaque = opaque_report(result.mlir_text)
    sites = result.quantization_manifest["sites"]
    expected = [(f"functional:matmul{'' if i == 0 else '_' + str(i)}",
                 "quantized", "mxfp8", [64, 64, 64]) for i in range(3)]
    if (opaque or [(s["site_id"], s["status"], s["format"], s["shape"])
                   for s in sites] != expected):
        raise ValueError(f"three FP8 sites were not captured without opaque calls: {sites}, {opaque}")
    validate_original_branch_trace(result.capture_trace)
    contract_bytes, policy_bytes = contract.read_bytes(), policy.read_bytes()
    validate_handoff(result, contract_bytes, policy_bytes)
    args.out_dir.mkdir(parents=True)
    raw = args.out_dir / "chain_pipelined.model2mlir.mlir"
    handoff = args.out_dir / "chain_pipelined.handoff.mlir"
    bound = args.out_dir / "chain_pipelined.profile_bound.mlir"
    manifest = args.out_dir / "quantization_manifest.json"
    trace = args.out_dir / "capture_trace.json"
    original = args.out_dir / "original_graph.json"
    raw.write_text(result.mlir_text)
    handoff.write_text(render_handoff(result, contract_bytes, policy_bytes))
    bound.write_text(bind_handoff(handoff.read_text(), profile))
    manifest.write_text(json.dumps(result.quantization_manifest, indent=2) + "\n")
    trace.write_text(json.dumps(result.capture_trace, indent=2, sort_keys=True) + "\n")
    original.write_text(json.dumps(result.capture_trace["graphs"]["original"],
                                   indent=2, sort_keys=True) + "\n")
    checked = verify_ir(bound.read_text(), profile)
    if checked["contracts"] != 3:
        raise ValueError("profile-bound handoff lost a contraction site")
    subprocess.run([str(args.mx_opt.resolve()), str(bound), "-o", "/dev/null"], check=True)
    receipt = {
        "schema": "mx_gemmini.nicolas_chain_pipelined_model2mlir_capture.v1",
        "status": "three_site_frontend_handoff_only",
        "scope": "shared MM1 followed by two scalar branches and MM2 sites; source driver preloads C1 and tests three schedules",
        "model2mlir_revision": _revision(m2m_root),
        "mxq_revision": _revision(mxq_root),
        "rtl_revision": _revision(args.rtl_root),
        "software_revision": _revision(software),
        "compiler_revision": _revision(root),
        "profile_sha256": profile_sha256(profile),
        "source_sha256": _sha(source), "header_sha256": _sha(header),
        "contract_sha256": _sha(contract), "policy_sha256": _sha(policy),
        "source_mlir_sha256": _sha(raw), "handoff_mlir_sha256": _sha(handoff),
        "bound_mlir_sha256": _sha(bound), "manifest_sha256": _sha(manifest),
        "capture_trace_sha256": _sha(trace),
        "original_graph_sha256": _sha(original),
        "sites": sites, "opaque_calls": opaque,
    }
    (args.out_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"captured shared MM1 and two MM2 sites: {bound}")


if __name__ == "__main__":
    main()
