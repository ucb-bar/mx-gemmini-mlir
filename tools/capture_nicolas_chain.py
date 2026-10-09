"""Capture Nicolas's two 64-cubed MX contractions through current model2MLIR.

The PyTorch graph establishes the two contraction sites. Nicolas's checked-in
packed tensors and scales are separate source specializations; this command
does not claim that random PyTorch example inputs generate those bytes.
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


def _git(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", required=True, type=Path)
    parser.add_argument("--mxq-root", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--mx-opt", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    root = Path(__file__).resolve().parents[1]
    m2m_root = args.model2mlir_root.resolve()
    mxq_root = args.mxq_root.resolve()
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

    if Path(m2m.__file__).resolve().parents[1] != m2m_root or \
            Path(mxq.__file__).resolve().parents[1] != mxq_root:
        raise ValueError("model2MLIR or MXQuant resolved to another checkout")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if profile["geometry"]["mesh_columns"] != 16 or \
            "fp8_e4m3" not in profile["candidate_output_modes"] or \
            not profile["resources"].get("vpu") or \
            not profile["resources"].get("spad_requant"):
        raise ValueError("selected profile cannot execute Nicolas's FP8 VPU chain")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    source = software / "bareMetalC/chain_vpu_spad_requant.c"
    header = software / "include/matmul_fp8_64x64_chain.h"
    if not source.is_file() or not header.is_file():
        raise ValueError("Nicolas's checked-in chain source/header is absent")

    class Chain(torch.nn.Module):
        def forward(self, a: torch.Tensor, b1: torch.Tensor,
                    b2: torch.Tensor) -> torch.Tensor:
            return torch.matmul(torch.matmul(a, b1), b2)

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
        raise RuntimeError(f"model2MLIR two-matmul capture failed: {result.diagnostics}")
    opaque = opaque_report(result.mlir_text)
    if opaque:
        raise RuntimeError(f"model2MLIR left opaque operations: {opaque}")
    sites = result.quantization_manifest["sites"]
    if (len(sites) != 2 or
            [(site["site_id"], site["status"], site["format"], site["shape"])
             for site in sites] != [
                 ("functional:matmul", "quantized", "mxfp8", [64, 64, 64]),
                 ("functional:matmul_1", "quantized", "mxfp8", [64, 64, 64])]):
        raise RuntimeError(f"two MX FP8 contraction sites were not selected: {sites}")
    contract_bytes, policy_bytes = contract.read_bytes(), policy.read_bytes()
    validate_handoff(result, contract_bytes, policy_bytes)
    args.out_dir.mkdir(parents=True)
    source_mlir = args.out_dir / "nicolas_chain.model2mlir.mlir"
    handoff = args.out_dir / "nicolas_chain.handoff.mlir"
    bound = args.out_dir / "nicolas_chain.profile_bound.mlir"
    manifest = args.out_dir / "quantization_manifest.json"
    source_mlir.write_text(result.mlir_text)
    handoff.write_text(render_handoff(result, contract_bytes, policy_bytes))
    bound.write_text(bind_handoff(handoff.read_text(), profile))
    manifest.write_text(json.dumps(result.quantization_manifest, indent=2) + "\n")
    checked = verify_ir(bound.read_text(), profile)
    if checked["contracts"] != 2:
        raise RuntimeError("profile-bound handoff lost a contraction site")
    subprocess.run([str(args.mx_opt.resolve()), str(bound), "-o", "/dev/null"], check=True)
    receipt = {
        "schema": "mx_gemmini.nicolas_chain_model2mlir_capture.v1",
        "status": "two_site_frontend_handoff_only",
        "model2mlir_revision": _git(m2m_root), "mxq_revision": _git(mxq_root),
        "rtl_revision": _git(args.rtl_root), "software_revision": _git(software),
        "compiler_revision": _git(root),
        "source_sha256": _sha(source), "header_sha256": _sha(header),
        "contract_sha256": _sha(contract), "policy_sha256": _sha(policy),
        "profile_sha256": profile_sha256(profile),
        "source_mlir_sha256": _sha(source_mlir), "handoff_mlir_sha256": _sha(handoff),
        "bound_mlir_sha256": _sha(bound), "manifest_sha256": _sha(manifest),
        "sites": sites, "opaque_calls": opaque,
        "numerical_scope": "source A1/B1/B2 packed bytes and scales are not PyTorch example inputs",
        "physical_scope": "two contraction sites verified; MM1 and chain binding remain separate gates",
    }
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"captured {len(sites)} MX sites: {bound}")


if __name__ == "__main__":
    main()
