"""Capture Nicolas's 64- or 128-cubed FP4 chain with pinned model2MLIR."""

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
    for name in ("model2mlir-root", "mxq-root", "rtl-root", "profile",
                 "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--matrix-dim", type=int, choices=(64, 128), default=64)
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
    if (profile["name"] != "MxGemminiRocketConfig" or
            "fp4_e2m1" not in profile["candidate_output_modes"]):
        raise ValueError("FP4 capture needs Nicolas's plain MX profile")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    dim = args.matrix_dim
    source = software / f"bareMetalC/matmul_tiled_fp4_{dim}x{dim}_chain.c"
    header = software / f"include/matmul_fp4_{dim}x{dim}_chain.h"
    if not source.is_file() or not header.is_file():
        raise ValueError("Nicolas FP4 source or header is absent")

    class Chain(torch.nn.Module):
        def forward(self, a: torch.Tensor, b1: torch.Tensor,
                    b2: torch.Tensor) -> torch.Tensor:
            return torch.matmul(torch.matmul(a, b1), b2)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        example = tuple(torch.randn((dim, dim), dtype=torch.float32) for _ in range(3))
    contract = root / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
    policy = root / "examples/fp4-policy.yaml"
    result = m2m.convert(
        Chain().eval(), example,
        quantization=ExternalQuantizationConfig("mx_gemmini", contract, policy),
        backend="fx_importer", capture_trace=True)
    if not result.ok or opaque_report(result.mlir_text):
        raise RuntimeError(f"FP4 two-matmul capture failed: {result.diagnostics}")
    sites = result.quantization_manifest["sites"]
    expected = [(f"functional:matmul{'' if i == 0 else '_1'}", "quantized",
                 "mxfp4", [dim, dim, dim]) for i in range(2)]
    if [(site["site_id"], site["status"], site["format"], site["shape"])
            for site in sites] != expected:
        raise ValueError("FP4 source sites differ from captured model2MLIR graph")
    validate_handoff(result, contract.read_bytes(), policy.read_bytes())
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    raw = out / "nicolas_fp4_chain.model2mlir.mlir"
    handoff = out / "nicolas_fp4_chain.handoff.mlir"
    bound = out / "nicolas_fp4_chain.profile_bound.mlir"
    manifest = out / "quantization_manifest.json"
    raw.write_text(result.mlir_text)
    handoff.write_text(render_handoff(result, contract.read_bytes(), policy.read_bytes()))
    bound.write_text(bind_handoff(handoff.read_text(), profile))
    manifest.write_text(json.dumps(result.quantization_manifest, indent=2) + "\n")
    if verify_ir(bound.read_text(), profile)["contracts"] != 2:
        raise ValueError("profile bound FP4 capture lost a contraction site")
    subprocess.run([str(args.mx_opt.resolve()), str(bound), "-o", "/dev/null"], check=True)
    receipt = {
        "schema": "mx_gemmini.nicolas_fp4_resident_model2mlir_capture.v1",
        "status": "two_site_frontend_handoff_only",
        "model2mlir_revision": _git(m2m_root), "mxq_revision": _git(mxq_root),
        "rtl_revision": _git(args.rtl_root), "software_revision": _git(software),
        "source_sha256": _sha(source), "header_sha256": _sha(header),
        "matrix_dim": dim,
        "profile_sha256": profile_sha256(profile),
        "source_mlir_sha256": _sha(raw), "handoff_mlir_sha256": _sha(handoff),
        "bound_mlir_sha256": _sha(bound), "manifest_sha256": _sha(manifest),
        "sites": sites, "opaque_calls": [],
        "numerical_scope": "PyTorch sites only; Nicolas packed bytes and scales bind in the resident compiler gate",
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"captured two FP4 sites: {bound}")


if __name__ == "__main__":
    main()
