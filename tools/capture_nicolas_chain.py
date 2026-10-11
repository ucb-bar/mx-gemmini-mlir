"""Capture Nicolas's two MX contractions, including selected rectangular widths.

The PyTorch graph establishes the two contraction sites. Nicolas's checked-in
packed tensors and scales are separate source specializations. The 96-wide
case derives its wire bytes from the 128³ source and has model-derived outputs.
Random PyTorch example inputs do not generate those bytes.
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
    parser.add_argument("--matrix-dim", type=int, choices=(64, 96, 128), default=64)
    parser.add_argument("--first-k", type=int,
                        help="MM1 K dimension; 64 with matrix-dim 96 selects Nicolas's rectangular source")
    parser.add_argument("--second-width", type=int,
                        help="MM2 N dimension; 32 or 64 with matrix-dim 96 selects a rectangular pair")
    parser.add_argument("--output-rows", type=int, choices=tuple(range(16, 129, 16)),
                        help="row prefix of Nicolas's 128³ plain MX source")
    parser.add_argument("--vpu-derived", action="store_true",
                        help="capture the 128-cubed source pair for a derived MX/VPU handoff")
    args = parser.parse_args()
    output_rows = args.output_rows or args.matrix_dim
    first_k = args.first_k or args.matrix_dim
    second_width = args.second_width or args.matrix_dim
    rectangular = (first_k, args.matrix_dim) == (64, 96) and second_width in (32, 64)
    narrow_vpu = (first_k, args.matrix_dim, second_width, output_rows) == (
        64, 64, 32, 64)
    wide_vpu = ((first_k, args.matrix_dim, output_rows) == (64, 64, 64)
                and second_width in (96, 128))
    square_vpu = (args.vpu_derived and
                  (first_k, args.matrix_dim, second_width, output_rows) ==
                  (128, 128, 128, 128))
    if args.vpu_derived and not square_vpu:
        parser.error("derived square VPU capture needs the 128-cubed source")
    if ((first_k, second_width) != (args.matrix_dim, args.matrix_dim) and
            not narrow_vpu and not wide_vpu and
            (not rectangular or output_rows != 64)):
        parser.error("selected connected MX shape has no checked source specialization")
    if args.matrix_dim == 64 and output_rows != 64:
        parser.error("the 64³ VPU source has only 64 rows")
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
    plain64 = args.matrix_dim == 64 and profile["name"] == "MxGemminiRocketConfig"
    if plain64 and (_git(m2m_root) !=
                    "e9ded36eb85abf2d9097ac4dc11457c825853388" or
                    _git(mxq_root) !=
                    "b4af5430bac147f4a16126931cc0177367cc3982"):
        raise ValueError("direct Nicolas 64³ chain needs the pinned latest frontend")
    if profile["geometry"]["mesh_columns"] != 16 or \
            "fp8_e4m3" not in profile["candidate_output_modes"]:
        raise ValueError("selected profile cannot execute Nicolas's FP8 chain")
    if plain64 and not profile["resources"].get("requantizer"):
        raise ValueError("selected plain FP8 profile lacks the requantizer")
    if args.matrix_dim == 64 and not plain64 and (
            not profile["resources"].get("vpu") or
            not profile["resources"].get("spad_requant")):
        raise ValueError("selected profile cannot execute Nicolas's FP8 VPU chain")
    if args.matrix_dim != 64 and not square_vpu and (
            profile["name"] != "MxGemminiRocketConfig" or
            profile["resources"].get("vpu") or
            not profile["resources"].get("requantizer")):
        raise ValueError("selected profile cannot execute Nicolas's plain FP8 chain")
    if square_vpu and (not profile["resources"].get("vpu") or
                       not profile["resources"].get("spad_requant")):
        raise ValueError("selected profile cannot execute derived 128-cubed VPU chain")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    source = software / ("bareMetalC/matmul_tiled_fp8_64x64_chain.c" if plain64 else
                         "bareMetalC/matmul_tiled_fp8_64x96x64.c" if rectangular else
                         "bareMetalC/chain_vpu_spad_requant.c" if args.matrix_dim == 64
                         else "bareMetalC/matmul_tiled_fp8_128x128_chain.c")
    header = software / ("include/matmul_fp8_64x96x64.h" if rectangular else
                         "include/matmul_fp8_64x64_chain.h" if args.matrix_dim == 64
                         else "include/matmul_fp8_128x128_chain.h")
    if not source.is_file() or not header.is_file():
        raise ValueError("Nicolas's checked-in chain source/header is absent")

    class Chain(torch.nn.Module):
        def forward(self, a: torch.Tensor, b1: torch.Tensor,
                    b2: torch.Tensor) -> torch.Tensor:
            return torch.matmul(torch.matmul(a, b1), b2)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        example = (torch.randn((output_rows, first_k), dtype=torch.float32),
                   torch.randn((first_k, args.matrix_dim), dtype=torch.float32),
                   torch.randn((args.matrix_dim, second_width), dtype=torch.float32))
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
                 ("functional:matmul", "quantized", "mxfp8",
                  [output_rows, args.matrix_dim, first_k]),
                 ("functional:matmul_1", "quantized", "mxfp8",
                  [output_rows, second_width, args.matrix_dim])]):
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
        "schema": ("mx_gemmini.nicolas_plain_chain_64_model2mlir_capture.v1"
                   if plain64 else
                   "mx_gemmini.nicolas_rectangular_chain_model2mlir_capture.v1"
                   if rectangular else
                   "mx_gemmini.nicolas_square_128_vpu_model2mlir_capture.v1"
                   if square_vpu else
                   "mx_gemmini.nicolas_derived_wide_vpu_chain_model2mlir_capture.v1"
                   if wide_vpu else
                   "mx_gemmini.nicolas_narrow_vpu_chain_model2mlir_capture.v1"
                   if narrow_vpu else
                   "mx_gemmini.nicolas_chain_model2mlir_capture.v1" if args.matrix_dim == 64
                   else f"mx_gemmini.nicolas_chain_{args.matrix_dim}_model2mlir_capture.v1"),
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
    if args.matrix_dim != 64 or plain64:
        receipt["matrix_dim"] = args.matrix_dim
    if output_rows != args.matrix_dim:
        receipt["output_rows"] = output_rows
        receipt["numerical_scope"] = (
            "row-prefix specialization of Nicolas's checked-in 128³ packed source")
    if args.matrix_dim == 96:
        receipt["numerical_scope"] = (
            "96-wide slice of Nicolas's checked-in 128³ packed inputs; "
            "both outputs require a separate pinned mesh-model reference")
    if rectangular:
        receipt["first_shape_mnk"] = [output_rows, args.matrix_dim, first_k]
        receipt["second_shape_mnk"] = [output_rows, second_width, args.matrix_dim]
        receipt["b2_header_sha256"] = _sha(
            software / "include/matmul_fp8_128x128_chain.h")
        receipt["numerical_scope"] = (
            "MM1 codes/scales have an unchanged Nicolas 64x96x64 source golden; "
            "MM2 uses a checked 128³ B2 wire slice and a pinned mesh-model reference")
    if narrow_vpu:
        receipt["first_shape_mnk"] = [64, 64, 64]
        receipt["second_shape_mnk"] = [64, 32, 64]
        receipt["numerical_scope"] = (
            "MM1 and VPU use Nicolas's checked 64³ source; MM2 uses its first "
            "32 weight columns and the corresponding first output block")
    if wide_vpu:
        receipt["first_shape_mnk"] = [64, 64, 64]
        receipt["second_shape_mnk"] = [64, second_width, 64]
        receipt["numerical_scope"] = (
            "MM1 and VPU use Nicolas's checked 64³ source; the wider MM2 "
            "weight columns are derived from its B2 wire bytes and require "
            "a separately checked pinned mesh-model output reference")
    if square_vpu:
        receipt["first_shape_mnk"] = [128, 128, 128]
        receipt["second_shape_mnk"] = [128, 128, 128]
        receipt["numerical_scope"] = (
            "the 128-cubed operands and original MM1 BF16 are Nicolas source "
            "bytes; VPU x2 and changed MM2 outputs need pinned-model references")
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"captured {len(sites)} MX sites: {bound}")


if __name__ == "__main__":
    main()
