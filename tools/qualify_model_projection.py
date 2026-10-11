"""Compile one full-K model2MLIR matmul stripe to a data-free MX object and Spike ELF.

The full model graph names the site; supplied .npy files provide its actual
runtime operands. The result is an executed projection stripe, not a compiled
whole-model inference session.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.bind_payload import bind_payload, select_bf16_output_layout
from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.model_projection import (portable_projection_shape,
                                                 write_model_projection_bundle)
from mx_gemmini_support.target_profile import load_profile, profile_sha256, require_compute
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_mx import _run, _sha, _source_closure


ROOT = Path(__file__).resolve().parents[1]
MODEL2MLIR_SOURCE_CLOSURE = "6eb6648cf2eebd72cd481dff084b0e154e034fbe50b7bce31935091b1a237e39"
MXQ_REVISION = "b4af5430bac147f4a16126931cc0177367cc3982"


def _revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full-model-mlir", "activation-npy", "weight-npy", "model2mlir-root",
                 "mxq-root", "reference-root", "profile", "rtl-root", "riscv-root",
                 "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--site-region", required=True)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-count", type=int, required=True)
    parser.add_argument("--column-start", type=int, default=0)
    parser.add_argument("--column-count", type=int, default=32)
    parser.add_argument("--tile-k", type=int, default=128)
    parser.add_argument("--mx-opt", type=Path)
    parser.add_argument("--run-spike", action="store_true")
    args = parser.parse_args()
    for name in ("full_model_mlir", "activation_npy", "weight_npy", "model2mlir_root",
                 "mxq_root", "reference_root", "profile", "rtl_root", "riscv_root",
                 "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    if (_source_closure(args.model2mlir_root,
                        list((args.model2mlir_root / "m2m").rglob("*.py"))) !=
            MODEL2MLIR_SOURCE_CLOSURE or _revision(args.mxq_root) != MXQ_REVISION):
        raise ValueError("model2MLIR or MXQuant source differs from pinned frontend")
    sys.path[:0] = [str(args.model2mlir_root), str(args.mxq_root)]
    import m2m
    import mxq
    import numpy as np
    import torch

    if (Path(m2m.__file__).resolve().parents[1] != args.model2mlir_root or
            Path(mxq.__file__).resolve().parents[1] != args.mxq_root):
        raise ValueError("selected frontend imports differ from pinned source trees")
    raw = args.full_model_mlir.read_bytes()
    full_model = (gzip.decompress(raw) if args.full_model_mlir.name.endswith(".gz")
                  else raw).decode()
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    worklist = build_model2mlir_worklist(full_model, profile)
    matches = [site for site in worklist["rank2_matmuls"]
               if site["region_id"] == args.site_region]
    if len(matches) != 1:
        raise ValueError("full model has no unique rank-two matmul at the selected region")
    site = matches[0]
    activation = np.load(args.activation_npy, allow_pickle=False)
    weight = np.load(args.weight_npy, allow_pickle=False)
    args.out_dir.mkdir(parents=True)
    bundle_dir = args.out_dir / "bundle"
    manifest = write_model_projection_bundle(
        bundle_dir, worklist=worklist, site=site,
        activation=activation, weight=weight,
        row_start=args.row_start, row_count=args.row_count,
        column_start=args.column_start, column_count=args.column_count,
        tile_k=args.tile_k, profile=profile, reference_root=args.reference_root)
    m, n, k = manifest["shape_mnk"]
    a = torch.zeros((m, k), dtype=torch.float32)
    a[:args.row_count] = torch.from_numpy(
        activation[args.row_start:args.row_start + args.row_count])
    b = torch.from_numpy(weight[:, args.column_start:
                                args.column_start + args.column_count].copy())

    class Matmul(torch.nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.matmul(lhs, rhs)

    capture = m2m.convert(Matmul().eval(), (a, b), backend="fx_importer")
    if not capture.ok:
        raise ValueError(f"model2MLIR did not capture the projection stripe: {capture.diagnostics}")
    if portable_projection_shape(capture.mlir_text) != (m, n, k):
        raise ValueError("isolated model2MLIR projection differs from the selected stripe")
    (args.out_dir / "projection.model2mlir.mlir").write_text(capture.mlir_text)
    contract = ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
    policy = ROOT / "examples/default-policy.yaml"
    require_compute(profile, "fp8_e4m3", "fp8_e4m3", pe_mode=8,
                    activation_projection="direct", weight_projection="direct")
    contract_digest = _sha(contract)
    policy_digest = _sha(policy)
    specialization = {
        "schema": "mx_gemmini.model2mlir_projection_quantization.v1",
        "source_mlir_sha256": worklist["source_mlir_sha256"],
        "isolated_mlir_sha256": hashlib.sha256(capture.mlir_text.encode()).hexdigest(),
        "selected_site": manifest["model2mlir_projection"],
        "format": "fp8_e4m3", "scale_format": "e8m0",
        "rounding": "mxq_bf16_operand_rne",
    }
    specialization_digest = hashlib.sha256(json.dumps(
        specialization, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    binding = (f'contract_sha256 = "{contract_digest}", '
               f'policy_sha256 = "{policy_digest}", '
               f'manifest_sha256 = "{specialization_digest}", '
               f'profile_sha256 = "{profile_sha256(profile)}"')
    handoff = f'''module attributes {{mx.profile_sha256 = "{profile_sha256(profile)}",
  mx.contract_sha256 = "{contract_digest}", mx.policy_sha256 = "{policy_digest}",
  prov.quantization_manifest_sha256 = "{specialization_digest}",
  mx.frontend_mlir_sha256 = "{specialization["isolated_mlir_sha256"]}"}} {{
  func.func @model_projection_slice(%a: tensor<{m}x{k}xi8>,
      %as: tensor<{k // 32}x{m}xi8>, %b: tensor<{k}x{n}xi8>,
      %bs: tensor<{k // 32}x{n}xi8>) -> tensor<{m}x{n}xbf16> {{
    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) {{
      site_id = "functional:matmul", activation_format = "fp8_e4m3",
      weight_format = "fp8_e4m3", activation_projection = "direct",
      weight_projection = "direct", pe_mode = 8 : i32, {binding}}}
      : (tensor<{m}x{k}xi8>, tensor<{k // 32}x{m}xi8>, tensor<{k}x{n}xi8>,
         tensor<{k // 32}x{n}xi8>) -> tensor<{m}x{n}xbf16>
    %out = "mx_gemmini.readout_bf16"(%acc) {{site_id = "functional:matmul", {binding}}}
      : (tensor<{m}x{n}xbf16>) -> tensor<{m}x{n}xbf16>
    func.return %out : tensor<{m}x{n}xbf16>
  }}
}}
'''
    verify_ir(handoff, profile)
    (args.out_dir / "projection.handoff.mlir").write_text(handoff)
    (args.out_dir / "projection_quantization.json").write_text(
        json.dumps(specialization, indent=2, sort_keys=True) + "\n")
    bound = bind_payload(handoff, profile, manifest)
    bound = select_bf16_output_layout(bound, profile, manifest)
    mlir = args.out_dir / "projection.payload_bound.mlir"
    mlir.write_text(bound)
    (args.out_dir / "full_model_site.json").write_text(
        json.dumps({"worklist_schema": worklist["schema"],
                    "source_mlir_sha256": worklist["source_mlir_sha256"],
                    "selected_site": site,
                    "payload_manifest_sha256": _sha(bundle_dir / "manifest.json")},
                   indent=2, sort_keys=True) + "\n")
    obj = args.out_dir / "object"
    command = [sys.executable, "-m", "tools.compile_object", "--mlir", str(mlir),
               "--bundle", str(bundle_dir), "--profile", str(args.profile),
               "--rtl-root", str(args.rtl_root), "--riscv-root", str(args.riscv_root),
               "--out-dir", str(obj)]
    if args.mx_opt:
        command += ["--mx-opt", str(args.mx_opt.resolve())]
    _run(command, cwd=ROOT, log=args.out_dir / "object_compile.log")
    run = args.out_dir / "run"
    command = [sys.executable, "-m", "tools.compile_mx", "--mlir", str(mlir),
               "--bundle", str(bundle_dir), "--profile", str(args.profile),
               "--rtl-root", str(args.rtl_root), "--riscv-root", str(args.riscv_root),
               "--issuer-object", str(obj / "mx_issue.o"), "--out-dir", str(run)]
    if args.run_spike:
        command.append("--run-spike")
    _run(command, cwd=ROOT, log=args.out_dir / "run_compile.log")
    receipt = json.loads((run / "artifact_manifest.json").read_text())
    golden_words = np.frombuffer((bundle_dir / "golden_bf16.bin").read_bytes(),
                                 dtype="<u2").reshape(m, n)
    hardware_values = (golden_words.astype(np.uint32) << 16).view(np.float32)
    pytorch_values = torch.matmul(a[:args.row_count], b).detach().cpu().numpy()
    difference = np.abs(hardware_values[:args.row_count] - pytorch_values)
    if not np.isfinite(difference).all():
        raise ValueError("model projection comparison has nonfinite values")
    summary = {
        "schema": "mx_gemmini.model2mlir_projection_execution.v1",
        "status": "full_k_projection_stripe_matched_on_spike" if args.run_spike else
                  "full_k_projection_stripe_elf_built_unexecuted",
        "scope": "one model2MLIR matmul output stripe; other model operations uncompiled",
        "model2mlir_source_closure_sha256": MODEL2MLIR_SOURCE_CLOSURE,
        "mxq_revision": MXQ_REVISION,
        "full_model_mlir_sha256": worklist["source_mlir_sha256"],
        "site": manifest["model2mlir_projection"],
        "padded_shape_mnk": manifest["shape_mnk"],
        "compared_bf16_values": m * n if args.run_spike else 0,
        "model_output_values": args.row_count * n,
        "padding_output_values": (m - args.row_count) * n,
        "pytorch_f32_vs_mx_bf16_mean_absolute_error": float(difference.mean()),
        "pytorch_f32_vs_mx_bf16_max_absolute_error": float(difference.max()),
        "object_sha256": _sha(obj / "mx_issue.o"),
        "elf_sha256": _sha(run / "mx_program.elf"),
        "run_status": receipt.get("status"),
        "run_receipt_sha256": _sha(run / "artifact_manifest.json"),
    }
    if args.run_spike and receipt.get("status") != "model_projection_matched_on_pinned_spike":
        raise ValueError("MX Spike did not match every BF16 output")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
