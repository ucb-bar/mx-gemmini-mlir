"""Compile one real model2MLIR attention QK or PV head through MXFP8."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

from mx_gemmini_support.bind_payload import bind_payload, select_bf16_output_layout
from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.model_attention_qk import (
    write_model_attention_pv_bundle, write_model_attention_qk_bundle,
)
from mx_gemmini_support.model_projection import portable_projection_shape
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_mx import _run, _sha, _source_closure


ROOT = Path(__file__).resolve().parents[1]
MODEL2MLIR_SOURCE_CLOSURE = "6eb6648cf2eebd72cd481dff084b0e154e034fbe50b7bce31935091b1a237e39"
MXQ_REVISION = "b4af5430bac147f4a16126931cc0177367cc3982"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full-model-mlir", "activation-npy", "weight-npy", "model2mlir-root",
                 "mxq-root", "reference-root", "profile", "rtl-root", "riscv-root",
                 "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--site-region", required=True)
    parser.add_argument("--stage", choices=("qk", "pv"), default="qk")
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--tile-k", type=int, required=True)
    parser.add_argument("--mx-opt", type=Path)
    parser.add_argument("--run-spike", action="store_true")
    args = parser.parse_args()
    for name in ("full_model_mlir", "activation_npy", "weight_npy", "model2mlir_root",
                 "mxq_root", "reference_root", "profile", "rtl_root", "riscv_root",
                 "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    if _source_closure(args.model2mlir_root,
                       list((args.model2mlir_root / "m2m").rglob("*.py"))) != MODEL2MLIR_SOURCE_CLOSURE:
        raise ValueError("selected model2MLIR source closure differs")
    import subprocess
    if subprocess.check_output(["git", "-C", str(args.mxq_root), "rev-parse", "HEAD"],
                               text=True).strip() != MXQ_REVISION:
        raise ValueError("selected MXQuant revision differs")
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
    matches = [site for site in worklist["rank3_batch_matmuls"]
               if site["region_id"] == args.site_region]
    if len(matches) != 1:
        raise ValueError("full model has no unique batched contraction at selected region")
    site = matches[0]
    batch, source_m, source_n, source_k = site["shape_bmnk"]
    if (".self_attn" not in (site["fqn"] or "") or
            (args.stage == "qk" and (source_n > source_k or source_k % 32)) or
            (args.stage == "pv" and source_n < source_k) or
            not 0 <= args.batch_index < batch):
        raise ValueError("selected batch matmul is not an eligible attention head")
    activation = np.load(args.activation_npy, allow_pickle=False)
    weight = np.load(args.weight_npy, allow_pickle=False)
    args.out_dir.mkdir(parents=True)
    writer = (write_model_attention_qk_bundle if args.stage == "qk" else
              write_model_attention_pv_bundle)
    manifest = writer(
        args.out_dir / "bundle", worklist=worklist, site=site,
        batch_index=args.batch_index, activation=activation, weight=weight,
        tile_k=args.tile_k, profile=profile, reference_root=args.reference_root)
    m, n, k = manifest["shape_mnk"]
    a = torch.zeros((m, k), dtype=torch.float32)
    b = torch.zeros((k, n), dtype=torch.float32)
    a[:source_m, :source_k] = torch.from_numpy(activation)
    b[:source_k, :source_n] = torch.from_numpy(weight)

    class Matmul(torch.nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.matmul(lhs, rhs)

    capture = m2m.convert(Matmul().eval(), (a, b), backend="fx_importer")
    if not capture.ok or portable_projection_shape(capture.mlir_text) != (m, n, k):
        raise ValueError("isolated attention QK matmul is not valid portable model2MLIR")
    (args.out_dir / f"{args.stage}.model2mlir.mlir").write_text(capture.mlir_text)
    contract = ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
    policy = ROOT / "examples/default-policy.yaml"
    specialization = {
        "schema": f"mx_gemmini.model2mlir_attention_{args.stage}_quantization.v1",
        "source_mlir_sha256": worklist["source_mlir_sha256"],
        "isolated_mlir_sha256": hashlib.sha256(capture.mlir_text.encode()).hexdigest(),
        "selected_site": manifest[f"model2mlir_attention_{args.stage}"],
        "format": "fp8_e4m3", "scale_format": "e8m0",
        "rounding": "mxq_bf16_operand_rne",
    }
    specialization_digest = hashlib.sha256(json.dumps(
        specialization, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    binding = (f'contract_sha256 = "{_sha(contract)}", '
               f'policy_sha256 = "{_sha(policy)}", '
               f'manifest_sha256 = "{specialization_digest}", '
               f'profile_sha256 = "{profile_sha256(profile)}"')
    handoff = f'''module attributes {{mx.profile_sha256 = "{profile_sha256(profile)}",
  mx.contract_sha256 = "{_sha(contract)}", mx.policy_sha256 = "{_sha(policy)}",
  prov.quantization_manifest_sha256 = "{specialization_digest}",
  mx.frontend_mlir_sha256 = "{specialization["isolated_mlir_sha256"]}"}} {{
  func.func @model_attention_{args.stage}_head(%a: tensor<{m}x{k}xi8>,
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
    (args.out_dir / f"{args.stage}.handoff.mlir").write_text(handoff)
    (args.out_dir / "quantization.json").write_text(
        json.dumps(specialization, indent=2, sort_keys=True) + "\n")
    bound = bind_payload(handoff, profile, manifest)
    bound = select_bf16_output_layout(bound, profile, manifest)
    mlir = args.out_dir / f"{args.stage}.payload_bound.mlir"
    mlir.write_text(bound)
    obj = args.out_dir / "object"
    command = [sys.executable, "-m", "tools.compile_object", "--mlir", str(mlir),
               "--bundle", str(args.out_dir / "bundle"), "--profile", str(args.profile),
               "--rtl-root", str(args.rtl_root), "--riscv-root", str(args.riscv_root),
               "--out-dir", str(obj)]
    if args.mx_opt:
        command += ["--mx-opt", str(args.mx_opt.resolve())]
    _run(command, cwd=ROOT, log=args.out_dir / "object_compile.log")
    run = args.out_dir / "run"
    command = [sys.executable, "-m", "tools.compile_mx", "--mlir", str(mlir),
               "--bundle", str(args.out_dir / "bundle"), "--profile", str(args.profile),
               "--rtl-root", str(args.rtl_root), "--riscv-root", str(args.riscv_root),
               "--issuer-object", str(obj / "mx_issue.o"), "--out-dir", str(run)]
    if args.run_spike:
        command.append("--run-spike")
    _run(command, cwd=ROOT, log=args.out_dir / "run_compile.log")
    receipt = json.loads((run / "artifact_manifest.json").read_text())
    expected_status = f"model_attention_{args.stage}_matched_on_pinned_spike"
    if args.run_spike and receipt.get("status") != expected_status:
        raise ValueError("MX Spike did not match every attention BF16 output")
    golden_words = np.frombuffer((args.out_dir / "bundle/golden_bf16.bin").read_bytes(),
                                 dtype="<u2").reshape(m, n)
    hardware_values = (golden_words.astype(np.uint32) << 16).view(np.float32)
    pytorch_values = torch.matmul(torch.from_numpy(activation),
                                  torch.from_numpy(weight)).numpy()
    difference = np.abs(hardware_values[:source_m, :source_n] - pytorch_values)
    if not np.isfinite(difference).all():
        raise ValueError("attention comparison has nonfinite values")
    summary = {
        "schema": f"mx_gemmini.model2mlir_attention_{args.stage}_execution.v1",
        "status": (f"{args.stage}_head_matched_on_spike" if args.run_spike else
                   f"{args.stage}_head_elf_built_unexecuted"),
        "scope": ("one batched self-attention QK head; scaling, mask, softmax, and PV uncompiled"
                  if args.stage == "qk" else
                  "one batched self-attention PV head; QK, scale, mask, and softmax uncompiled"),
        "site": manifest[f"model2mlir_attention_{args.stage}"],
        "padded_shape_mnk": manifest["shape_mnk"],
        "model_output_values": source_m * source_n,
        "compared_bf16_values": m * n if args.run_spike else 0,
        "pytorch_f32_vs_mx_bf16_mean_absolute_error": float(difference.mean()),
        "pytorch_f32_vs_mx_bf16_max_absolute_error": float(difference.max()),
        "object_sha256": _sha(obj / "mx_issue.o"),
        "elf_sha256": _sha(run / "mx_program.elf"),
        "run_receipt_sha256": _sha(run / "artifact_manifest.json"),
        "run_status": receipt.get("status"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
