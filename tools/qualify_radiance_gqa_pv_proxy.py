"""Capture and run the first source-derived GQA PV contraction on Nicolas Spike.

The P operand follows the pinned Muon requantization formula but substitutes
BF16-rounded Torch exp for mu_fexp. The result qualifies MX PV consumption of
those concrete bytes, not executed Muon softmax or full attention parity.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.source_attention_pv import derive_first_pv_proxy
from mx_gemmini_support.source_attention_qk import LATEST_SOURCE_REVISION, SUBMODULE_REVISION
from mx_gemmini_support.source_gemm import SourceGemm
from mx_gemmini_support.source_payload import (ATTENTION_PV_PROXY_ORIGIN,
                                               make_manifest)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_radiance_gqa_qk import PROFILE
from tools.qualify_radiance_ws_roster import (MODEL2MLIR_REVISION, MXQ_REVISION,
                                              RTL_REVISION, _revision)


ROOT = Path(__file__).resolve().parents[1]
QK_ROSTER = ROOT / "docs/evidence/radiance_gqa_qk_roster_80f84ca"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run(command: list[str], log: Path) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "model2mlir-root", "mxq-root", "rtl-root",
                 "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--baseline-index", type=Path)
    args = parser.parse_args()
    for name in ("source_root", "model2mlir_root", "mxq_root", "rtl_root",
                 "riscv_root", "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    for path, revision in ((args.source_root, LATEST_SOURCE_REVISION),
                           (args.source_root / "lib/mxgemmini", SUBMODULE_REVISION),
                           (args.model2mlir_root, MODEL2MLIR_REVISION),
                           (args.mxq_root, MXQ_REVISION),
                           (args.rtl_root, RTL_REVISION)):
        if _revision(path) != revision:
            raise ValueError(f"selected PV source or tool differs from pinned revision: {path}")
    qk_bundle = QK_ROSTER / "head0_block0/bundle"
    header_gz = QK_ROSTER / "fa_data.h.gz"
    muon_source = args.source_root / "kernels/flash_attention_mx_gqa/flash_mx_impl.hpp"
    sys.path[:0] = [str(args.model2mlir_root), str(args.mxq_root),
                    str(args.source_root / "lib/mxgemmini")]
    import torch
    import m2m
    import mxq
    import fp8_matmul_model as low_level_model
    from m2m.capture.external_quantization import ExternalQuantizationConfig
    from m2m.coverage import opaque_report
    from mx_gemmini_support.bind_payload import bind_payload
    from mx_gemmini_support.bind_profile import bind_handoff
    from mx_gemmini_support.handoff import render_handoff, validate_handoff
    from mx_gemmini_support.verify_profile_ir import verify_ir

    if (Path(m2m.__file__).resolve().parents[1] != args.model2mlir_root or
            Path(mxq.__file__).resolve().parents[1] != args.mxq_root or
            Path(low_level_model.__file__).resolve() !=
            (args.source_root / "lib/mxgemmini/fp8_matmul_model.py").resolve()):
        raise ValueError("PV frontend or numerical model resolved to another checkout")
    resources, policy = derive_first_pv_proxy(
        qk_bundle, header_gz, muon_source, torch=torch,
        low_level_model=low_level_model)
    profile = load_profile(PROFILE, rtl_root=args.rtl_root)

    class Pv(torch.nn.Module):
        def forward(self, probability: torch.Tensor,
                    values: torch.Tensor) -> torch.Tensor:
            return torch.matmul(probability, values)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        examples = (torch.randn(64, 64), torch.randn(64, 64))
    contract = ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
    quant_policy = ROOT / "examples/default-policy.yaml"
    captured = m2m.convert(
        Pv().eval(), examples,
        quantization=ExternalQuantizationConfig("mx_gemmini", contract, quant_policy),
        backend="fx_importer", capture_trace=True)
    if not captured.ok or opaque_report(captured.mlir_text):
        raise ValueError(f"model2MLIR PV capture failed: {captured.diagnostics}")
    sites = captured.quantization_manifest["sites"]
    if (len(sites) != 1 or
            (sites[0]["site_id"], sites[0]["status"], sites[0]["format"],
             sites[0]["shape"]) !=
            ("functional:matmul", "quantized", "mxfp8", [64, 64, 64])):
        raise ValueError("model2MLIR did not select the source PV contraction")
    validate_handoff(captured, contract.read_bytes(), quant_policy.read_bytes())
    handoff = render_handoff(captured, contract.read_bytes(), quant_policy.read_bytes())
    profile_bound = bind_handoff(handoff, profile)
    args.out_dir.mkdir(parents=True)
    header = args.out_dir / "fa_data.h"
    header.write_bytes(gzip.decompress(header_gz.read_bytes()))
    driver = args.source_root / "kernels/flash_attention_mx_gqa/kernel.cpp"
    manifest = make_manifest(
        SourceGemm(driver, header, (64, 64, 64), (64, 64, 64),
                   "FP8", False, False, True), resources,
        site_id="functional:matmul", profile_sha256=profile_sha256(profile))
    manifest["origin"] = ATTENTION_PV_PROXY_ORIGIN
    manifest["source_derivation"] = policy
    bound = bind_payload(profile_bound, profile, manifest)
    verify_ir(bound, profile)
    for name, content in (("model2mlir.mlir", captured.mlir_text),
                          ("handoff.mlir", handoff),
                          ("profile_bound.mlir", profile_bound),
                          ("payload_bound.mlir", bound)):
        (args.out_dir / name).write_text(content)
    _run([str(args.mx_opt), str(args.out_dir / "payload_bound.mlir"),
          "-o", "/dev/null"], args.out_dir / "mx_parse.log")
    bundle = args.out_dir / "bundle"
    bundle.mkdir()
    for name, resource in resources.items():
        (bundle / f"{name}.bin").write_bytes(resource.data)
    _write(bundle / "manifest.json", manifest)
    _run([sys.executable, "-m", "tools.compile_mx", "--mlir",
          str(args.out_dir / "payload_bound.mlir"), "--bundle", str(bundle),
          "--profile", str(PROFILE), "--rtl-root", str(args.rtl_root),
          "--riscv-root", str(args.riscv_root),
          "--out-dir", str(args.out_dir / "build"), "--run-spike"],
         args.out_dir / "compile.log")
    build = args.out_dir / "build"
    result = json.loads((build / "artifact_manifest.json").read_text())
    if (result.get("status") != "source_golden_matched_on_pinned_spike" or
            result.get("compared_bf16_outputs") != 4096 or
            result.get("spike_exit_code") != 0 or
            result.get("source_header_sha256") != policy["source_header_sha256"]):
        raise ValueError("source-derived PV proxy failed the full Spike golden")
    _run([sys.executable, "-m", "tools.emit_mx_object", "--mlir",
          str(args.out_dir / "payload_bound.mlir"), "--bundle", str(bundle),
          "--profile", str(PROFILE), "--rtl-root", str(args.rtl_root),
          "--riscv-root", str(args.riscv_root),
          "--out-dir", str(args.out_dir / "linkable_object")],
         args.out_dir / "emit_object.log")
    object_receipt = json.loads((args.out_dir / "linkable_object/object_manifest.json").read_text())
    if (object_receipt.get("status") != "rv64_rocc_issuer_object_built" or
            object_receipt.get("embedded_operand_bytes") != 0 or
            object_receipt.get("object_sha256") !=
            _sha(QK_ROSTER.parent / "radiance_gqa_runtime_object_80f84ca/head0_object/mx_issue.o")):
        raise ValueError("PV proxy did not lower to the checked linkable MX schedule")
    index = {
        "schema": "mx_gemmini.gqa_pv_proxy_spike.v1",
        "status": "source_derived_pv_proxy_matched_on_pinned_spike",
        "scope": "head0 block0 PV with Torch-exp proxy for Muon P; no Muon execution or full attention parity",
        "source_revision": _revision(args.source_root),
        "model2mlir_revision": _revision(args.model2mlir_root),
        "mxq_revision": _revision(args.mxq_root),
        "rtl_revision": _revision(args.rtl_root),
        "source_header_sha256": policy["source_header_sha256"],
        "muon_requant_source_sha256": policy["muon_requant_source_sha256"],
        "qk_golden_bf16_sha256": policy["qk_golden_bf16_sha256"],
        "frontend_mlir_sha256": _sha(args.out_dir / "model2mlir.mlir"),
        "payload_bound_mlir_sha256": _sha(args.out_dir / "payload_bound.mlir"),
        "bundle_manifest_sha256": _sha(bundle / "manifest.json"),
        "activation_sha256": _sha(bundle / "activation.bin"),
        "activation_scales_sha256": _sha(bundle / "activation_scales.bin"),
        "weight_sha256": _sha(bundle / "weight.bin"),
        "weight_scales_sha256": _sha(bundle / "weight_scales.bin"),
        "golden_bf16_sha256": _sha(bundle / "golden_bf16.bin"),
        "physical_program_sha256": _sha(build / "physical_program.json"),
        "issuer_c_sha256": _sha(build / "mx_issue.c"),
        "object_sha256": object_receipt["object_sha256"],
        "elf_sha256": result["elf_sha256"],
        "spike_log_sha256": _sha(build / "spike.log"),
        "spike_exit_code": result["spike_exit_code"],
        "compared_bf16_outputs": result["compared_bf16_outputs"],
    }
    if args.baseline_index and json.loads(args.baseline_index.read_text()) != index:
        raise ValueError("PV proxy qualification differs from archived baseline")
    _write(args.out_dir / "index.json", index)
    print("source-derived PV proxy: 0/4096 BF16 mismatches on Nicolas Spike")


if __name__ == "__main__":
    main()
