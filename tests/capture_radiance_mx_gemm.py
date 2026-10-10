"""Capture a Radiance MX GEMM shape through current model2MLIR and the MX adapter.

This is a structural frontend/handoff test. The PyTorch inputs do not reproduce
the handwritten MX code and scale blobs. Native output is checked separately
against the source BF16 golden or a named, profile-specific quantized oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha(root: Path, package: str) -> str:
    files = sorted((root / package).rglob("*.py"))
    record = {str(path.relative_to(root)): sha(path) for path in files}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", type=Path, required=True)
    parser.add_argument("--mxq-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--driver", type=Path,
                        default=Path("kernels/gemm_mxgemmini/"
                                     "mxgemm.fp8.m128n128k512.tm128tn128tk128.fullout.cpp"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mx-opt", type=Path, required=True)
    parser.add_argument("--radiance-opt", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--rtl-root", type=Path)
    parser.add_argument("--policy", type=Path)
    args = parser.parse_args()
    if bool(args.profile) != bool(args.rtl_root):
        parser.error("--profile and --rtl-root must be supplied together")
    m2m_root = args.model2mlir_root.resolve()
    mxq_root = args.mxq_root.resolve()
    source_root = args.source_root.resolve()
    support_root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(support_root))
    from mx_gemmini_support.source_gemm import (plan_source_gemm, read_source_gemm,
                                                source_scratchpad_bytes)
    driver = args.driver if args.driver.is_absolute() else source_root / args.driver
    driver = driver.resolve()
    if not driver.is_relative_to(source_root):
        parser.error("source driver must be inside the selected radiance-kernels checkout")
    kernel = read_source_gemm(driver)
    if kernel.datatype not in ("FP8", "FP6", "FP4") or not kernel.data_header_present:
        parser.error("this capture requires a source FP8/FP6/FP4 driver with data")
    if kernel.datatype == "FP6" and args.policy is None:
        parser.error("source FP6 capture requires an explicit structural codebook policy")
    source_bytes = source_scratchpad_bytes(source_root / "lib/mxgemm/mxgemm_lib.hpp")
    source_plan_error = None
    try:
        source_plan = plan_source_gemm(kernel, scratchpad_bytes=source_bytes)
    except ValueError as error:
        if str(error) != "C does not fit beside double-buffered A/B tiles":
            raise
        source_plan = None
        source_plan_error = str(error)
    shape = (kernel.shape[0], kernel.shape[2], kernel.shape[1])
    for root in (m2m_root, mxq_root):
        sys.path.insert(0, str(root))
    import m2m
    import mxq
    import torch
    from m2m.capture.external_quantization import ExternalQuantizationConfig
    from m2m.coverage import opaque_report
    from mx_gemmini_support.handoff import render_handoff, validate_handoff
    from mx_gemmini_support.bind_profile import bind_handoff
    from mx_gemmini_support.target_profile import load_profile, profile_sha256
    from mx_gemmini_support.verify_profile_ir import verify_ir

    if Path(m2m.__file__).resolve().parents[1] != m2m_root:
        raise RuntimeError("model2MLIR resolved to a different checkout")
    if Path(mxq.__file__).resolve().parents[1] != mxq_root:
        raise RuntimeError("microscaling-quant resolved to a different checkout")
    generator = source_root / "kernels/gemm_mxgemmini/gen_mxgemm_data.py"
    spec = importlib.util.spec_from_file_location("radiance_mx_data_generator", generator)
    assert spec and spec.loader
    source_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source_data)
    if kernel.datatype != "FP6" and driver.parent.name == "gemm_mxgemmini":
        generated_shapes = source_data.MISSING_FP8 if kernel.datatype == "FP8" else source_data.MISSING_FP4
        if (kernel.shape[0], kernel.shape[1], kernel.shape[2]) not in generated_shapes:
            raise RuntimeError("the chosen GEMM shape is absent from source generator")

    class Gemm(torch.nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.matmul(lhs, rhs)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        lhs = torch.randn((shape[0], shape[1]), dtype=torch.float32)
        rhs = torch.randn((shape[1], shape[2]), dtype=torch.float32)
    contract = support_root / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
    policy = args.policy or support_root / ("examples/fp4-policy.yaml" if kernel.datatype == "FP4"
                                           else "examples/default-policy.yaml")
    policy = policy.resolve()
    result = m2m.convert(
        Gemm().eval(), (lhs, rhs),
        quantization=ExternalQuantizationConfig("mx_gemmini", contract, policy),
        backend="fx_importer", capture_trace=True,
    )
    if not result.ok:
        raise RuntimeError(f"model2MLIR failed: {result.diagnostics}")
    opaque = opaque_report(result.mlir_text)
    if opaque:
        raise RuntimeError(f"model2MLIR left opaque calls: {opaque}")
    sites = result.quantization_manifest["sites"]
    if len(sites) != 1 or any(sites[0].get(key) != value for key, value in {
            "site_id": "functional:matmul", "status": "quantized",
            "format": {"FP8": "mxfp8", "FP6": "mxfp6", "FP4": "mxfp4"}[kernel.datatype],
            "shape": list(kernel.shape)}.items()):
        raise RuntimeError(f"MX contraction site was not selected: {sites}")
    contract_bytes, policy_bytes = contract.read_bytes(), policy.read_bytes()
    validate_handoff(result, contract_bytes, policy_bytes)
    source_mlir = out / "mx_gemm.model2mlir.mlir"
    source_mlir.write_text(result.mlir_text)
    handoff = out / "mx_gemm.handoff.mlir"
    handoff.write_text(render_handoff(result, contract_bytes, policy_bytes))
    for name, tool in (("mx", args.mx_opt), ("radiance", args.radiance_opt)):
        completed = subprocess.run([str(tool.resolve()), str(handoff), "-o", "/dev/null"],
                                   capture_output=True, text=True)
        (out / f"{name}_parse.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{name} dialect rejected handoff; see {name}_parse.log")
    target_receipt = None
    if args.profile:
        profile = load_profile(args.profile, rtl_root=args.rtl_root)
        target_plan = plan_source_gemm(
            kernel, scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
            profile=profile)
        bound = out / "mx_gemm.profile_bound.mlir"
        bound.write_text(bind_handoff(handoff.read_text(), profile))
        verify_ir(bound.read_text(), profile)
        subprocess.run([str(args.mx_opt.resolve()), str(bound), "-o", "/dev/null"], check=True)
        target_receipt = {
            "profile_name": profile["name"],
            "profile_sha256": profile_sha256(profile),
            "bound_mlir_sha256": sha(bound),
            "scratchpad_geometry_matches_source":
                target_plan["scratchpad_bytes"] == source_bytes,
            "target_layout": {key: target_plan[key] for key in (
                "scratchpad_bytes", "a_rows", "b_rows", "c_rows", "c_spad_dest",
                "a_scale_bytes_per_wave", "b_scale_bytes_per_wave", "lut_once", "move_out")},
            "qualification": profile["qualification"],
        }
    (out / "quantization_manifest.json").write_text(
        json.dumps(result.quantization_manifest, indent=2) + "\n")
    receipt = {
        "schema": "mx_gemmini_model2mlir_radiance_gemm_capture.v1",
        "status": "source_shape_frontend_handoff_only",
        "source_revision": git(source_root, "rev-parse", "HEAD"),
        "source_generator_sha256": sha(generator),
        "source_driver": str(driver.relative_to(source_root)),
        "source_driver_sha256": sha(driver),
        "source_data_header_sha256": sha(kernel.data_header),
        "source_data_header_origin": "tracked" if subprocess.run(
            ["git", "-C", str(source_root), "ls-files", "--error-unmatch",
             str(kernel.data_header.relative_to(source_root))],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            else "generated_or_untracked",
        "source_shape": list(kernel.shape),
        "source_tile": list(kernel.tile),
        "source_layout": {key: source_plan[key] for key in (
            "scratchpad_bytes", "a_rows", "b_rows", "c_rows", "c_spad_dest",
            "a_scale_bytes_per_wave", "b_scale_bytes_per_wave", "lut_once", "move_out")}
            if source_plan is not None else None,
        "source_plan_error": source_plan_error,
        "source_k_waves": len(source_plan["waves"]) if source_plan is not None
                          else kernel.shape[2] // kernel.tile[2],
        "model2mlir_revision": git(m2m_root, "rev-parse", "HEAD"),
        "model2mlir_source_tree_sha256": tree_sha(m2m_root, "m2m"),
        "mxq_revision": git(mxq_root, "rev-parse", "HEAD"),
        "mxq_source_tree_sha256": tree_sha(mxq_root, "mxq"),
        "mx_support_revision": git(support_root, "rev-parse", "HEAD"),
        "contract_sha256": sha(contract), "policy_sha256": sha(policy),
        "source_mlir_sha256": sha(source_mlir), "handoff_mlir_sha256": sha(handoff),
        "quantization_manifest_sha256": sha(out / "quantization_manifest.json"),
        "selected_site": sites[0], "opaque_calls": opaque,
        "target_binding": target_receipt,
        "numerical_scope": "PyTorch inputs are not source MX blobs; no MX numerical parity claimed",
        "lowering_scope": ("frontend capture and profile-bound MX handoff; "
                           "the source-specialized physical compiler is qualified separately"),
    }
    if kernel.quant_output:
        receipt["source_quant_output"] = True
        receipt["mx_support_source_tree_sha256"] = tree_sha(support_root, "mx_gemmini_support")
        receipt["capture_script_sha256"] = sha(Path(__file__))
        receipt["frontend_output_scope"] = (
            "PyTorch captures BF16 matmul structure; the source-bound payload "
            "specializes the terminal readout to source quantized output")
    destination = out / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"site": sites[0]["site_id"], "handoff": str(handoff),
                      "receipt": str(destination)}))


if __name__ == "__main__":
    main()
