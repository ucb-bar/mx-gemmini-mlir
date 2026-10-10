"""Capture and compile a four-tile Radiance FP8 GEMM with a BF16 VPU MULS epilogue.

The PyTorch trace must contain exactly matmul followed by scalar multiplication.
Source operand bytes and the BF16 matrix golden come from the selected
Radiance header; the compiler derives the BF16 reference and emits all RoCC
commands, including one in-place VPU command for each output tile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

from mx_gemmini_support.bind_payload import (append_tilewise_vpu_muls,
                                            append_tilewise_vpu_x2, bind_payload)
from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.handoff import render_handoff, validate_handoff
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import load_bundle, write_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _run, _sha, _source_closure


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
CONTRACT = ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
POLICY = ROOT / "examples/default-policy.yaml"
DRIVER = "kernels/gemm_mxgemmini/mxgemm.fp8.m256n256k256.tm128tn128tk256.fullout.cpp"
SOURCE_REVISION = "80f84caedbabc663a7433c1da4455b936cca41f3"
SOURCE_DRIVER_SHA = "05f12ce0aeeb5a49d7d0dc782a1aeda65ce48811e31760a8145494ecf858c22e"
SOURCE_HEADER_SHA = "d61f5358db4f6e39e51e0cf7f9867e393a67ce0c1b4d41c596d73a72d2496c70"
MODEL2MLIR_REVISION = "e9ded36eb85abf2d9097ac4dc11457c825853388"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_trace(result, *, quant_format: str = "mxfp8",
                 scalar: float = 2.0) -> None:
    from m2m.coverage import opaque_report

    trace = result.capture_trace or {}
    nodes = trace.get("graphs", {}).get("original", {}).get("nodes", [])
    calls = [node for node in nodes if node.get("op") == "call_function"]
    if (trace.get("status") != "complete" or trace.get("blockers") or
            opaque_report(result.mlir_text) or
            [node.get("target") for node in calls] != [
                "aten.matmul.default", "aten.mul.Tensor"] or
            len([node for node in nodes if node.get("op") == "placeholder"]) != 2 or
            len(calls[0].get("args", [])) != 2 or
            [arg.get("node_id") for arg in calls[0]["args"]] != [
                node["id"] for node in nodes[:2]] or
            calls[1].get("args") != [
                {"node_id": calls[0]["id"], "value_id": calls[0]["id"] + ":v0"}, scalar] or
            "linalg.matmul" not in result.mlir_text or
            'prov.aten = "aten.mul.Tensor"' not in result.mlir_text or
            f"{scalar:.6e}" not in result.mlir_text):
        raise ValueError("model2MLIR trace does not prove matmul followed by selected scalar")
    sites = result.quantization_manifest.get("sites", [])
    if (len(sites) != 1 or sites[0].get("site_id") != "functional:matmul" or
            sites[0].get("status") != "quantized" or
            sites[0].get("format") != quant_format or
            sites[0].get("shape") != [256, 256, 256]):
        raise ValueError("model2MLIR MX site differs from selected Radiance source")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "source-root", "rtl-root",
                 "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--scalar-bits", type=lambda value: int(value, 0),
                        default=0x4000, help="finite BF16 scalar bits (default: 0x4000, 2.0)")
    args = parser.parse_args()
    scalar_bits = args.scalar_bits
    if not 0 <= scalar_bits <= 0xffff or (scalar_bits & 0x7f80) == 0x7f80:
        parser.error("--scalar-bits must encode a finite BF16 value")
    scalar = struct.unpack("<f", (scalar_bits << 16).to_bytes(4, "little"))[0]
    model2mlir, mxq_root, source, rtl, riscv, mx_opt, out = (
        args.model2mlir_root.resolve(), args.mxq_root.resolve(),
        args.source_root.resolve(), args.rtl_root.resolve(),
        args.riscv_root.resolve(), args.mx_opt.resolve(), args.out_dir.resolve())
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    driver = source / DRIVER
    kernel = read_source_gemm(driver)
    if (_git_revision(source) != SOURCE_REVISION or
            _sha(driver) != SOURCE_DRIVER_SHA or
            not kernel.data_header_present or
            _sha(kernel.data_header) != SOURCE_HEADER_SHA or
            tuple(kernel.shape) != (256, 256, 256) or
            tuple(kernel.tile) != (128, 128, 256) or
            kernel.datatype != "FP8" or kernel.quant_output):
        parser.error("selected Radiance source/header differs from qualified FP8 tile workload")
    if _git_revision(model2mlir) != MODEL2MLIR_REVISION:
        parser.error("selected model2MLIR differs from audited current capture")
    profile = load_profile(PROFILE, rtl_root=rtl)
    if not mx_opt.is_file():
        parser.error("compiled mx-gemmini-opt is required")
    sys.path[:0] = [str(model2mlir), str(mxq_root)]
    import m2m
    import mxq
    import torch
    from m2m.capture.external_quantization import ExternalQuantizationConfig

    if (Path(m2m.__file__).resolve().parents[1] != model2mlir or
            Path(mxq.__file__).resolve().parents[1] != mxq_root):
        parser.error("frontend or MX quantization package resolved to another checkout: "
                     f"{m2m.__file__}, {mxq.__file__}")

    class GemmScalar(torch.nn.Module):
        def forward(self, lhs, rhs):
            return torch.matmul(lhs, rhs) * scalar

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        lhs = torch.randn((256, 256), dtype=torch.float32)
        rhs = torch.randn((256, 256), dtype=torch.float32)
    result = m2m.convert(
        GemmScalar().eval(), (lhs, rhs),
        quantization=ExternalQuantizationConfig("mx_gemmini", CONTRACT, POLICY),
        backend="fx_importer", capture_trace=True)
    if not result.ok:
        raise RuntimeError(f"model2MLIR capture failed: {result.diagnostics}")
    _check_trace(result, scalar=scalar)
    validate_handoff(result, CONTRACT.read_bytes(), POLICY.read_bytes())
    handoff = render_handoff(result, CONTRACT.read_bytes(), POLICY.read_bytes())
    selected = bind_handoff(handoff, profile)
    out.mkdir(parents=True)
    for name, content in (("frontend.mlir", result.mlir_text),
                          ("handoff.mlir", handoff),
                          ("profile_bound.mlir", selected),
                          ("capture_trace.json", json.dumps(result.capture_trace, indent=2, sort_keys=True) + "\n"),
                          ("quantization_manifest.json", json.dumps(
                              result.quantization_manifest, indent=2, sort_keys=True) + "\n")):
        (out / name).write_text(content)
    manifest = write_bundle(out / "bundle", kernel, site_id="functional:matmul",
                            profile_sha256=profile_sha256(profile))
    _, resources = load_bundle(out / "bundle")
    payload_bound = bind_payload(selected, profile, manifest)
    bound = (append_tilewise_vpu_x2(payload_bound, profile, manifest)
             if scalar_bits == 0x4000 else
             append_tilewise_vpu_muls(payload_bound, profile, manifest, scalar_bits))
    (out / "payload_bound.mlir").write_text(payload_bound)
    (out / "tilewise_bound.mlir").write_text(bound)
    program = lower_bound_source(bound, profile, manifest, resources)
    if (len(program.plan.get("output_tiles", [])) != 4 or
            sum(step.phase == "vpu" for step in program.steps) != 4 or
            program.derived_expected_bf16 is None):
        raise ValueError("compiler did not apply one VPU epilogue to each output tile")
    _run([str(mx_opt), str(out / "tilewise_bound.mlir"), "-o", "/dev/null"],
         cwd=ROOT, log=out / "mx_opt.log")
    _run([sys.executable, "-m", "tools.compile_mx",
          "--mlir", str(out / "tilewise_bound.mlir"),
          "--bundle", str(out / "bundle"),
          "--profile", str(PROFILE), "--rtl-root", str(rtl),
          "--riscv-root", str(riscv), "--out-dir", str(out / "build"),
          "--run-spike"], cwd=ROOT, log=out / "compile.log")
    compiled = json.loads((out / "build/artifact_manifest.json").read_text())
    if (compiled["status"] != "derived_vpu_golden_matched_on_pinned_spike" or
            compiled["compared_bf16_outputs"] != 65536 or
            compiled["golden_basis"] != (
                "derived_bf16_x2" if scalar_bits == 0x4000 else "derived_bf16_muls")):
        raise RuntimeError("tilewise VPU full-output Spike qualification failed")
    index = {"schema": ("mx_gemmini.radiance_tilewise_vpu_x2_spike.v1"
                         if scalar_bits == 0x4000 else
                         "mx_gemmini.radiance_tilewise_vpu_scalar_spike.v1"),
             "status": ("source_derived_vpu_x2_matched_on_pinned_spike"
                        if scalar_bits == 0x4000 else
                        "source_derived_vpu_scalar_matched_on_pinned_spike"),
             "source_revision": _git_revision(source),
             "source_driver_sha256": _sha(driver),
             "source_header_sha256": _sha(kernel.data_header),
             "model2mlir_revision": _git_revision(model2mlir),
             "mxq_revision": _git_revision(mxq_root),
             "rtl_revision": _git_revision(rtl),
             "profile_sha256": profile_sha256(profile),
             "compiler_revision": _git_revision(ROOT),
             "compiler_source_closure_sha256": _source_closure(
                 ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
                 sorted((ROOT / "tools").glob("*.py"))),
             "shape_mnk": [256, 256, 256], "tile_mnk": [128, 128, 256],
             "output_tiles": 4, "vpu_commands": 4,
             "compared_bf16_outputs": 65536,
             "files_sha256": {name: _sha(out / name) for name in (
                 "frontend.mlir", "handoff.mlir", "profile_bound.mlir",
                 "payload_bound.mlir", "tilewise_bound.mlir", "capture_trace.json",
                 "quantization_manifest.json", "bundle/manifest.json",
                 "build/mx_issue.c", "build/mx_driver.c",
                 "build/physical_program.json", "build/spike.log",
                 "build/artifact_manifest.json")},
             "elf_sha256": compiled["elf_sha256"],
             "extension_sha256": compiled["extension_sha256"]}
    if scalar_bits != 0x4000:
        index["scalar_bf16_bits"] = scalar_bits
        index["scalar_value"] = scalar
    (out / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"four-tile FP8 MX+VPU MULS {scalar}: 65,536/65,536 BF16 outputs matched pinned Spike")


if __name__ == "__main__":
    main()
