"""Generate a clearly derived FP4 four-tile fixture and compile its MX+VPU graph.

Radiance does not contain this 256x256 FP4 driver. The pinned Radiance FP8
driver supplies only a source template; its generator and golden model produce
new FP4 operands and a BF16 matrix reference. The PyTorch graph supplies the
matmul -> x2 structure. This test qualifies the compiler and selected Spike
profile, not parity with a committed Radiance source ELF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.bind_payload import append_tilewise_vpu_x2, bind_payload
from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.handoff import render_handoff, validate_handoff
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import load_bundle, write_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _run, _sha, _source_closure
from tools.qualify_radiance_tilewise_vpu_x2 import _check_trace


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
CONTRACT = ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
POLICY = ROOT / "examples/fp4-policy.yaml"
SOURCE_REVISION = "80f84caedbabc663a7433c1da4455b936cca41f3"
MODEL2MLIR_REVISION = "e9ded36eb85abf2d9097ac4dc11457c825853388"
BASE = "mxgemm.fp8.m256n256k256.tm128tn128tk256.fullout.cpp"
DRIVER = "mxgemm.fp4.m256n256k256.tm128tn128tk128.fullout.cpp"
HEADER = "mxgemm.data.fp4.m256n256k256.h"
SOURCE_HASHES = {
    "kernels/gemm_mxgemmini/" + BASE:
        "05f12ce0aeeb5a49d7d0dc782a1aeda65ce48811e31760a8145494ecf858c22e",
    "kernels/gemm_mxgemmini/gen_mxgemm_data.py":
        "5e405988edf10a9c25a12a6111599cfa93963b0e2586bc9fc2fad30f7b4abef7",
    "lib/golden/Makefile":
        "b2b17a123be425f97705f74a37c6aae3978fd004fa33872a1be1ee5066395705",
    "lib/golden/golden.py":
        "da25bd520b0a1d694743a3224cc3f368dcc102902044c1c8a8da583cd4431747",
    "lib/golden/mx_fp_math.h":
        "e91f2f83ff58c8d6a4c5c1161c9df4a63fc34e7052528a0b87f4dc27116e7b60",
    "lib/golden/mx_golden.cpp":
        "538e83ffa93b33cfb5ad335a90a93318f12128a0202229e64a85300b0f2b3988",
}
DRIVER_SHA256 = "e8e5d52682a32b2e22a52611f6bf5607a2c4f9a02f6b3274946f8bf83dc55ebb"
HEADER_SHA256 = "5eaa3b83ca451eea8ddad9d74c4f1c103b0adfa83f3af8082dfba68e9c98fc25"


def _stage_fixture(source: Path, out: Path) -> tuple[Path, dict]:
    if _git_revision(source) != SOURCE_REVISION or any(
            _sha(source / name) != digest for name, digest in SOURCE_HASHES.items()):
        raise ValueError("Radiance source template or golden generator differs from pinned revision")
    fixture = out / "fixture"
    kernel_dir = fixture / "kernels/gemm_mxgemmini"
    golden_dir = fixture / "lib/golden"
    kernel_dir.mkdir(parents=True)
    golden_dir.mkdir(parents=True)
    for name in SOURCE_HASHES:
        target = fixture / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    base = (kernel_dir / BASE).read_text()
    derived = (base.replace("mxgemm.data.fp8.m256n256k256.h", HEADER)
               .replace(".TILE_K = 256,", ".TILE_K = 128,")
               .replace(".DATATYPE = GemmDatatype::FP8,",
                        ".DATATYPE = GemmDatatype::FP4,")
               .replace('\n\n#include "mxgemm_lib.hpp"',
                        '\nstatic const uint8_t *A_in = &A_in_hw[0][0];\n\n'
                        '#include "mxgemm_lib.hpp"'))
    if (derived == base or derived.count(HEADER) != 1 or
            derived.count("GemmDatatype::FP4") != 1 or
            derived.count("static const uint8_t *A_in = &A_in_hw[0][0];") != 1):
        raise ValueError("Radiance driver template no longer has the selected transformation")
    driver = kernel_dir / DRIVER
    driver.write_text(derived)
    if _sha(driver) != DRIVER_SHA256:
        raise ValueError("derived FP4 driver differs from pinned transformation")
    _run(["make", "mx_golden"], cwd=golden_dir, log=out / "golden_build.log")
    _run([sys.executable, str(kernel_dir / "gen_mxgemm_data.py"),
          "fp4", "256", "256", "256"], cwd=kernel_dir,
         log=out / "header_generation.log")
    if _sha(kernel_dir / HEADER) != HEADER_SHA256:
        raise ValueError("generated FP4 header differs from pinned Radiance golden")
    derivation = {
        "schema": "mx_gemmini.radiance_generated_fp4_gemm_fixture.v1",
        "transformation": "fp8_m256n256k256_tk256_to_fp4_tk128_with_activation_alias_v2",
        "source_revision": SOURCE_REVISION,
        "base_driver_sha256": SOURCE_HASHES["kernels/gemm_mxgemmini/" + BASE],
        "source_generator_sha256": SOURCE_HASHES["kernels/gemm_mxgemmini/gen_mxgemm_data.py"],
        "golden_model_sha256": SOURCE_HASHES["lib/golden/golden.py"],
        "golden_cpp_sha256": SOURCE_HASHES["lib/golden/mx_golden.cpp"],
        "derived_driver_sha256": DRIVER_SHA256,
        "generated_header_sha256": HEADER_SHA256,
    }
    return driver, derivation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "source-root", "rtl-root",
                 "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    model2mlir, mxq_root, source, rtl, riscv, mx_opt, out = (
        args.model2mlir_root.resolve(), args.mxq_root.resolve(),
        args.source_root.resolve(), args.rtl_root.resolve(),
        args.riscv_root.resolve(), args.mx_opt.resolve(), args.out_dir.resolve())
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    if _git_revision(model2mlir) != MODEL2MLIR_REVISION:
        parser.error("selected model2MLIR differs from audited current capture")
    profile = load_profile(PROFILE, rtl_root=rtl)
    if not mx_opt.is_file():
        parser.error("compiled mx-gemmini-opt is required")
    out.mkdir(parents=True)
    driver, derivation = _stage_fixture(source, out)
    kernel = read_source_gemm(driver)
    if (tuple(kernel.shape) != (256, 256, 256) or
            tuple(kernel.tile) != (128, 128, 128) or
            kernel.datatype != "FP4" or kernel.quant_output):
        raise ValueError("generated FP4 fixture geometry differs")
    sys.path[:0] = [str(model2mlir), str(mxq_root)]
    import m2m
    import mxq
    import torch
    from m2m.capture.external_quantization import ExternalQuantizationConfig

    if (Path(m2m.__file__).resolve().parents[1] != model2mlir or
            Path(mxq.__file__).resolve().parents[1] != mxq_root):
        parser.error("frontend or MX quantization package resolved to another checkout")

    class GemmX2(torch.nn.Module):
        def forward(self, lhs, rhs):
            return torch.matmul(lhs, rhs) * 2.0

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        lhs = torch.randn((256, 256), dtype=torch.float32)
        rhs = torch.randn((256, 256), dtype=torch.float32)
    result = m2m.convert(
        GemmX2().eval(), (lhs, rhs),
        quantization=ExternalQuantizationConfig("mx_gemmini", CONTRACT, POLICY),
        backend="fx_importer", capture_trace=True)
    if not result.ok:
        raise RuntimeError(f"model2MLIR capture failed: {result.diagnostics}")
    _check_trace(result, quant_format="mxfp4")
    contract, policy = CONTRACT.read_bytes(), POLICY.read_bytes()
    validate_handoff(result, contract, policy)
    handoff = render_handoff(result, contract, policy)
    selected = bind_handoff(handoff, profile)
    for name, content in (("frontend.mlir", result.mlir_text),
                          ("handoff.mlir", handoff),
                          ("profile_bound.mlir", selected),
                          ("capture_trace.json", json.dumps(
                              result.capture_trace, indent=2, sort_keys=True) + "\n"),
                          ("quantization_manifest.json", json.dumps(
                              result.quantization_manifest, indent=2, sort_keys=True) + "\n")):
        (out / name).write_text(content)
    manifest = write_bundle(out / "bundle", kernel, site_id="functional:matmul",
                            profile_sha256=profile_sha256(profile),
                            source_derivation=derivation)
    _, resources = load_bundle(out / "bundle")
    payload_bound = bind_payload(selected, profile, manifest)
    bound = append_tilewise_vpu_x2(payload_bound, profile, manifest)
    (out / "payload_bound.mlir").write_text(payload_bound)
    (out / "tilewise_bound.mlir").write_text(bound)
    program = lower_bound_source(bound, profile, manifest, resources)
    if (len(program.plan.get("output_tiles", [])) != 4 or
            sum(step.phase == "vpu" for step in program.steps) != 4 or
            program.derived_expected_bf16 is None):
        raise ValueError("compiler did not apply one VPU epilogue to each FP4 output tile")
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
            compiled["golden_basis"] != "derived_bf16_x2"):
        raise RuntimeError("derived FP4 tilewise VPU full-output Spike test failed")
    index = {
        "schema": "mx_gemmini.radiance_generated_fp4_tilewise_vpu_spike.v1",
        "status": "generated_fp4_fixture_vpu_x2_matched_on_pinned_spike",
        "scope": "derived FP4 fixture; no committed Radiance driver or source ELF parity",
        "source_revision": _git_revision(source),
        "source_derivation": derivation,
        "model2mlir_revision": _git_revision(model2mlir),
        "mxq_revision": _git_revision(mxq_root),
        "rtl_revision": _git_revision(rtl),
        "profile_sha256": profile_sha256(profile),
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "shape_mnk": [256, 256, 256], "tile_mnk": [128, 128, 128],
        "output_tiles": 4, "vpu_commands": 4,
        "compared_bf16_outputs": 65536,
        "files_sha256": {name: _sha(out / name) for name in (
            "fixture/kernels/gemm_mxgemmini/" + DRIVER,
            "fixture/kernels/gemm_mxgemmini/" + HEADER,
            "frontend.mlir", "handoff.mlir", "profile_bound.mlir",
            "payload_bound.mlir", "tilewise_bound.mlir", "capture_trace.json",
            "quantization_manifest.json", "bundle/manifest.json",
            "build/mx_issue.c", "build/mx_driver.c",
            "build/physical_program.json", "build/spike.log",
            "build/artifact_manifest.json")},
        "elf_sha256": compiled["elf_sha256"],
        "extension_sha256": compiled["extension_sha256"],
    }
    (out / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print("generated FP4 four-tile MX+VPU: 65,536/65,536 BF16 outputs matched Spike")


if __name__ == "__main__":
    main()
