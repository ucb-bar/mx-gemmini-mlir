"""One-command source specialization, MX lowering, RV64 build, and Spike run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from mx_gemmini_support.bind_payload import (
    append_tilewise_vpu_muls, append_tilewise_vpu_x2,
    append_vpu_spad_requant_x2, bind_payload,
    select_bf16_output_layout)
from mx_gemmini_support.capture_epilogue import append_captured_tilewise_vpu_scalar
from mx_gemmini_support.mesh_reference import derive_mesh_reference
from mx_gemmini_support.source_gemm import plan_source_gemm, read_source_gemm
from mx_gemmini_support.source_payload import (load_bundle,
                                                replace_source_golden_with_mesh_reference,
                                                replace_source_quantized_with_mesh_reference,
                                                write_bundle)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.generate_radiance_fp6_header import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", required=True, type=Path,
                        help="profile-bound model2MLIR handoff")
    parser.add_argument("--driver", required=True, type=Path,
                        help="Radiance source driver and adjacent data header")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--riscv-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--site-id", default="functional:matmul")
    parser.add_argument("--fp6-quantized-specialization", action="store_true",
                        help="project the checked-in FP6 fullout BF16 result onto its C LUT")
    parser.add_argument("--vpu-spad-requant-x2", action="store_true",
                        help="compose BF16 matrix, VPU x2, and tiled resident FP8 requant")
    parser.add_argument("--tilewise-vpu-x2", action="store_true",
                        help="apply in-place BF16 VPU x2 before each complete output tile readout")
    parser.add_argument("--tilewise-vpu-muls-bf16-bits", type=lambda value: int(value, 0),
                        help="apply in-place BF16 VPU MULS with this finite scalar immediate per output tile")
    parser.add_argument("--tilewise-vpu-from-capture", action="store_true",
                        help="derive tilewise BF16 MULS/ADDS or a scalar chain from a digest-checked model2MLIR capture")
    parser.add_argument("--capture-trace", type=Path)
    parser.add_argument("--capture-quantization-manifest", type=Path)
    parser.add_argument("--capture-frontend-mlir", type=Path)
    parser.add_argument("--bf16-output-layout", choices=("row_major_bf16",
                                                          "output_tile_major_bf16"),
                        help="select the physical memory layout of final BF16 readout")
    parser.add_argument("--source-header-quantized", action="store_true",
                        help="lower the Radiance FP8 or FP6 C_out convention as a host BF16 epilogue")
    parser.add_argument("--target-mesh-reference", action="store_true",
                        help="derive DIM8/32 BF16 oracle from Radiance's pinned host model")
    parser.add_argument("--rtl-product-floor-reference", action="store_true",
                        help="include Nicolas's below-2^-16 product flush in target reference")
    parser.add_argument("--generate-missing-fp6-header", action="store_true",
                        help="stage a checked source-derived FP6 header without editing Radiance")
    parser.add_argument("--physical-mode", choices=("spike_serial", "rtl_alternating"),
                        default="spike_serial")
    parser.add_argument("--experimental-spike-extension-root", type=Path,
                        help="isolated modified Spike source for an experimental physical mode")
    args = parser.parse_args()
    if args.physical_mode == "rtl_alternating" and not args.experimental_spike_extension_root:
        parser.error("rtl_alternating Spike execution requires an explicit experimental extension")
    if sum((args.tilewise_vpu_x2, args.tilewise_vpu_muls_bf16_bits is not None,
            args.vpu_spad_requant_x2, args.tilewise_vpu_from_capture)) > 1:
        parser.error("select one MX VPU epilogue")
    capture_paths = (args.capture_trace, args.capture_quantization_manifest,
                     args.capture_frontend_mlir)
    if ((args.tilewise_vpu_from_capture and not all(path is not None for path in capture_paths)) or
            (not args.tilewise_vpu_from_capture and any(path is not None for path in capture_paths))):
        parser.error("capture-derived VPU needs all three capture sidecar paths")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if args.rtl_product_floor_reference and not args.target_mesh_reference:
        parser.error("RTL product-floor reference requires --target-mesh-reference")
    if (args.target_mesh_reference and args.source_header_quantized and
            not args.rtl_product_floor_reference):
        parser.error("target-mesh requant requires Nicolas's RTL product-floor reference")
    if args.target_mesh_reference and (
            profile["geometry"]["mesh_columns"] not in {8, 32} or
            args.physical_mode != "spike_serial" or
            args.fp6_quantized_specialization or args.vpu_spad_requant_x2 or
            args.tilewise_vpu_x2 or args.tilewise_vpu_muls_bf16_bits is not None or
            args.tilewise_vpu_from_capture):
        parser.error("target mesh reference needs DIM8/32 source lowering")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    fixture = None
    try:
        selected_driver = args.driver
        generation = None
        if args.generate_missing_fp6_header:
            if args.driver.parent.name != "gemm_mxgemmini":
                parser.error("FP6 fixture generation requires a Radiance gemm_mxgemmini driver")
            if read_source_gemm(args.driver).data_header_present:
                parser.error("FP6 fixture generation requires a missing source header")
            fixture = TemporaryDirectory(prefix="mx-fp6-source-")
            selected_driver = Path(fixture.name) / args.driver.name
            shutil.copy2(args.driver, selected_driver)
            pending = read_source_gemm(selected_driver)
            supported = (pending.shape[:2] == (128, 128) and
                         ((pending.quant_output and pending.shape[2] in {128, 512} and
                           pending.tile == pending.shape) or
                          (not pending.quant_output and
                           pending.tile[2] == {128: 128, 256: 128,
                                               512: 512, 1024: 512}.get(
                                                   pending.shape[2]) and
                           pending.tile[:2] == (128, 128))))
            if pending.data_header_present or pending.datatype != "FP6" or not supported:
                parser.error("FP6 fixture generation requires a missing supported source header")
            generation = generate(args.driver.resolve().parents[2],
                                  pending.shape[2], pending.data_header)
        kernel = read_source_gemm(selected_driver)
        plan_source_gemm(kernel, scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
                         profile=profile)
        args.out_dir.mkdir(parents=True)
        bundle = args.out_dir / "bundle"
        manifest = write_bundle(bundle, kernel, site_id=args.site_id,
                                profile_sha256=profile_sha256(profile),
                                fp6_quantized_specialization=args.fp6_quantized_specialization,
                                vpu_spad_requant_x2=args.vpu_spad_requant_x2)
        if args.target_mesh_reference:
            _, resources = load_bundle(bundle)
            target, policy = derive_mesh_reference(
                selected_driver.resolve().parents[2], resources, kernel.shape,
                kernel.datatype, profile["geometry"]["mesh_columns"],
                product_floor=args.rtl_product_floor_reference)
            (args.out_dir / "source_golden_bf16.bin").write_bytes(resources["golden_bf16"])
            manifest = (replace_source_quantized_with_mesh_reference(bundle, target, policy)
                        if manifest.get("output_format") is not None else
                        replace_source_golden_with_mesh_reference(bundle, target, policy))
        if generation is not None:
            generated = args.out_dir / "source_fixture"
            generated.mkdir()
            shutil.copy2(selected_driver, generated / selected_driver.name)
            shutil.copy2(kernel.data_header, generated / kernel.data_header.name)
            (generated / "generation_receipt.json").write_text(
                json.dumps(generation, indent=2, sort_keys=True) + "\n")
    finally:
        if fixture is not None:
            fixture.cleanup()
    mlir = args.out_dir / "payload_bound.mlir"
    bound = bind_payload(args.mlir.read_text(), profile, manifest,
                         source_header_quantized=args.source_header_quantized)
    if args.vpu_spad_requant_x2:
        bound = append_vpu_spad_requant_x2(bound, profile, manifest)
    if args.tilewise_vpu_x2:
        bound = append_tilewise_vpu_x2(bound, profile, manifest)
    if args.tilewise_vpu_muls_bf16_bits is not None:
        bound = append_tilewise_vpu_muls(
            bound, profile, manifest, args.tilewise_vpu_muls_bf16_bits)
    if args.tilewise_vpu_from_capture:
        capture = SimpleNamespace(
            ok=True,
            capture_trace=json.loads(args.capture_trace.read_text()),
            quantization_manifest=json.loads(
                args.capture_quantization_manifest.read_text()),
            mlir_text=args.capture_frontend_mlir.read_text())
        bound = append_captured_tilewise_vpu_scalar(
            bound, profile, manifest, capture)
    if args.bf16_output_layout:
        bound = select_bf16_output_layout(
            bound, profile, manifest, layout=args.bf16_output_layout)
    mlir.write_text(bound)
    command = [sys.executable, "-m", "tools.compile_mx",
               "--mlir", str(mlir.resolve()), "--bundle", str(bundle.resolve()),
               "--profile", str(args.profile.resolve()),
               "--rtl-root", str(args.rtl_root.resolve()),
               "--riscv-root", str(args.riscv_root.resolve()),
               "--out-dir", str((args.out_dir / "build").resolve()), "--run-spike",
               "--physical-mode", args.physical_mode]
    if args.experimental_spike_extension_root:
        command += ["--experimental-spike-extension-root",
                    str(args.experimental_spike_extension_root.resolve())]
    subprocess.run(command,
                   cwd=Path(__file__).resolve().parents[1], check=True)


if __name__ == "__main__":
    main()
