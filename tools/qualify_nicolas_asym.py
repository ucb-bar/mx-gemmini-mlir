"""Capture PyTorch matmul, bind a Nicolas MX header, and run Spike.

The PyTorch graph supplies the contraction site and shape. Its random example
inputs do not produce the packed operands; the explicit source recipe changes
the captured symmetric MX policy to the selected profile's formats. Named C
tests use checked-in headers. Generated modes use Nicolas's pinned data model
with a separate generation manifest and no replacement handwritten kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from mx_gemmini_support.asymmetric_specialization import (bind_asymmetric_payload,
                                                           emit_baremetal,
                                                           generated_header_recipe,
                                                           lower_asymmetric_physical,
                                                           sha256, source_recipe,
                                                           specialize_handoff)
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _require_gitlink, _run, _source_closure
from tools.qualify_nicolas_plain_matrix_object import (
    MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)


def _revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("lut", "direct", "fp6_lut", "fp4_fp6_lut",
                                              "e4m3_e2m3_lut", "e5m2_fp4_lut",
                                              "e4m3_direct_e3m2", "fp4_direct_e4m3"), default="lut",
                        help="activation format and projection in Nicolas's DIM16 source test")
    parser.add_argument("--source-suffix", type=str,
                        help="Nicolas source pair, for example e2m3_e5m2")
    parser.add_argument("--symmetric-lut", choices=("e2m3", "e3m2", "e4m3", "e5m2"),
                        help="Nicolas's named same-format LUT source test")
    parser.add_argument("--symmetric-lut-requant", action="store_true",
                        help="select the packed same-format LUT-index output source")
    parser.add_argument("--symmetric-fp4", action="store_true",
                        help="Nicolas's named direct FP4 by FP4 BF16 source test")
    parser.add_argument("--generated-mode", choices=(
        "fp4_fp4", "fp4_e4m3", "fp4_e4m3s", "e2m3_e4m3s",
        "e2m3_e2m3", "e2m3_e4m3", "e3m2_e4m3", "e3m2_e3m2",
        "e3m2_e4m3s", "e4m3s_e2m3", "e4m3s_e4m3s",
        "e4m3s_e4m3", "e4m3_e4m3s", "e4m3_e4m3", "e5m2_e5m2"),
        help="header generated with Nicolas's pinned gen_asym.py")
    parser.add_argument("--mesh-dim", type=int, choices=(8, 16, 32), default=16,
                        help="selected Rocket mesh dimension (default: 16)")
    parser.add_argument("--source-shape", choices=("16x32", "64x64", "128x128", "128x128x256",
                                                  "64x128x128", "128x64x128"),
                        default="64x64", help="named Nicolas source shape")
    issue = parser.add_mutually_exclusive_group()
    issue.add_argument("--physical", dest="physical", action="store_true", default=True,
                       help="compile through shared physical command IR (default)")
    issue.add_argument("--diagnostic", dest="physical", action="store_false",
                       help="reproduce the earlier bounded C diagnostic")
    parser.add_argument("--public-object", action="store_true",
                        help="link the data-free object returned by tools.compile_object")
    for name in ("model2mlir-root", "mxq-root", "rtl-root", "profile",
                 "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.symmetric_lut_requant and (
            args.symmetric_lut not in {"e2m3", "e3m2", "e4m3", "e5m2"} or not args.physical or
            (args.symmetric_lut == "e3m2" and
             (args.mesh_dim, args.source_shape) not in {
                 (8, "64x64"), (32, "64x64"),
                 (32, "128x128"), (32, "64x128x128"),
                 (32, "128x64x128")}) or
            (args.symmetric_lut in {"e2m3", "e5m2"} and
             (args.mesh_dim, args.source_shape) not in {
                 (16, "64x64"), (8, "64x64"), (32, "64x64"),
                 (32, "128x128"), (32, "64x128x128"),
                 (32, "128x64x128")}) or
            (args.symmetric_lut == "e4m3" and
             (args.mesh_dim, args.source_shape) not in {
                (16, "64x64"), (8, "64x64"), (32, "64x64"),
                (8, "128x128"), (32, "128x128"),
                (32, "64x128x128"), (32, "128x64x128")})):
        parser.error("packed FP8 LUT readout needs a registered format, mesh, and shape")
    if args.public_object and not args.physical:
        parser.error("public asymmetric object requires physical lowering")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    root = Path(__file__).resolve().parents[1]
    model2mlir, mxq_root = args.model2mlir_root.resolve(), args.mxq_root.resolve()
    sys.path[:0] = [str(root), str(model2mlir), str(mxq_root)]
    import m2m
    import mxq
    import torch
    from m2m.capture.external_quantization import ExternalQuantizationConfig
    from m2m.coverage import opaque_report
    from mx_gemmini_support.handoff import render_handoff, validate_handoff

    if (Path(m2m.__file__).resolve().parents[1] != model2mlir or
            Path(mxq.__file__).resolve().parents[1] != mxq_root):
        raise ValueError("model2MLIR or MXQuant resolved to a different checkout")
    if args.public_object and (
            _revision(model2mlir) != MODEL2MLIR_REVISION or
            _revision(mxq_root) != MXQ_REVISION or
            _revision(args.rtl_root) != RTL_REVISION):
        raise ValueError("public Nicolas asymmetric replay needs pinned frontend and RTL revisions")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    _require_gitlink(args.rtl_root, "software/gemmini-rocc-tests")
    _require_gitlink(args.rtl_root, "software/libgemmini")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    if args.generated_mode:
        if (args.source_suffix or args.symmetric_lut or args.symmetric_fp4 or
                args.source_shape != "64x64" or
                not args.physical):
            parser.error("generated MX mode needs 64x64 physical lowering only")
        if (args.mesh_dim == 16 and args.generated_mode not in {
                "e2m3_e4m3s", "e3m2_e3m2", "e4m3s_e2m3",
                "e4m3s_e4m3", "e4m3_e4m3s"}):
            parser.error("this generated mode is registered for DIM8/DIM32 only")
        suffix = args.generated_mode
        source = software / "gen_asym.py"
        dim_suffix = f"_dim{args.mesh_dim}" if args.mesh_dim != 16 else ""
        header = software / f"include/matmul_data_asym_{suffix}{dim_suffix}.h"
    elif args.symmetric_fp4:
        if (args.source_suffix or args.symmetric_lut or args.source_shape != "64x64" or
                args.mesh_dim != 16):
            parser.error("direct FP4 source selection needs DIM16 64x64 and no other source")
        suffix = "fp4_fp4"
        source = software / "bareMetalC/matmul_tiled_fp4_64x64.c"
        header = software / "include/matmul_fp4_64x64.h"
    elif args.symmetric_lut:
        if args.source_suffix:
            parser.error("same-format LUT source selection does not take a source suffix")
        name = args.symmetric_lut
        precision = "fp6" if name in {"e2m3", "e3m2"} else "fp8"
        lut_suffix = "_lut" if name != "e5m2" else ""
        suffix = f"{name}_{name}"
        if args.mesh_dim == 16 and args.source_shape == "64x64":
            source_name = (f"matmul_tiled_{precision}_{name}{lut_suffix}_64x64"
                           f"{'_requant' if args.symmetric_lut_requant else ''}.c")
            header_name = f"matmul_data_mx_lut_{name}_64x64.h"
        elif (args.mesh_dim, args.source_shape) in {
                (32, "64x64"), (8, "64x64"), (8, "128x128"),
                (32, "128x128"), (32, "64x128x128"),
                (32, "128x64x128")}:
            dim_suffix = f"_dim{args.mesh_dim}"
            nonrequant = "_nonrequant" if args.mesh_dim == 8 and not args.symmetric_lut_requant else ""
            source_name = (f"matmul_tiled_{precision}_{name}{lut_suffix}_{args.source_shape}"
                           f"{'_requant' if args.symmetric_lut_requant else nonrequant}"
                           f"{dim_suffix}.c")
            header_name = f"matmul_data_mx_lut_{name}_{args.source_shape}{dim_suffix}.h"
        else:
            parser.error("same-format LUT source shape and mesh are not registered")
        source = software / "bareMetalC" / source_name
        header = software / "include" / header_name
    else:
        suffix = args.source_suffix or {"lut": "e4m3_fp4", "direct": "e4m3s_fp4",
                  "fp6_lut": "fp6_fp4", "fp4_fp6_lut": "fp4_fp6",
                  "e4m3_e2m3_lut": "e4m3_e2m3",
                  "e5m2_fp4_lut": "e5m2_fp4",
                  "e4m3_direct_e3m2": "e4m3s_e3m2",
                  "fp4_direct_e4m3": "fp4_e4m3s"}[args.variant]
        if not re.fullmatch(r"[a-z0-9]+_[a-z0-9]+", suffix):
            parser.error("source suffix must name one asymmetric source pair")
        dim_suffix = f"_dim{args.mesh_dim}" if args.mesh_dim != 16 else ""
        source = software / f"bareMetalC/matmul_tiled_asym_{suffix}_{args.source_shape}{dim_suffix}.c"
        header_suffix = (f"_{args.source_shape}" if args.source_shape != "64x64" else "")
        header = software / f"include/matmul_data_asym_{suffix}{header_suffix}{dim_suffix}.h"
    recipe = (generated_header_recipe(source, header, profile) if args.generated_mode
              else source_recipe(source, header, profile))
    if not args.physical and recipe["shape"] != [64, 64, 64]:
        parser.error("historical C diagnostic is limited to 64-cubed sources")
    m, n, k = recipe["shape"]

    class Matmul(torch.nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        example = (torch.randn((m, k), dtype=torch.float32),
                   torch.randn((k, n), dtype=torch.float32))
    contract = root / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
    policy = root / "examples/default-policy.yaml"
    result = m2m.convert(
        Matmul().eval(), example,
        quantization=ExternalQuantizationConfig("mx_gemmini", contract, policy),
        backend="fx_importer", capture_trace=True)
    if not result.ok:
        raise RuntimeError(f"model2MLIR capture failed: {result.diagnostics}")
    opaque = opaque_report(result.mlir_text)
    sites = result.quantization_manifest["sites"]
    if opaque or [(site["site_id"], site["status"], site["format"], site["shape"])
                  for site in sites] != [
                      ("functional:matmul", "quantized", "mxfp8", [m, n, k])]:
        raise RuntimeError(f"frontend did not capture one FP8 matmul: {opaque}, {sites}")
    contract_bytes, policy_bytes = contract.read_bytes(), policy.read_bytes()
    validate_handoff(result, contract_bytes, policy_bytes)
    handoff = render_handoff(result, contract_bytes, policy_bytes)
    bound = specialize_handoff(handoff, profile, recipe)
    bound = bind_asymmetric_payload(bound, profile, recipe,
                                    source=source, header=header)
    physical = (lower_asymmetric_physical(bound, profile, recipe,
                                          source=source, header=header)
                if args.physical else None)
    emitted = None if args.physical else emit_baremetal(
        bound, profile, recipe, source=source, header=header)
    args.out_dir.mkdir(parents=True)
    outputs = {
        "model2mlir.mlir": result.mlir_text,
        "handoff.mlir": handoff,
        "asymmetric_bound.mlir": bound,
        "recipe.json": json.dumps(recipe, indent=2, sort_keys=True) + "\n",
        "quantization_manifest.json": json.dumps(result.quantization_manifest, indent=2) + "\n",
    }
    if emitted is not None:
        outputs["asymmetric_issue.c"] = emitted
    for name, content in outputs.items():
        (args.out_dir / name).write_text(content)
    _run([str(args.mx_opt.resolve()), str((args.out_dir / "asymmetric_bound.mlir").resolve()),
          "-o", "/dev/null"], cwd=args.out_dir, log=args.out_dir / "mx_opt.log")
    build_dir = args.out_dir / "physical" if args.physical else args.out_dir
    standalone_receipt = None
    if physical is not None:
        program, resources, resource_manifest = physical
        standalone_receipt = write_standalone_sources(build_dir, program, resources)
        (build_dir / "resource_manifest.json").write_text(
            json.dumps(resource_manifest, indent=2, sort_keys=True) + "\n")
    object_dir = args.out_dir / "object"
    if args.public_object:
        _run([sys.executable, "-m", "tools.compile_object",
              "--mlir", str(args.out_dir / "asymmetric_bound.mlir"),
              "--recipe-json", str(args.out_dir / "recipe.json"),
              "--source-driver", str(source.resolve()),
              "--source-header", str(header.resolve()),
              "--profile", str(args.profile.resolve()),
              "--rtl-root", str(args.rtl_root.resolve()),
              "--riscv-root", str(args.riscv_root.resolve()),
              "--mx-opt", str(args.mx_opt.resolve()),
              "--out-dir", str(object_dir)], cwd=root,
             log=args.out_dir / "object_compile.log")
        if ((object_dir / "mx_issue.c").read_bytes() !=
                (build_dir / "mx_issue.c").read_bytes() or
                (object_dir / "physical_program.json").read_bytes() !=
                (build_dir / "physical_program.json").read_bytes()):
            raise ValueError("public asymmetric object differs from standalone physical commands")

    bench = software / "riscv-tests/benchmarks/common"
    riscv_cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if not (bench / "test.ld").is_file() or not riscv_cc.is_file() or not spike.is_file():
        parser.error("pinned Gemmini benchmark software, RISC-V GCC, or Spike is absent")
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             f"-ffile-prefix-map={build_dir.resolve()}=.",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    source_files = ([build_dir / name for name in (
                    ("mx_driver.c", "mx_data.S") if args.public_object else
                    ("mx_issue.c", "mx_driver.c", "mx_data.S"))]
                    if args.physical else [build_dir / "asymmetric_issue.c"])
    source_files += sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))
    objects = []
    for index, source_file in enumerate(source_files):
        obj = build_dir / f"asymmetric_{index}.o"
        _run([str(riscv_cc), *flags, "-c", str(source_file), "-o", str(obj)],
             cwd=build_dir, log=build_dir / f"compile_{index}.log")
        objects.append(obj)
    elf = build_dir / "asymmetric_program.elf"
    _run([str(riscv_cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"),
          *([str(object_dir / "mx_issue.o")] if args.public_object else []),
          *(str(obj) for obj in objects), "-lm", "-lgcc",
          "-o", str(elf)], cwd=build_dir, log=build_dir / "link.log")

    extension = args.rtl_root / "software/libgemmini"
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = build_dir / "libgemmini.so"
    _run(["g++", *([f"-DGEMMINI_DIM={args.mesh_dim}"] if args.mesh_dim != 16 else []),
          "-L", str(args.riscv_root / "lib"),
          f"-Wl,-rpath,{args.riscv_root / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(args.riscv_root / "include"), "-fPIC", "-O3",
          *(str(path) for path in extension_sources)],
         cwd=build_dir, log=build_dir / "extension_build.log")
    extension_name = "gemmini" if args.mesh_dim == 16 else f"gemmini_dim{args.mesh_dim}"
    run = subprocess.run([str(spike), f"--extlib={so}",
                          f"--extension={extension_name}", str(elf)],
                         cwd=build_dir, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=False)
    (build_dir / "spike.log").write_text(run.stdout)
    expected = (f"lowered MX {m}x{n}x{k}: 0 {args.symmetric_lut.upper()} packed-LUT-index mismatches, "
                "0 E8M0 scale mismatches" if args.symmetric_lut_requant else
                f"lowered MX {m}x{n}x{k}: 0 BF16 mismatches" if args.physical else
                "lowered asymmetric E4M3xFP4 64x64x64: 0 BF16 mismatches")
    passed = run.returncode == 0 and expected in run.stdout
    receipt = {
        "schema": ("mx_gemmini.nicolas_asymmetric_physical_spike_qualification.v1"
                   if args.physical else "mx_gemmini.nicolas_asymmetric_spike_qualification.v1"),
        "status": "source_golden_matched_on_pinned_spike" if passed else
                  "source_golden_failed_on_pinned_spike",
        "scope": (f"DIM{args.mesh_dim} {m}x{n}x{k} {suffix} source specialization of one captured PyTorch matmul; " +
                  ("new header from Nicolas's pinned gen_asym.py, not a checked-in C test; "
                   if args.generated_mode else
                   "checked-in packed inputs, not random PyTorch example inputs; ") +
                  ("shared physical command IR and standalone emitter; " if args.physical else
                   "bounded C diagnostic; ") +
                  "standalone asymmetric MX profile without VPU"),
        "model2mlir_revision": _revision(model2mlir), "mxq_revision": _revision(mxq_root),
        "mesh_dim": args.mesh_dim,
        "compiler_revision": _revision(root),
        "compiler_source_closure_sha256": _source_closure(
            root, sorted((root / "mx_gemmini_support").glob("*.py")) +
            sorted((root / "tools").glob("*.py"))),
        "rtl_revision": _revision(args.rtl_root),
        "software_revision": _revision(software),
        "spike_extension_revision": _revision(extension),
        "profile_name": profile["name"], "profile_sha256": profile_sha256(profile),
        "activation_projection": recipe["compute"]["activation_projection"],
        "activation_format": recipe["compute"]["activation_format"],
        "source_header_sha256": sha256(header),
        **({"source_generator_sha256": sha256(source)} if args.generated_mode else
           {"source_driver_sha256": sha256(source)}),
        **({"source_generation_manifest_sha256":
            recipe["source_generation_manifest_sha256"]} if args.generated_mode else {}),
        "frontend_contract_sha256": sha256(contract), "frontend_policy_sha256": sha256(policy),
        "capture_sites": sites, "opaque_calls": opaque,
        "files_sha256": {name: sha256(args.out_dir / name) for name in outputs},
        "generated_c_sha256": sha256(build_dir / ("mx_issue.c" if args.physical else
                                                  "asymmetric_issue.c")),
        "mx_opt_sha256": sha256(args.mx_opt),
        "mx_opt_log_sha256": sha256(args.out_dir / "mx_opt.log"),
        "riscv_gcc_sha256": sha256(riscv_cc), "spike_sha256": sha256(spike),
        "host_cxx_sha256": sha256(Path(shutil.which("g++"))),
        "extension_source_closure_sha256": _source_closure(
            extension, extension_sources + sorted(extension.rglob("*.h"))),
        "extension_sha256": sha256(so), "elf_sha256": sha256(elf),
        "build_log_sha256": {path.name: sha256(path) for path in
                               sorted(build_dir.glob("compile_*.log")) +
                               [build_dir / "link.log", build_dir / "extension_build.log"]},
        "object_sha256": {obj.name: sha256(obj) for obj in objects},
        "spike_log_sha256": sha256(build_dir / "spike.log"),
        "spike_exit_code": run.returncode,
        **({"compared_packed_lut_bytes": m * n // 2,
            "compared_e8m0_scales": m * n // 32} if args.symmetric_lut_requant else
           {"compared_bf16_outputs": m * n}),
    }
    if standalone_receipt is not None:
        receipt["physical_program_sha256"] = sha256(build_dir / "physical_program.json")
        receipt["resource_manifest_sha256"] = sha256(build_dir / "resource_manifest.json")
        receipt["physical_command_count"] = standalone_receipt["command_count"]
        receipt["physical_fence_count"] = standalone_receipt["fence_count"]
        receipt["generated_driver_sha256"] = sha256(build_dir / "mx_driver.c")
        receipt["generated_data_sha256"] = sha256(build_dir / "mx_data.S")
        receipt["physical_inputs_sha256"] = {
            path.name: sha256(path) for path in sorted(build_dir.iterdir())
            if path.suffix == ".bin" or path.name in
            {"mx_issue.c", "mx_driver.c", "mx_data.S", "physical_program.json",
             "resource_manifest.json"}}
    if args.public_object:
        dispatch = json.loads((object_dir / "compile_manifest.json").read_text())
        object_manifest = json.loads((object_dir / "object_manifest.json").read_text())
        if (dispatch["lowering_family"] != "asymmetric_source" or
                object_manifest["allocated_data_section_bytes"] != 0 or
                object_manifest["object_sha256"] != sha256(object_dir / "mx_issue.o")):
            raise ValueError("public asymmetric object dispatch or data-free ABI differs")
        receipt["scope"] += "; public data-free object compiler"
        receipt["public_object_dispatch_sha256"] = sha256(object_dir / "compile_manifest.json")
        receipt["public_object_manifest_sha256"] = sha256(object_dir / "object_manifest.json")
        receipt["public_object_sha256"] = sha256(object_dir / "mx_issue.o")
        receipt["public_object_abi"] = object_manifest["buffer_abi"]
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{receipt['status']}: {run.stdout.strip()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
