"""Capture PyTorch matmul, specialize Nicolas's E4M3-LUT x FP4 source, run Spike.

The PyTorch graph supplies the contraction site and shape. Its random example
inputs do not produce the checked-in packed operands; the explicit source
recipe changes the captured symmetric MX policy to the source's asymmetric
formats and binds the checked-in data header separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.asymmetric_specialization import (emit_baremetal,
                                                           lower_asymmetric_physical,
                                                           sha256, source_recipe,
                                                           specialize_handoff)
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _require_gitlink, _run, _source_closure


def _revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("lut", "direct"), default="lut",
                        help="E4M3 activation projection in Nicolas's DIM16 source test")
    issue = parser.add_mutually_exclusive_group()
    issue.add_argument("--physical", dest="physical", action="store_true", default=True,
                       help="compile through shared physical command IR (default)")
    issue.add_argument("--diagnostic", dest="physical", action="store_false",
                       help="reproduce the earlier bounded C diagnostic")
    for name in ("model2mlir-root", "mxq-root", "rtl-root", "profile",
                 "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
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
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    _require_gitlink(args.rtl_root, "software/gemmini-rocc-tests")
    _require_gitlink(args.rtl_root, "software/libgemmini")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    suffix = "e4m3_fp4" if args.variant == "lut" else "e4m3s_fp4"
    source = software / f"bareMetalC/matmul_tiled_asym_{suffix}_64x64.c"
    header = software / f"include/matmul_data_asym_{suffix}.h"
    recipe = source_recipe(source, header, profile)

    class Matmul(torch.nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        example = (torch.randn((64, 64), dtype=torch.float32),
                   torch.randn((64, 64), dtype=torch.float32))
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
                      ("functional:matmul", "quantized", "mxfp8", [64, 64, 64])]:
        raise RuntimeError(f"frontend did not capture one FP8 matmul: {opaque}, {sites}")
    contract_bytes, policy_bytes = contract.read_bytes(), policy.read_bytes()
    validate_handoff(result, contract_bytes, policy_bytes)
    handoff = render_handoff(result, contract_bytes, policy_bytes)
    bound = specialize_handoff(handoff, profile, recipe)
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
    source_files = ([build_dir / name for name in ("mx_issue.c", "mx_driver.c", "mx_data.S")]
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
          str(bench / "test.ld"), *(str(obj) for obj in objects), "-lm", "-lgcc",
          "-o", str(elf)], cwd=build_dir, log=build_dir / "link.log")

    extension = args.rtl_root / "software/libgemmini"
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = build_dir / "libgemmini.so"
    _run(["g++", "-L", str(args.riscv_root / "lib"),
          f"-Wl,-rpath,{args.riscv_root / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(args.riscv_root / "include"), "-fPIC", "-O3",
          *(str(path) for path in extension_sources)],
         cwd=build_dir, log=build_dir / "extension_build.log")
    run = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini", str(elf)],
                         cwd=build_dir, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=False)
    (build_dir / "spike.log").write_text(run.stdout)
    expected = ("lowered MX 64x64x64: 0 BF16 mismatches" if args.physical else
                "lowered asymmetric E4M3xFP4 64x64x64: 0 BF16 mismatches")
    passed = run.returncode == 0 and expected in run.stdout
    receipt = {
        "schema": ("mx_gemmini.nicolas_asymmetric_physical_spike_qualification.v1"
                   if args.physical else "mx_gemmini.nicolas_asymmetric_spike_qualification.v1"),
        "status": "source_golden_matched_on_pinned_spike" if passed else
                  "source_golden_failed_on_pinned_spike",
        "scope": (f"64-cubed {args.variant} E4M3 source specialization of one captured PyTorch matmul; "
                  "checked-in packed inputs, not random PyTorch example inputs; " +
                  ("shared physical command IR and standalone emitter; " if args.physical else
                   "bounded C diagnostic; ") +
                  "standalone asymmetric MX profile without VPU"),
        "model2mlir_revision": _revision(model2mlir), "mxq_revision": _revision(mxq_root),
        "compiler_revision": _revision(root),
        "compiler_source_closure_sha256": _source_closure(
            root, sorted((root / "mx_gemmini_support").glob("*.py")) +
            sorted((root / "tools").glob("*.py"))),
        "rtl_revision": _revision(args.rtl_root),
        "software_revision": _revision(software),
        "spike_extension_revision": _revision(extension),
        "profile_name": profile["name"], "profile_sha256": profile_sha256(profile),
        "activation_projection": args.variant,
        "source_driver_sha256": sha256(source), "source_header_sha256": sha256(header),
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
        "compared_bf16_outputs": 4096,
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
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{receipt['status']}: {run.stdout.strip()}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
