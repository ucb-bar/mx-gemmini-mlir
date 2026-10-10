"""Compile one payload-bound MX MLIR contraction to a standalone RV64 ELF.

The source specialization bundle contains the quantized operands and golden
result. This command lowers the typed contraction, emits physical commands,
builds a Rocket executable, and optionally runs it on Nicolas's pinned Spike.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                   text=True).strip()


def _require_gitlink(root: Path, submodule: str) -> None:
    fields = subprocess.check_output(["git", "-C", str(root), "ls-tree", "HEAD", submodule],
                                     text=True).split()
    if len(fields) < 3 or fields[1] != "commit" or fields[2] != _git_revision(root / submodule):
        raise ValueError(f"selected RTL {submodule} checkout differs from its pinned gitlink")


def _source_closure(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path, log: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command[0]}; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--riscv-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-spike", action="store_true")
    args = parser.parse_args()
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    manifest, resources = load_bundle(args.bundle)
    mlir_text = args.mlir.read_text()
    program = lower_bound_source(mlir_text, profile, manifest, resources)
    software = args.rtl_root / "software/gemmini-rocc-tests"
    _require_gitlink(args.rtl_root, "software/gemmini-rocc-tests")
    bench = software / "riscv-tests/benchmarks/common"
    if not (bench / "test.ld").is_file():
        parser.error("selected Gemmini software checkout lacks riscv-tests benchmark linker script")
    riscv_cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    if not riscv_cc.is_file():
        parser.error("selected RISC-V toolchain lacks riscv64-unknown-elf-gcc")
    receipt = write_standalone_sources(args.out_dir, program, resources)
    compiler_root = Path(__file__).resolve().parents[1]
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             f"-ffile-prefix-map={args.out_dir.resolve()}=.",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [args.out_dir / name for name in ("mx_issue.c", "mx_driver.c", "mx_data.S")]
    sources += sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))
    objects = []
    for index, source in enumerate(sources):
        obj = args.out_dir / f"mx_{index}.o"
        _run([str(riscv_cc), *flags, "-c", str(source), "-o", str(obj)],
             cwd=args.out_dir, log=args.out_dir / f"compile_{index}.log")
        objects.append(obj)
    elf = args.out_dir / "mx_program.elf"
    _run([str(riscv_cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(path) for path in objects), "-lm", "-lgcc",
          "-o", str(elf)], cwd=args.out_dir, log=args.out_dir / "link.log")
    receipt.update({
        "status": "rv64_elf_built", "bound_mlir_sha256": _sha(args.mlir),
        "compiler_revision": _git_revision(compiler_root),
        "compiler_source_closure_sha256": _source_closure(
            compiler_root, sorted((compiler_root / "mx_gemmini_support").glob("*.py")) +
            sorted((compiler_root / "tools").glob("*.py"))),
        "source_driver_sha256": manifest["source_driver_sha256"],
        "source_header_sha256": manifest["source_header_sha256"],
        "rtl_revision": _git_revision(args.rtl_root),
        "gemmini_software_revision": _git_revision(software),
        "gemmini_software_source_closure_sha256": _source_closure(
            software, sorted(software.glob("include/*.h")) + sources[3:]),
        "riscv_gcc_sha256": _sha(riscv_cc),
        "elf_sha256": _sha(elf),
        "object_sha256": {path.name: _sha(path) for path in objects},
        "build_log_sha256": {path.name: _sha(path) for path in
                             sorted(args.out_dir.glob("compile_*.log")) + [args.out_dir / "link.log"]},
    })
    if args.run_spike:
        extension = args.rtl_root / "software/libgemmini"
        _require_gitlink(args.rtl_root, "software/libgemmini")
        spike = args.riscv_root / "bin/spike"
        if not spike.is_file() or shutil.which("g++") is None:
            parser.error("Spike run requires pinned spike and host g++")
        extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
        extension_sources += sorted((extension / "perf").rglob("*.cc"))
        so = args.out_dir / "libgemmini.so"
        _run(["g++", "-L", str(args.riscv_root / "lib"),
              f"-Wl,-rpath,{args.riscv_root / 'lib'}", "-shared", "-o", str(so),
              "-std=c++17", "-I", str(args.riscv_root / "include"), "-fPIC", "-O3",
              *(str(path) for path in extension_sources)],
             cwd=args.out_dir, log=args.out_dir / "extension_build.log")
        result = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini", str(elf)],
                                cwd=args.out_dir, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False)
        (args.out_dir / "spike.log").write_text(result.stdout)
        prefix = f"lowered MX {'x'.join(map(str, program.shape))}: "
        expected = (prefix + "0 Radiance FP8 code mismatches, 0 E8M0 scale mismatches"
                    if program.output_format == "radiance_header_fp8" else
                    prefix + "0 FP8 code mismatches, 0 E8M0 scale mismatches"
                    if program.output_format == "fp8_e4m3" else
                    prefix + "0 FP4 packed-code mismatches, 0 E8M0 scale mismatches"
                    if program.output_format == "fp4_e2m1" else
                    prefix + "0 FP6 packed-index mismatches, 0 E8M0 scale mismatches"
                    if program.output_format == "fp6_e3m2" else
                    prefix + "0 BF16 mismatches")
        passed = result.returncode == 0 and expected in result.stdout
        qualifier = ("radiance_header" if program.output_format == "radiance_header_fp8" else
                     "nicolas_oracle" if program.output_format in {"fp8_e4m3", "fp4_e2m1", "fp6_e3m2"} else
                     "derived_vpu_golden" if program.derived_expected_bf16 is not None else
                     "source_golden")
        receipt.update({
            "status": f"{qualifier}_matched_on_pinned_spike" if passed else
                      f"{qualifier}_failed_on_pinned_spike",
            "gemmini_extension_revision": _git_revision(extension),
            "gemmini_extension_source_closure_sha256": _source_closure(
                extension, extension_sources + sorted(extension.rglob("*.h"))),
            "host_cxx_sha256": _sha(Path(shutil.which("g++"))),
            "spike_sha256": _sha(spike), "extension_sha256": _sha(so),
            "spike_exit_code": result.returncode,
            "spike_log_sha256": _sha(args.out_dir / "spike.log"),
            "fp6_spike_scale_selector_workaround": manifest["precision"] == "FP6",
        })
        if program.output_format == "radiance_header_fp8":
            receipt["compared_source_fp8_codes"] = program.shape[0] * program.shape[1]
            receipt["compared_source_e8m0_scales"] = program.shape[0] * program.shape[1] // 32
        elif program.output_format == "fp8_e4m3":
            receipt["compared_fp8_codes"] = program.shape[0] * program.shape[1]
            receipt["compared_e8m0_scales"] = program.shape[0] * program.shape[1] // 32
        elif program.output_format == "fp4_e2m1":
            receipt["compared_fp4_packed_bytes"] = program.shape[0] * program.shape[1] // 2
            receipt["compared_e8m0_scales"] = program.shape[0] * program.shape[1] // 32
        elif program.output_format == "fp6_e3m2":
            receipt["compared_fp6_packed_bytes"] = program.shape[0] * program.shape[1] // 2
            receipt["compared_e8m0_scales"] = program.shape[0] * program.shape[1] // 32
        else:
            receipt["compared_bf16_outputs"] = program.shape[0] * program.shape[1]
    (args.out_dir / "artifact_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{receipt['status']}: {elf}")
    if receipt["status"].endswith("_failed_on_pinned_spike"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
