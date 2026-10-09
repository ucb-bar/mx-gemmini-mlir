"""Compile the source-audited Nicolas VPU→requant seam and run pinned Spike."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.source_vector_chain import capture_nicolas_vpu_requant
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.vector_standalone import (write_resident_chain_sources,
                                                   write_vector_requant_sources)
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--riscv-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mx-opt", type=Path)
    parser.add_argument("--with-resident-matmul", action="store_true",
                        help="continue the typed VPU/requant chain through resident MM2")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    software = args.rtl_root / "software/gemmini-rocc-tests"
    extension = args.rtl_root / "software/libgemmini"
    _require_gitlink(args.rtl_root, "software/gemmini-rocc-tests")
    _require_gitlink(args.rtl_root, "software/libgemmini")
    source = software / "bareMetalC/chain_vpu_spad_requant.c"
    header = software / "include/matmul_fp8_64x64_chain.h"
    mlir, resources, facts = capture_nicolas_vpu_requant(
        source, header, profile, include_resident_matmul=args.with_resident_matmul)
    args.out_dir.mkdir(parents=True)
    mlir_path = args.out_dir / "source_bound.mlir"
    mlir_path.write_text(mlir)
    if args.mx_opt is not None:
        _run([str(args.mx_opt.resolve()), str(mlir_path), "-o", "/dev/null"],
             cwd=args.out_dir, log=args.out_dir / "native_verify.log")
    build = args.out_dir / "build"
    writer = (write_resident_chain_sources if args.with_resident_matmul else
              write_vector_requant_sources)
    receipt = writer(build, mlir, profile, resources, facts)
    riscv_cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if not riscv_cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        parser.error("selected RISC-V toolchain or host C++ compiler is absent")
    bench = software / "riscv-tests/benchmarks/common"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             f"-ffile-prefix-map={build.resolve()}=.",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [build / name for name in ("mx_issue.c", "mx_driver.c", "mx_data.S")]
    sources += sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))
    objects = []
    for index, path in enumerate(sources):
        obj = build / f"mx_{index}.o"
        _run([str(riscv_cc), *flags, "-c", str(path), "-o", str(obj)],
             cwd=build, log=build / f"compile_{index}.log")
        objects.append(obj)
    elf = build / "mx_program.elf"
    _run([str(riscv_cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(path) for path in objects), "-lm", "-lgcc",
          "-o", str(elf)], cwd=build, log=build / "link.log")
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = build / "libgemmini.so"
    _run(["g++", "-L", str(args.riscv_root / "lib"),
          f"-Wl,-rpath,{args.riscv_root / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(args.riscv_root / "include"), "-fPIC", "-O3",
          *(str(path) for path in extension_sources)],
         cwd=build, log=build / "extension_build.log")
    result = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini", str(elf)],
                            cwd=build, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log = build / "spike.log"
    log.write_text(result.stdout)
    marker = ("lowered resident chain: C1 0 codes 0 scales, C2 0 codes 0 scales mismatches"
              if args.with_resident_matmul else
              "lowered VPU requant 64x64: 0 FP8 code mismatches, 0 E8M0 scale mismatches")
    passed = result.returncode == 0 and marker in result.stdout
    compiler_root = Path(__file__).resolve().parents[1]
    receipt.update({
        "status": ("source_resident_chain_matched_on_pinned_spike" if passed else
                   "source_resident_chain_failed_on_pinned_spike") if args.with_resident_matmul else
                  ("source_vector_seam_matched_on_pinned_spike" if passed else
                   "source_vector_seam_failed_on_pinned_spike"),
        "source_revision": _git_revision(software),
        "rtl_revision": _git_revision(args.rtl_root),
        "gemmini_extension_revision": _git_revision(extension),
        "compiler_revision": _git_revision(compiler_root),
        "compiler_source_closure_sha256": _source_closure(
            compiler_root, sorted((compiler_root / "mx_gemmini_support").glob("*.py")) +
            sorted((compiler_root / "tools").glob("*.py"))),
        "riscv_gcc_sha256": _sha(riscv_cc), "spike_sha256": _sha(spike),
        "elf_sha256": _sha(elf), "extension_sha256": _sha(so),
        "object_sha256": {path.name: _sha(path) for path in objects},
        "spike_log_sha256": _sha(log), "spike_exit_code": result.returncode,
        "compared_fp8_codes": 8192 if args.with_resident_matmul else 4096,
        "compared_e8m0_scales": 256 if args.with_resident_matmul else 128,
    })
    (build / "artifact_manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{receipt['status']}: {elf}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
