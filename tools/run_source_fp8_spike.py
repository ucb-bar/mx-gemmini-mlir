"""Build and run the bound four-wave source FP8 diagnostic on pinned Spike.

This builds Nicolas's libgemmini gitlink in the output directory, so an older
installed Gemmini extension cannot silently stand in for the selected source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.source_baremetal import emit_source_fp8_baremetal
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.target_profile import load_profile, profile_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def _gitlink(root: Path, submodule: str) -> str:
    line = subprocess.check_output(["git", "-C", str(root), "ls-tree", "HEAD", submodule],
                                   text=True).split()
    if len(line) < 3 or line[1] != "commit":
        raise ValueError(f"selected RTL has no {submodule} gitlink")
    return line[2]


def _source_closure(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def _run(command: list[str], *, log: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", required=True, type=Path)
    parser.add_argument("--capture-receipt", required=True, type=Path)
    parser.add_argument("--driver", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--riscv-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    kernel = read_source_gemm(args.driver)
    capture = json.loads(args.capture_receipt.read_text())
    if (capture.get("source_revision") != _revision(args.source_root) or
            capture.get("source_driver_sha256") != _sha(args.driver) or
            capture.get("source_data_header_sha256") != _sha(kernel.data_header) or
            capture.get("target_binding", {}).get("profile_sha256") != profile_sha256(profile) or
            capture.get("target_binding", {}).get("bound_mlir_sha256") != _sha(args.mlir)):
        parser.error("frontend capture does not match selected source, bound MLIR, and profile")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    extension = args.rtl_root / "software/libgemmini"
    if _revision(extension) != _gitlink(args.rtl_root, "software/libgemmini"):
        parser.error("Gemmini extension checkout differs from RTL gitlink")
    bench = software / "riscv-tests/benchmarks/common"
    if not (bench / "test.ld").is_file() or not (software / "riscv-tests/env/encoding.h").is_file():
        parser.error("initialize Gemmini software and nested riscv-tests submodules")

    source = emit_source_fp8_baremetal(
        args.mlir.read_text(), kernel, profile=profile,
        source_root=args.source_root, rtl_root=args.rtl_root)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    c_file = args.out_dir / "source_fp8_128x128x512.c"
    so_file = args.out_dir / "libgemmini.so"
    elf_file = args.out_dir / "source_fp8_128x128x512.elf"
    c_file.write_text(source)

    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    extension_headers = sorted(extension.rglob("*.h"))
    cxx = "g++"
    cxx_executable = shutil.which(cxx)
    if cxx_executable is None:
        parser.error("host g++ is required to build the pinned Gemmini extension")
    riscv_cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if not riscv_cc.is_file() or not spike.is_file():
        parser.error("selected riscv-tools root lacks GCC or Spike")
    build_ext = [cxx, "-L", str(args.riscv_root / "lib"),
                 f"-Wl,-rpath,{args.riscv_root / 'lib'}", "-shared", "-o", str(so_file),
                 "-std=c++17", "-I", str(args.riscv_root / "include"), "-fPIC", "-O3",
                 *(str(path) for path in extension_sources)]
    ext_result = _run(build_ext, log=args.out_dir / "extension_build.log")
    if ext_result.returncode:
        parser.error("pinned Gemmini extension build failed; see extension_build.log")

    common_sources = sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))
    compile_flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET",
                     "-DBAREMETAL=1", "-mcmodel=medany", "-std=gnu99", "-O2",
                     f"-ffile-prefix-map={args.out_dir.resolve()}=.",
                     "-ffast-math", "-fno-common", "-fno-builtin-printf",
                     "-fno-tree-loop-distribute-patterns", "-march=rv64gc", "-Wa,-march=rv64gc",
                     "-I", str(software / "riscv-tests"),
                     "-I", str(software / "riscv-tests/env"), "-I", str(software),
                     "-I", str(bench), "-I", str(kernel.data_header.parent)]
    objects = []
    build_output = []
    for source_file in [c_file, *common_sources]:
        obj_file = args.out_dir / (source_file.stem + ".o")
        compiled = subprocess.run(
            [str(riscv_cc), *compile_flags, "-c", str(source_file), "-o", str(obj_file)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        build_output.append(compiled.stdout)
        if compiled.returncode:
            (args.out_dir / "elf_build.log").write_text("".join(build_output))
            parser.error("source FP8 object build failed; see elf_build.log")
        objects.append(obj_file)
    linked = subprocess.run(
        [str(riscv_cc), "-nostdlib", "-nostartfiles", "-static", "-T",
         str(bench / "test.ld"), *(str(path) for path in objects), "-lm", "-lgcc",
         "-o", str(elf_file)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    build_output.append(linked.stdout)
    (args.out_dir / "elf_build.log").write_text("".join(build_output))
    if linked.returncode:
        parser.error("source FP8 ELF link failed; see elf_build.log")
    result = _run([str(spike), f"--extlib={so_file}", "--extension=gemmini", str(elf_file)],
                  log=args.out_dir / "spike.log")
    passed = result.returncode == 0 and "source FP8 128x128x512: 0 BF16 mismatches" in result.stdout
    receipt = {
        "schema": "mx_gemmini.source_fp8_spike_execution.v1",
        "status": "source_golden_matched_on_pinned_spike" if passed else "source_golden_failed_on_pinned_spike",
        "source_revision": capture["source_revision"],
        "source_driver_sha256": _sha(args.driver),
        "source_data_header_sha256": _sha(kernel.data_header),
        "model2mlir_revision": capture["model2mlir_revision"],
        "bound_mlir_sha256": _sha(args.mlir),
        "target_profile_sha256": profile_sha256(profile),
        "gemmini_revision": _revision(args.rtl_root),
        "gemmini_software_revision": _revision(software),
        "gemmini_extension_revision": _revision(extension),
        "gemmini_extension_source_closure_sha256": _source_closure(
            extension, extension_sources + extension_headers),
        "generated_c_sha256": _sha(c_file),
        "host_cxx_sha256": _sha(Path(cxx_executable)),
        "riscv_gcc_sha256": _sha(riscv_cc),
        "spike_executable_sha256": _sha(spike),
        "extension_build_log_sha256": _sha(args.out_dir / "extension_build.log"),
        "elf_build_log_sha256": _sha(args.out_dir / "elf_build.log"),
        "extension_so_sha256": _sha(so_file),
        "elf_sha256": _sha(elf_file),
        "spike_log_sha256": _sha(args.out_dir / "spike.log"),
        "spike_exit_code": result.returncode,
        "compared_bf16_outputs": 128 * 128,
        "spike_output": result.stdout.strip(),
        "scope": "one serial Rocket FP8 four-wave source-data diagnostic; not the Muon MMIO schedule or RTL",
    }
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{receipt['status']}: {receipt['compared_bf16_outputs']} BF16 outputs")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
