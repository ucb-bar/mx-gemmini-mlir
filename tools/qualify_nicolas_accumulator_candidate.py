"""Numerically replay three MX accumulator objects on an experimental Spike model.

The patch models BF16 accumulator-tagged MVOUTs from Spike's existing MX
shadow result. It checks the compiler's output addresses and full BF16 result;
it does not reproduce Nicolas's MMIO scale initialization, RTL packing,
queue timing, or FPGA execution.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from tools.compile_nicolas_accumulator_readout import CASES_HARDWARE
from tools.qualify_nicolas_plain_matrix_object import CASES, RTL_REVISION, _driver


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "tools/patches/nicolas_spike_mx_accumulator_candidate_266c593.patch"
DEFAULT_OBJECTS = ROOT / "docs/evidence/nicolas_accumulator_readout_objects_266c593"
PINNED_SOURCES = {
    "gemmini.cc": "92627314fb30593eee63f86da7baf6b00d911a1e79a107ed184378a3baef830e",
    "gemmini.h": "737bb086909e313f4753f1e1ebf2d6ce49780f1049674fda7f649b81f8e17e84",
    "vpu_ref.h": "459726e52e7d19c51bada75b2e3ded16f64371799d5bc443750eb22e8616b17b",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, cwd: Path, log: Path) -> None:
    run = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    log.write_text(run.stdout)
    if run.returncode:
        raise RuntimeError(f"command failed ({run.returncode}); see {log}")


def _check_rtl(rtl_root: Path) -> tuple[Path, Path]:
    revision = subprocess.check_output(
        ["git", "-C", str(rtl_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != RTL_REVISION:
        raise ValueError("Nicolas RTL revision differs from pinned accumulator model")
    extension = rtl_root / "software/libgemmini"
    vpu_ref = rtl_root / "software/gemmini-rocc-tests/include/vpu_ref.h"
    for name in ("gemmini.cc", "gemmini.h"):
        if _sha(extension / name) != PINNED_SOURCES[name]:
            raise ValueError(f"Nicolas {name} differs from pinned accumulator model")
    if _sha(vpu_ref) != PINNED_SOURCES["vpu_ref.h"]:
        raise ValueError("Nicolas VPU reference differs from pinned accumulator model")
    return extension, vpu_ref


def _build_extension(source: Path, vpu_ref: Path, riscv_root: Path,
                     out_dir: Path) -> Path:
    copied = out_dir / "libgemmini"
    shutil.copytree(source, copied, ignore=shutil.ignore_patterns(
        "*.o", "*.so", "build", "__pycache__"))
    ref_target = out_dir / "gemmini-rocc-tests/include/vpu_ref.h"
    ref_target.parent.mkdir(parents=True)
    shutil.copy2(vpu_ref, ref_target)
    _run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(PATCH)], cwd=copied,
         log=out_dir / "patch.log")
    sources = [copied / "gemmini.cc", copied / "gemmini_perf.cc",
               *sorted((copied / "perf").rglob("*.cc"))]
    so = out_dir / "libgemmini_accumulator_candidate.so"
    command = ["g++", "-L", str(riscv_root / "lib"),
               f"-Wl,-rpath,{riscv_root / 'lib'}", "-shared", "-o", str(so),
               "-std=c++17", "-I", str(riscv_root / "include"), "-fPIC", "-O3",
               f"-ffile-prefix-map={out_dir}=/mx-accumulator-candidate",
               *(str(path) for path in sources)]
    _run(command, cwd=out_dir, log=out_dir / "extension_build.log")
    return so


def _run_case(case_key: str, rtl_root: Path, riscv_root: Path,
              object_root: Path, extension: Path, out_dir: Path) -> dict:
    case = CASES[case_key]
    source = rtl_root / "software/gemmini-rocc-tests/bareMetalC" / case.source_name
    header = rtl_root / "software/gemmini-rocc-tests/include" / case.header_name
    if _sha(source) != case.source_sha256 or _sha(header) != case.header_sha256:
        raise ValueError(f"{case_key} source driver or header differs from pinned case")
    object_dir = object_root / case_key / "object"
    object_manifest = json.loads((object_dir / "object_manifest.json").read_text())
    if (object_manifest["object_sha256"] != _sha(object_dir / "mx_issue.o") or
            object_manifest["mode"] != "rtl_accumulator" or
            object_manifest["stock_spike_supported"] is not False or
            object_manifest["allocated_data_section_bytes"] != 0 or
            [entry["name"] for entry in object_manifest["buffer_abi"]] !=
            list(case.buffer_abi)):
        raise ValueError(f"{case_key} is not a checked data-free accumulator object")

    software = rtl_root / "software/gemmini-rocc-tests"
    bench = software / "riscv-tests/benchmarks/common"
    cc = riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = riscv_root / "bin/spike"
    if not cc.is_file() or not spike.is_file() or not (bench / "test.ld").is_file():
        raise ValueError("RISC-V compiler, Spike, or Nicolas linker is unavailable")
    out_dir.mkdir()
    compiler_case = replace(case, label=case.label.replace("Spike fallback",
                                                   "accumulator candidate"))
    driver = out_dir / "mx_driver.c"
    driver.write_text(_driver(compiler_case))
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench), "-I", str(object_dir)]
    source_files = [driver, *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S"))]
    objects = []
    for index, source_file in enumerate(source_files):
        obj = out_dir / f"driver_{index}.o"
        _run([str(cc), *flags, "-c", str(source_file), "-o", str(obj)], cwd=out_dir,
             log=out_dir / f"compile_{index}.log")
        objects.append(obj)
    elf = out_dir / "mx_program.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), str(object_dir / "mx_issue.o"),
          *(str(obj) for obj in objects), "-lm", "-lgcc", "-o", str(elf)],
         cwd=out_dir, log=out_dir / "link.log")
    run = subprocess.run([str(spike), f"--extlib={extension}",
                          "--extension=gemmini", str(elf)], cwd=out_dir,
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = out_dir / "spike.log"
    log.write_text(run.stdout)
    comparisons = case.shape[0] * case.shape[1]
    if (run.returncode or
            f"0 mismatches / {comparisons} BF16 values" not in run.stdout or
            "accumulator candidate" not in run.stdout):
        raise ValueError(f"{case_key} accumulator candidate did not match all BF16 values")
    return {"schema": "mx_gemmini.nicolas_accumulator_candidate_spike.v1",
            "status": "experimental_accumulator_spike_full_bf16_matched",
            "case": case_key, "rtl_revision": RTL_REVISION,
            "source_driver_sha256": case.source_sha256,
            "source_header_sha256": case.header_sha256,
            "object_sha256": object_manifest["object_sha256"],
            "object_manifest_sha256": _sha(object_dir / "object_manifest.json"),
            "driver_sha256": _sha(driver), "elf_sha256": _sha(elf),
            "extension_sha256": _sha(extension), "spike_sha256": _sha(spike),
            "spike_log_sha256": _sha(log),
            "bf16_values_checked": comparisons, "mismatches": 0,
            "model_scope": "experimental_mx_smem_shadow_to_accumulator_mvout",
            "rtl_or_fpga_qualified": False,
            "source_hardware_scale_semantics_matched": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, default=DEFAULT_OBJECTS)
    args = parser.parse_args()
    for name in ("rtl_root", "riscv_root", "object_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    source, vpu_ref = _check_rtl(args.rtl_root)
    args.out_dir.mkdir(parents=True)
    extension = _build_extension(source, vpu_ref, args.riscv_root, args.out_dir)
    receipts = [_run_case(case, args.rtl_root, args.riscv_root, args.object_root,
                          extension, args.out_dir / case) for case in CASES_HARDWARE]
    index = {"schema": "mx_gemmini.nicolas_accumulator_candidate_index.v1",
             "status": "experimental_accumulator_spike_full_bf16_matched",
             "rtl_revision": RTL_REVISION,
             "patch_sha256": _sha(PATCH),
             "pinned_extension_source_sha256": PINNED_SOURCES,
             "extension_sha256": _sha(extension),
             "cases": receipts, "total_bf16_values_checked": sum(
                 receipt["bf16_values_checked"] for receipt in receipts),
             "rtl_or_fpga_qualified": False,
             "source_hardware_scale_semantics_matched": False}
    (args.out_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(args.out_dir / "index.json")


if __name__ == "__main__":
    main()
