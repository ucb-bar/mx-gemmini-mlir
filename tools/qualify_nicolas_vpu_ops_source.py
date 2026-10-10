"""Run Nicolas's complete BF16 VPU source oracle on the pinned Spike model.

This establishes the 29-check source baseline for compiler-issued VPU parity.
The executable is Nicolas's handwritten vpu_ops.c, not compiler output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json"
EXPECTED = ("add", "sub", "mul", "mul same-bank", "mul bcast", "max",
            "max same-bank", "max bcast", "sub bcast sb", "expsub",
            "expsub bcast", "expsub sb", "expsum bcast", "expsum sums",
            "expsum sb", "expsum sb sums", "adds", "muls", "exp", "rcp",
            "rsqrt", "rmax", "ramax", "rsum", "rsum rlen1", "chain",
            "war mvin", "dual X,Y", "dual Z=X*B")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--riscv-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    rtl, riscv, out = (args.rtl_root.resolve(), args.riscv_root.resolve(),
                       args.out_dir.resolve())
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    software = rtl / "software/gemmini-rocc-tests"
    extension = rtl / "software/libgemmini"
    _require_gitlink(rtl, "software/gemmini-rocc-tests")
    _require_gitlink(rtl, "software/libgemmini")
    profile = load_profile(PROFILE, rtl_root=rtl)
    if (profile["transport"] != "rocket_rocc" or
            profile["resources"].get("vpu_config") !=
            {"units": 2, "exp_sub": True, "exp_sum": True}):
        parser.error("selected Nicolas profile lacks the two-unit fused VPU")
    cc, spike = (riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike")
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        parser.error("RISC-V GCC, Spike, and host g++ are required")
    source = software / "bareMetalC/vpu_ops.c"
    reference = software / "include/vpu_ref.h"
    bench = software / "riscv-tests/benchmarks/common"
    if not source.is_file() or not reference.is_file() or not (bench / "test.ld").is_file():
        parser.error("Nicolas VPU source, oracle, or benchmark linker script is absent")
    out.mkdir(parents=True)
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-DVPU_FUSED=1", "-mcmodel=medany", "-std=gnu99", "-O2",
             "-ffast-math", "-fno-common", "-fno-builtin-printf",
             "-fno-tree-loop-distribute-patterns", "-march=rv64gc",
             "-Wa,-march=rv64gc",
             f"-ffile-prefix-map={software}=software/gemmini-rocc-tests",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [source, *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S"))]
    objects = []
    for index, path in enumerate(sources):
        obj = out / f"vpu_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(obj)],
             cwd=out, log=out / f"compile_{index}.log")
        objects.append(obj)
    elf = out / "vpu_ops.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(path) for path in objects),
          "-lm", "-lgcc", "-o", str(elf)], cwd=out, log=out / "link.log")
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = out / "libgemmini.so"
    _run(["g++", "-L", str(riscv / "lib"),
          f"-Wl,-rpath,{riscv / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(riscv / "include"), "-fPIC", "-O3",
          f"-ffile-prefix-map={extension}=software/libgemmini",
          *(str(path) for path in extension_sources)],
         cwd=out, log=out / "extension_build.log")
    result = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini",
                             str(elf)], cwd=out, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log = out / "spike.log"
    log.write_text(result.stdout)
    checks = tuple(re.findall(r"^(.+?)\s+ok \(0 mismatches\)$",
                              result.stdout, re.MULTILINE))
    passed = result.returncode == 0 and checks == EXPECTED and "vpu_ops PASSED" in result.stdout
    receipt = {
        "schema": "mx_gemmini.nicolas_vpu_source_spike.v1",
        "status": "all_29_source_vpu_checks_matched_on_pinned_spike" if passed else
                  "source_vpu_checks_failed_on_pinned_spike",
        "scope": "Nicolas handwritten vpu_ops.c with VPU_FUSED=1; source oracle only",
        "rtl_revision": _git_revision(rtl),
        "gemmini_software_revision": _git_revision(software),
        "gemmini_extension_revision": _git_revision(extension),
        "profile_sha256": profile_sha256(profile),
        "source_sha256": _sha(source), "reference_sha256": _sha(reference),
        "qualifier_sha256": _sha(Path(__file__)),
        "source_closure_sha256": _source_closure(
            software, [source, reference, *sorted(software.glob("include/*.h")),
                       *sources[1:]]),
        "extension_source_closure_sha256": _source_closure(
            extension, extension_sources + sorted(extension.rglob("*.h"))),
        "riscv_gcc_sha256": _sha(cc), "spike_sha256": _sha(spike),
        "elf_sha256": _sha(elf), "extension_sha256": _sha(so),
        "spike_log_sha256": _sha(log), "spike_exit_code": result.returncode,
        "checks": list(checks), "check_count": len(checks),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{receipt['status']}: {log}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
