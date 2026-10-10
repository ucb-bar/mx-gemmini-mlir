"""Run Nicolas's three-schedule chain_pipelined.c oracle on pinned Spike.

This is a handwritten source baseline. It does not qualify compiler output or
RTL cycle overlap; those require the typed branch lowering and RTL execution.
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
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
EXPECTED = ("warmup", "fenced", "program", "pipelined")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    args = parser.parse_args()
    rtl, riscv, out = (args.rtl_root.resolve(), args.riscv_root.resolve(),
                       args.out_dir.resolve())
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    software, extension = (rtl / "software/gemmini-rocc-tests",
                           rtl / "software/libgemmini")
    _require_gitlink(rtl, "software/gemmini-rocc-tests")
    _require_gitlink(rtl, "software/libgemmini")
    profile = load_profile(args.profile, rtl_root=rtl)
    if (profile["transport"] != "rocket_rocc" or
            profile["geometry"]["mesh_columns"] != 16 or
            not profile["resources"].get("vpu") or
            not profile["resources"].get("spad_requant") or
            "fp8_e4m3" not in profile["candidate_output_modes"]):
        parser.error("selected profile cannot run Nicolas's FP8 MX+VPU chain")
    cc, spike = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike"
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        parser.error("RISC-V GCC, Spike, and host g++ are required")
    source = software / "bareMetalC/chain_pipelined.c"
    header = software / "include/matmul_fp8_64x64_chain.h"
    bench = software / "riscv-tests/benchmarks/common"
    if not source.is_file() or not header.is_file() or not (bench / "test.ld").is_file():
        parser.error("source, data header, or benchmark linker script is absent")
    out.mkdir(parents=True)
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DSPIKE_SIM",
             "-DBAREMETAL=1", "-mcmodel=medany", "-std=gnu99", "-O2",
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
        obj = out / f"chain_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(obj)],
             cwd=out, log=out / f"compile_{index}.log")
        objects.append(obj)
    elf = out / "chain_pipelined.elf"
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
    checks = tuple(re.findall(r"^(warmup|fenced|program|pipelined)\s+check: ok \((\d+) clamped scales skipped\)$",
                              result.stdout, re.MULTILINE))
    passed = (result.returncode == 0 and
              tuple(name for name, _ in checks) == EXPECTED and
              "chain_pipelined PASSED" in result.stdout)
    receipt = {
        "schema": "mx_gemmini.nicolas_chain_pipelined_source_spike.v1",
        "status": "four_source_checks_matched_on_pinned_spike" if passed else
                  "source_chain_pipelined_failed_on_pinned_spike",
        "scope": "Nicolas handwritten two-tile driver; functional source oracle and cycle numbers, no compiler or RTL overlap claim",
        "rtl_revision": _git_revision(rtl),
        "software_revision": _git_revision(software),
        "extension_revision": _git_revision(extension),
        "profile_sha256": profile_sha256(profile),
        "source_sha256": _sha(source), "header_sha256": _sha(header),
        "qualifier_sha256": _sha(Path(__file__)),
        "source_closure_sha256": _source_closure(
            software, [source, header, *sorted(software.glob("include/*.h")),
                       *sources[1:]]),
        "extension_source_closure_sha256": _source_closure(
            extension, extension_sources + sorted(extension.rglob("*.h"))),
        "riscv_gcc_sha256": _sha(cc), "spike_sha256": _sha(spike),
        "elf_sha256": _sha(elf), "extension_sha256": _sha(so),
        "spike_log_sha256": _sha(log), "spike_exit_code": result.returncode,
        "checks": [{"schedule": name, "clamped_scales_skipped": int(count)}
                   for name, count in checks],
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{receipt['status']}: {log}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
