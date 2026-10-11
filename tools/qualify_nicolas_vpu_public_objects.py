"""Replay Nicolas's fourteen VPU classes through the public object compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha
from tools.qualify_nicolas_vpu_elementwise import KINDS, REDUCING


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_vpu_elementwise_compiled_266c593"
FUSED_ARCHIVE = ROOT / "docs/evidence/nicolas_vpu_fused_compiled_266c593"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json"
ABI = ROOT / "examples/vpu-elementwise-abi.json"
FUSED = ("expsub", "expsum")


def _build_extension(extension: Path, riscv: Path, out: Path) -> Path:
    so = out / "libgemmini.so"
    sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc",
               *sorted((extension / "perf").rglob("*.cc"))]
    _run(["g++", "-L", str(riscv / "lib"),
          f"-Wl,-rpath,{riscv / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(riscv / "include"), "-fPIC", "-O3",
          f"-ffile-prefix-map={extension}=software/libgemmini",
          *(str(path) for path in sources)],
         cwd=out, log=out / "extension_build.log")
    return so


def _flags(software: Path, bench: Path, out: Path) -> list[str]:
    return ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET",
            "-DBAREMETAL=1", "-mcmodel=medany", "-std=gnu99", "-O2",
            "-ffast-math", "-fno-common", "-fno-builtin-printf",
            "-fno-tree-loop-distribute-patterns", "-march=rv64gc",
            "-Wa,-march=rv64gc", f"-ffile-prefix-map={out}=.",
            f"-ffile-prefix-map={software}=software/gemmini-rocc-tests",
            "-I", str(software / "riscv-tests"),
            "-I", str(software / "riscv-tests/env"),
            "-I", str(software), "-I", str(bench)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--mx-opt", type=Path)
    args = parser.parse_args()
    rtl, riscv, out = (args.rtl_root.resolve(), args.riscv_root.resolve(),
                       args.out_dir.resolve())
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    software, extension = (rtl / "software/gemmini-rocc-tests",
                           rtl / "software/libgemmini")
    _require_gitlink(rtl, "software/gemmini-rocc-tests")
    _require_gitlink(rtl, "software/libgemmini")
    profile = load_profile(PROFILE, rtl_root=rtl)
    baseline = json.loads((ARCHIVE / "index.json").read_text())
    fused_baseline = json.loads((FUSED_ARCHIVE / "index.json").read_text())
    source = software / "bareMetalC/vpu_ops.c"
    oracle = software / "include/vpu_ref.h"
    if (_git_revision(rtl) != baseline["rtl_revision"] or
            _sha(source) != baseline["source_sha256"] or
            _sha(oracle) != baseline["reference_sha256"] or
            profile_sha256(profile) != baseline["profile_sha256"] or
            {row["kind"] for row in baseline["rows"]} != set(KINDS) or
            _sha(source) != fused_baseline["source_sha256"] or
            _sha(oracle) != fused_baseline["reference_sha256"] or
            fused_baseline["profile_sha256"] != baseline["profile_sha256"] or
            {row["kind"] for row in fused_baseline["rows"]} != set(FUSED)):
        raise ValueError("Nicolas VPU baseline or selected profile differs")
    cc, spike = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike"
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        raise ValueError("Nicolas VPU replay requires RISC-V GCC, Spike, and host g++")
    bench = software / "riscv-tests/benchmarks/common"
    if not (bench / "test.ld").is_file():
        raise ValueError("Nicolas benchmark linker script is absent")
    out.mkdir(parents=True)
    so = _build_extension(extension, riscv, out)
    flags = _flags(software, bench, out)
    common = []
    for index, path in enumerate(sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))):
        obj = out / f"common_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(obj)],
             cwd=out, log=out / f"common_{index}.log")
        common.append(obj)
    by_kind = {row["kind"]: row for row in baseline["rows"]}
    by_fused_kind = {row["kind"]: row for row in fused_baseline["rows"]}
    rows = []
    for kind in (*KINDS, *FUSED):
        fused = kind in FUSED
        archived = (FUSED_ARCHIVE if fused else ARCHIVE) / kind
        bound, driver = archived / "bound.mlir", archived / "mx_driver.c"
        row = (by_fused_kind if fused else by_kind)[kind]
        if (_sha(bound) != row["bound_mlir_sha256"] or
                _sha(driver) != row["driver_sha256"] or
                _sha(archived / "mx_issue.c") != row["issuer_sha256"]):
            raise ValueError(f"Nicolas {kind} captured input differs")
        directory = out / kind
        directory.mkdir()
        object_dir = directory / "object"
        abi = (ROOT / f"examples/vpu-{kind}-abi.json") if fused else ABI
        command = [sys.executable, "-m", "tools.compile_object",
                   "--mlir", str(bound), "--profile", str(PROFILE),
                   "--rtl-root", str(rtl), "--riscv-root", str(riscv),
                   "--abi-json", str(abi), "--out-dir", str(object_dir)]
        if args.mx_opt is not None:
            command += ["--mx-opt", str(args.mx_opt.resolve())]
        env = {**os.environ, "TMPDIR": str(out)}
        run = subprocess.run(command, cwd=ROOT, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (directory / "compiler.log").write_text(run.stdout)
        if run.returncode:
            raise RuntimeError(f"public {kind} object compilation failed")
        manifest_path = object_dir / "object_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        issuer_equal_baseline = _sha(object_dir / "mx_issue.c") == row["issuer_sha256"]
        if (manifest["vpu_kind"] != kind or
                manifest["allocated_data_section_bytes"] != 0 or
                manifest["embedded_operand_bytes"] != 0 or
                manifest["embedded_golden_bytes"] != 0 or
                (not fused and not issuer_equal_baseline) or
                manifest["object_sha256"] != _sha(object_dir / "mx_issue.o")):
            raise ValueError(f"public {kind} object differs from qualified issuer")
        if fused and issuer_equal_baseline:
            raise ValueError(f"public {kind} should record its narrower broadcast transfer")
        linked_driver = driver
        if kind == "expsub":
            # Nicolas's fused driver passes an unused fourth pointer. Bind its
            # unchanged source oracle to the public object's exact three-slot ABI.
            original = driver.read_text()
            declaration = ("void mx_issue(const void *a, const void *b, "
                           "const void *output, const void *sums);")
            call = "mx_issue(a, b, output, sums);"
            if original.count(declaration) != 1 or original.count(call) != 1:
                raise ValueError("Nicolas EXPSUB host ABI differs from pinned source")
            adapted = original.replace(
                declaration,
                "void mx_issue(const void *a, const void *b, const void *output);")
            adapted = adapted.replace(call, "mx_issue(a, b, output);")
            linked_driver = directory / "mx_driver.c"
            linked_driver.write_text(adapted)
        driver_obj = directory / "driver.o"
        _run([str(cc), *flags, "-c", str(linked_driver), "-o", str(driver_obj)],
             cwd=directory, log=directory / "driver_compile.log")
        elf = directory / "vpu.elf"
        _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
              str(bench / "test.ld"), str(object_dir / "mx_issue.o"),
              str(driver_obj), *(str(path) for path in common),
              "-lm", "-lgcc", "-o", str(elf)],
             cwd=directory, log=directory / "link.log")
        result = subprocess.run(
            [str(spike), f"--extlib={so}", "--extension=gemmini", str(elf)],
            cwd=directory, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        log = directory / "spike.log"
        log.write_text(result.stdout)
        compared = (16 if kind in REDUCING else 64) * 8
        expected = (f"compiled fused {kind}: 0 output mismatches, 0 sum mismatches"
                    if fused else f"compiled VPU {kind}: 0 mismatches")
        if (result.returncode or expected not in result.stdout or
                row["compared_output_bf16"] != compared or
                (fused and row["compared_sum_bf16"] !=
                 (128 if kind == "expsum" else 0))):
            raise RuntimeError(f"public {kind} object differs from source oracle")
        rows.append({
            "kind": kind, "status": "public_object_source_vpu_reference_matched_on_pinned_spike",
            "source_sha256": baseline["source_sha256"],
            "bound_mlir_sha256": _sha(bound), "driver_sha256": _sha(driver),
            "linked_driver_sha256": _sha(linked_driver),
            "baseline_issuer_sha256": row["issuer_sha256"],
            "issuer_c_sha256": _sha(object_dir / "mx_issue.c"),
            "issuer_equal_baseline": issuer_equal_baseline,
            "object_sha256": _sha(object_dir / "mx_issue.o"),
            "object_manifest_sha256": _sha(manifest_path),
            "physical_program_sha256": _sha(object_dir / "physical_program.json"),
            "elf_sha256": _sha(elf), "spike_log_sha256": _sha(log),
            "compared_output_bf16": compared,
            "compared_sum_bf16": 128 if kind == "expsum" else 0,
            "mismatches": 0,
        })
    report = {
        "schema": "mx_gemmini.nicolas_vpu_public_object_replay.v1",
        "status": "all_base_and_fused_vpu_public_objects_match_source_reference",
        "rtl_revision": _git_revision(rtl),
        "software_revision": _git_revision(software),
        "compiler_revision": _git_revision(ROOT),
        "baseline_index_sha256": _sha(ARCHIVE / "index.json"),
        "fused_baseline_index_sha256": _sha(FUSED_ARCHIVE / "index.json"),
        "profile_sha256": profile_sha256(profile),
        "extension_sha256": _sha(so), "spike_sha256": _sha(spike),
        "riscv_gcc_sha256": _sha(cc),
        "total_output_bf16_compared": sum(row["compared_output_bf16"] for row in rows),
        "total_sum_bf16_compared": sum(row["compared_sum_bf16"] for row in rows),
        "rows": rows,
    }
    (out / "index.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(out / "index.json")


if __name__ == "__main__":
    main()
