"""Compile typed flat+tiled FP4 SPAD_REQUANT and compare Nicolas's source oracle.

The diagnostic driver retains Nicolas's input generator and every reference
comparison. Only its accelerator issue sites are replaced by a call into the
data-free object emitted from the typed MLIR program.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import difflib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.command_ir import Command, Fence, emit_c
from mx_gemmini_support.fp4_dual_requant import BUFFERS, lower_fp4_dual_requant, render_fp4_dual_requant
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure
from tools.emit_resident_pair_object import _compile_object


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
MARKER = "spad_requant_fp4 PASSED"


def _replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError("Nicolas FP4 source diagnostic anchor changed")
    return text.replace(before, after, 1)


def _index_once(text: str, anchor: str, start: int = 0) -> int:
    if text.count(anchor) != 1:
        raise ValueError("Nicolas FP4 source diagnostic anchor changed")
    return text.index(anchor, start)


def _compiler_driver(source: str) -> str:
    """Preserve the source input and golden code, replacing only issue sites."""
    required = (
        "#define M 64", "#define N 128",
        "#define SP_SRC   0x0000", "#define SP_FLAT  0x1000",
        "#define SP_TILED 0x2000",
        "gemmini_spad_requant_fp4(SP_FLAT, SP_SRC, M, N, 0, (uint64_t)scales_hw, 0);",
        "gemmini_spad_requant_fp4(SP_TILED, SP_SRC, M, N, 1, (uint64_t)scales_hw2, 0);",
        "uint8_t sc = mxr_scale(&X[m][32 * b], 32);",
        "codes_ref[m][32 * b + k] = fp4_code((uint16_t)(u >> 16));",
    )
    if any(source.count(fragment) != 1 for fragment in required):
        raise ValueError("Nicolas FP4 source geometry or reference changed")
    start = _index_once(source, "  gemmini_flush(0);")
    flat_check = _index_once(source, "  int bad_flat = 0;", start)
    second = _index_once(source, "  // ---- FP4 operand-A tiled:", flat_check)
    tiled_check = _index_once(source, "  int bad_tiled = 0;", second)
    if (source.count("  gemmini_flush(0);") != 1 or
            source.count("  int bad_flat = 0;") != 1 or
            source.count("  int bad_tiled = 0;") != 1 or
            source.count("gemmini_spad_requant_fp4(") != 2 or
            source.count("codes_hw[") != 3):
        raise ValueError("Nicolas FP4 source structure changed")
    flat = source[flat_check:second].replace("codes_hw[", "codes_flat_hw[")
    tiled = source[tiled_check:].replace("codes_hw[", "codes_tiled_hw[")
    modified = source[:start] + '''  gemmini_flush(0);
  memset(scales_hw, 0xa5, sizeof(scales_hw));
  memset(scales_hw2, 0xa5, sizeof(scales_hw2));
  memset(codes_flat_hw, 0xa5, sizeof(codes_flat_hw));
  memset(codes_tiled_hw, 0xa5, sizeof(codes_tiled_hw));
  uint64_t t0 = read_cycles();
  mx_issue(X, scales_hw, scales_hw2, codes_flat_hw, codes_tiled_hw);
  gemmini_fence();
  uint64_t t1 = read_cycles();
''' + flat + '''  // Operand-A tiled FP4 image was emitted by the same compiler object.
''' + tiled
    modified = _replace_once(
        modified,
        "static uint8_t codes_hw[M * N / 2] __attribute__((aligned(64)));",
        "static uint8_t codes_flat_hw[M * N / 2] __attribute__((aligned(64)));\n"
        "static uint8_t codes_tiled_hw[M * N / 2] __attribute__((aligned(64)));" )
    modified = _replace_once(modified, '#include "include/mx_e4m3_ref.h"',
                             '#include "include/mx_e4m3_ref.h"\n'
                             'void mx_issue(const void *, const void *, const void *, '
                             'const void *, const void *);')
    modified = _replace_once(modified, "cycles (requant+mvout)",
                             "cycles (compiler issue+readout)")
    if ("gemmini_spad_requant_fp4(" in modified or
            "gemmini_extended_mvin(" in modified or
            "gemmini_extended_mvout(" in modified or
            "codes_hw[" in modified):
        raise ValueError("compiler driver retained handwritten MX issue commands")
    return modified


def _compile_program(out: Path, source: Path, software: Path, cc: Path,
                     issuer: Path | None) -> Path:
    out.mkdir(parents=True)
    bench = software / "riscv-tests/benchmarks/common"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DSPIKE_SIM",
             "-DBAREMETAL=1", "-mcmodel=medany", "-std=gnu99", "-O2",
             "-ffast-math", "-fno-common", "-fno-builtin-printf",
             "-fno-tree-loop-distribute-patterns", "-march=rv64gc",
             "-Wa,-march=rv64gc",
             f"-ffile-prefix-map={software}=software/gemmini-rocc-tests",
             f"-ffile-prefix-map={out.resolve()}=.",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [source, *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S"))]
    objects = []
    for index, path in enumerate(sources):
        target = out / f"program_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(target)],
             cwd=out, log=out / f"compile_{index}.log")
        objects.append(target)
    elf = out / "program.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(path) for path in objects),
          *([str(issuer)] if issuer else []), "-lm", "-lgcc", "-o", str(elf)],
         cwd=out, log=out / "link.log")
    return elf


def _run_spike(spike: Path, extension: Path, elf: Path, log: Path) -> dict:
    result = subprocess.run([str(spike), f"--extlib={extension}",
                             "--extension=gemmini", str(elf)],
                            cwd=log.parent, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if (result.returncode or MARKER not in result.stdout or
            "flat 0, tiled 0 code mismatches; scales 0, 0 mismatches" not in result.stdout):
        raise ValueError(f"Nicolas FP4 source oracle failed: {log}")
    return {"exit_code": result.returncode, "elf_sha256": _sha(elf),
            "spike_log_sha256": _sha(log), "compared_fp4_codes": 2 * 64 * 128,
            "compared_e8m0_scales": 2 * 64 * 4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--baseline-receipt", type=Path,
                        help="require a fresh build to reproduce every recorded digest")
    args = parser.parse_args()
    rtl, riscv, out = args.rtl_root.resolve(), args.riscv_root.resolve(), args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    software, extension = (rtl / "software/gemmini-rocc-tests",
                           rtl / "software/libgemmini")
    _require_gitlink(rtl, "software/gemmini-rocc-tests")
    _require_gitlink(rtl, "software/libgemmini")
    profile = load_profile(args.profile, rtl_root=rtl)
    if (profile["transport"] != "rocket_rocc" or
            profile["geometry"]["mesh_columns"] != 16 or
            not profile["resources"].get("spad_requant") or
            "fp4_e2m1" not in profile["candidate_output_modes"]):
        parser.error("selected profile lacks Rocket DIM16 FP4 SPAD_REQUANT")
    cc, spike = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike"
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        parser.error("RISC-V GCC, Spike, and host g++ are required")
    source = software / "bareMetalC/spad_requant_fp4.c"
    header = software / "include/mx_e4m3_ref.h"
    if not source.is_file() or not header.is_file():
        parser.error("Nicolas FP4 source or reference header is absent")
    out.mkdir(parents=True)
    original = source.read_text()
    modified = _compiler_driver(original)
    driver = out / "compiler_driver.c"
    driver.write_text(modified)
    patch = out / "compiler_driver.patch"
    patch.write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), modified.splitlines(keepends=True),
        fromfile="spad_requant_fp4.c", tofile="compiler_driver.c")))
    mlir = out / "connected.mlir"
    mlir.write_text(render_fp4_dual_requant(
        profile, source_sha256=_sha(source), header_sha256=_sha(header)))
    _run([str(args.mx_opt.resolve()), str(mlir), "-o", "/dev/null"],
         cwd=out, log=out / "native_verify.log")
    commands = lower_fp4_dual_requant(mlir.read_text(), profile)
    physical = out / "physical_program.json"
    physical.write_text(json.dumps({
        "schema": "mx_gemmini.fp4_dual_spad_requant_physical.v1",
        "profile_sha256": profile_sha256(profile),
        "commands": [({"kind": "command", **asdict(item)} if isinstance(item, Command)
                      else {"kind": "fence"}) for item in commands],
    }, indent=2, sort_keys=True) + "\n")
    issuer = out / "mx_issue.c"
    issuer.write_text(emit_c(commands, transport="rocket_rocc", buffers=BUFFERS))
    obj, data_bytes = _compile_object(out, riscv)
    baseline_elf = _compile_program(out / "source", source, software, cc, None)
    compiled_elf = _compile_program(out / "compiled", driver, software, cc, obj)
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = out / "libgemmini.so"
    _run(["g++", "-L", str(riscv / "lib"),
          f"-Wl,-rpath,{riscv / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(riscv / "include"), "-fPIC", "-O3",
          f"-ffile-prefix-map={extension}=software/libgemmini",
          *(str(path) for path in extension_sources)],
         cwd=out, log=out / "extension_build.log")
    source_result = _run_spike(spike, so, baseline_elf, out / "source/spike.log")
    compiled_result = _run_spike(spike, so, compiled_elf, out / "compiled/spike.log")
    receipt = {
        "schema": "mx_gemmini.nicolas_fp4_spad_requant_compiler_spike.v1",
        "status": "source_and_compiled_flat_tiled_fp4_matched_on_pinned_spike",
        "scope": "Nicolas source input and per-element oracle; compiler-issued flat and tiled SPAD_REQUANT",
        "compiler_revision": _git_revision(ROOT), "rtl_revision": _git_revision(rtl),
        "software_revision": _git_revision(software),
        "extension_revision": _git_revision(extension),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "source_sha256": _sha(source), "reference_header_sha256": _sha(header),
        "driver_sha256": _sha(driver), "driver_patch_sha256": _sha(patch),
        "profile_sha256": profile_sha256(profile),
        "connected_mlir_sha256": _sha(mlir),
        "physical_program_sha256": _sha(physical),
        "issuer_c_sha256": _sha(issuer), "object_sha256": _sha(obj),
        "allocated_data_section_bytes": data_bytes,
        "command_count": sum(isinstance(item, Command) for item in commands),
        "spike_sha256": _sha(spike), "riscv_gcc_sha256": _sha(cc),
        "extension_sha256": _sha(so),
        "source_spike": source_result, "compiler_spike": compiled_result,
    }
    if (args.baseline_receipt and
            receipt != json.loads(args.baseline_receipt.read_text())):
        raise ValueError("FP4 source/compiler Spike run differs from baseline receipt")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt["status"])


if __name__ == "__main__":
    main()
