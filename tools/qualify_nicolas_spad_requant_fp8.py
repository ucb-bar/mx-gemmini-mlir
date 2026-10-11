"""Compile Nicolas's typed flat+tiled FP8 SPAD_REQUANT and check its source oracle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import difflib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.command_ir import Command, Fence, emit_c
from mx_gemmini_support.dual_spad_requant import (BUFFERS, FP8_SPEC,
                                                   lower_dual_requant,
                                                   render_dual_requant)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure
from tools.emit_resident_pair_object import _compile_object
from tools.qualify_nicolas_spad_requant_fp4 import _compile_program


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json"
MARKER = "spad_requant PASSED"


def _compiler_driver(source: str) -> str:
    """Keep source data generation and comparisons; replace only accelerator issue."""
    required = (
        "#define M 32", "#define N 64",
        "#define SP_SRC   0x0000", "#define SP_FLAT  0x1000",
        "#define SP_TILED 0x2000",
        "gemmini_spad_requant(SP_FLAT, SP_SRC, M, N, 0, (uint64_t)scales_hw, 0);",
        "gemmini_spad_requant(SP_TILED, SP_SRC, M, N, 1, (uint64_t)scales_hw2, 0);",
        "mxr_quant_block(&X[m][32 * b], &codes_ref[m][32 * b], &scales_ref[m][b]);",
    )
    if any(source.count(fragment) != 1 for fragment in required):
        raise ValueError("Nicolas FP8 source geometry or oracle changed")
    anchors = ("  gemmini_flush(0);", "  int bad_flat = 0;",
               "  // ---- operand-A tiled:", "  int bad_tiled = 0;")
    if any(source.count(anchor) != 1 for anchor in anchors):
        raise ValueError("Nicolas FP8 source diagnostic anchor changed")
    start, flat_check, tiled_start, tiled_check = (source.index(anchor) for anchor in anchors)
    if not start < flat_check < tiled_start < tiled_check:
        raise ValueError("Nicolas FP8 source diagnostic ordering changed")
    flat = source[flat_check:tiled_start].replace("codes_hw[", "codes_flat_hw[")
    tiled = source[tiled_check:].replace("codes_hw[", "codes_tiled_hw[")
    modified = source[:start] + '''  memset(scales_hw, 0xa5, sizeof(scales_hw));
  memset(scales_hw2, 0xa5, sizeof(scales_hw2));
  memset(codes_flat_hw, 0xa5, sizeof(codes_flat_hw));
  memset(codes_tiled_hw, 0xa5, sizeof(codes_tiled_hw));
  uint64_t t0 = read_cycles();
  mx_issue(X, scales_hw, scales_hw2, codes_flat_hw, codes_tiled_hw);
  gemmini_fence();
  uint64_t t1 = read_cycles();
''' + flat + '''  // Operand-A tiled E4M3 image was emitted by the same compiler object.
''' + tiled
    declaration = "static uint8_t codes_hw[M * N] __attribute__((aligned(64)));"
    if modified.count(declaration) != 1:
        raise ValueError("Nicolas FP8 output buffer declaration changed")
    modified = modified.replace(
        declaration,
        "static uint8_t codes_flat_hw[M * N] __attribute__((aligned(64)));\n"
        "static uint8_t codes_tiled_hw[M * N] __attribute__((aligned(64)));", 1)
    include = '#include "include/mx_e4m3_ref.h"'
    if modified.count(include) != 1:
        raise ValueError("Nicolas FP8 reference include changed")
    modified = modified.replace(
        include, include + "\nvoid mx_issue(const void *, const void *, const void *, "
        "const void *, const void *);", 1)
    modified = modified.replace("cycles (requant+mvout)",
                                "cycles (compiler issue+readout)")
    if any(token in modified for token in (
            "gemmini_flush(", "gemmini_spad_requant(", "gemmini_extended_mvin(",
            "gemmini_extended_mvout(", "codes_hw[")):
        raise ValueError("FP8 compiler driver retained handwritten MX commands")
    return modified


def _run_spike(spike: Path, extension: Path, elf: Path, log: Path) -> dict:
    result = subprocess.run([str(spike), f"--extlib={extension}",
                             "--extension=gemmini", str(elf)],
                            cwd=log.parent, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if (result.returncode or MARKER not in result.stdout or
            "flat 0, tiled 0 code mismatches; scales 0, 0 mismatches" not in result.stdout):
        raise ValueError(f"Nicolas FP8 source oracle failed: {log}")
    return {"exit_code": result.returncode, "elf_sha256": _sha(elf),
            "spike_log_sha256": _sha(log), "compared_fp8_codes": 2 * 32 * 64,
            "compared_e8m0_scales": 2 * 32 * 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
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
            "fp8_e4m3" not in profile["candidate_output_modes"]):
        parser.error("selected profile lacks Rocket DIM16 FP8 SPAD_REQUANT")
    cc, spike = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike"
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        parser.error("RISC-V GCC, Spike, and host g++ are required")
    source = software / "bareMetalC/spad_requant.c"
    header = software / "include/mx_e4m3_ref.h"
    if not source.is_file() or not header.is_file():
        parser.error("Nicolas FP8 source or reference header is absent")
    out.mkdir(parents=True)
    original = source.read_text()
    modified = _compiler_driver(original)
    driver = out / "compiler_driver.c"
    driver.write_text(modified)
    patch = out / "compiler_driver.patch"
    patch.write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), modified.splitlines(keepends=True),
        fromfile="spad_requant.c", tofile="compiler_driver.c")))
    mlir = out / "connected.mlir"
    mlir.write_text(render_dual_requant(
        profile, source_sha256=_sha(source), header_sha256=_sha(header),
        spec=FP8_SPEC))
    _run([str(args.mx_opt.resolve()), str(mlir), "-o", "/dev/null"],
         cwd=out, log=out / "native_verify.log")
    commands = lower_dual_requant(mlir.read_text(), profile, spec=FP8_SPEC)
    physical = out / "physical_program.json"
    physical.write_text(json.dumps({
        "schema": "mx_gemmini.dual_spad_requant_physical.v1",
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
        "schema": "mx_gemmini.nicolas_fp8_spad_requant_compiler_spike.v1",
        "status": "source_and_compiled_flat_tiled_fp8_matched_on_pinned_spike",
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
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt["status"])


if __name__ == "__main__":
    main()
