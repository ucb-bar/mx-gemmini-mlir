"""Replay Nicolas's MX memory benchmark with compiler-issued DMA phases.

The host driver keeps the source's counters, reporting, and spot check. Its
Gemmini issue sites are replaced by functions emitted from the physical MX
transfer planner. An instrumented source run and compiler run dump the entire
16 KiB mvout buffer for exact comparison; cycle counters are reported, not
treated as timing parity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.memory_phase import (
    lower_linear_spad_mvout, lower_matrix_mvin, lower_scale_load,
)
from mx_gemmini_support.physical_program import _cmd
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure
from tools.qualify_nicolas_spad_requant_fp4 import _compile_program


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"
SOURCE_SHA = "d23c749184ede0db1cb0b4ee67f3e29d8d80ddee4ddf589dd949fc7c40a3b922"
HEADER_SHA = "16241671c4df2d4f738e77225063caac4895cdcba4d4495941e599d0db185bcd"
PHASES = ("A_cold_64B", "A_warm_64B", "B_cold_16B", "B_warm_16B",
          "scale_cold", "scale_warm", "mvout_16B")
DUMP = '''  printf("MX_MEM_DUMP_BEGIN\\n");
  for (int byte = 0; byte < N * N; byte++)
    printf("%02x", out_buf[byte]);
  printf("\\nMX_MEM_DUMP_END\\n");
'''


def _replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError("Nicolas memory benchmark source structure changed")
    return text.replace(before, after, 1)


def _instrument(source: str) -> str:
    return _replace_once(source, "  return errors != 0;", DUMP + "  return errors != 0;")


def _compiler_driver(source: str) -> str:
    """Keep the original reporting while replacing every source DMA site."""
    if any(source.count(fragment) != 1 for fragment in (
            "  gemmini_config_ld(N);", "  gemmini_mx_load_scales((uint64_t)&A_scales_row,",
            "  gemmini_config_st(DIM * sizeof(uint8_t));", "  gemmini_flush(0);")):
        raise ValueError("Nicolas memory benchmark issue sites changed")
    decls = ("\nvoid mx_issue_setup(void);\n"
             "void mx_issue_a64(const void *src);\n"
             "void mx_issue_b16(const void *src);\n"
             "void mx_issue_scale(const void *src);\n"
             "void mx_issue_mvout(const void *dst);\n")
    source = _replace_once(source, '#include "include/mx_perf.h"',
                           '#include "include/mx_perf.h"' + decls)
    start = source.index("  gemmini_config_ld(N);", source.index("static void mvin_matrix("))
    end = source.index("  phase_end(p, t0);", start)
    source = source[:start] + (
        "  if (t == 4 && sp_base == 0) mx_issue_a64(src);\n"
        "  else if (t == 1 && sp_base == 1024) mx_issue_b16(src);\n"
        "  else return;\n") + source[end:]
    source = _replace_once(
        source, "  gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);",
        "  mx_issue_scale(&A_scales_row);")
    source = _replace_once(
        source,
        "  gemmini_flush(0);\n"
        "  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, 3, 0);",
        "  mx_issue_setup();")
    start = source.index("    gemmini_config_st(DIM * sizeof(uint8_t));")
    end = source.index("    phase_end(p, t0);", start)
    source = source[:start] + "    mx_issue_mvout(out_buf);\n" + source[end:]
    if any(token in source for token in (
            "gemmini_extended_mvin(", "gemmini_extended_mvout(",
            "gemmini_mx_load_scales(", "gemmini_config_ld(",
            "gemmini_config_st(", "gemmini_flush(",
            "gemmini_extended3_config_ex(")):
        raise ValueError("compiler benchmark driver retained handwritten MX issue sites")
    return source


def _phase_commands(profile: dict) -> dict[str, tuple[Command, ...]]:
    a = lower_matrix_mvin(profile, buffer="src", matrix_rows=128,
                          matrix_cols=128, burst_cols=64, spad_row=0)
    b = lower_matrix_mvin(profile, buffer="src", matrix_rows=128,
                          matrix_cols=128, burst_cols=16, spad_row=1024)
    scale = lower_scale_load(profile, buffer="src", payload_bytes=512,
                             operand="activation")
    output = lower_linear_spad_mvout(profile, buffer="dst",
                                     total_bytes=16384, tile_cols=16, spad_row=0)
    if [(p.transferred_bytes, p.row_requests) for p in (a, b, scale, output)] != [
            (16384, 256), (16384, 1024), (512, 64), (16384, 1024)]:
        raise ValueError("compiler memory phases differ from source phase geometry")
    return {
        "setup": (_cmd(7, 0, 0),
                  _cmd(0, (1 << 16) | (3 << 14) | (1 << 2), 1 << 48)),
        "a64": a.commands, "b16": b.commands,
        "scale": scale.commands, "mvout": output.commands,
    }


def _compile_issuer(out: Path, riscv: Path, phases: dict) -> Path:
    names = {"setup": (), "a64": ("src",), "b16": ("src",),
             "scale": ("src",), "mvout": ("dst",)}
    c = "\n".join(emit_c(list(commands), transport="rocket_rocc",
                         buffers=names[name], symbol=f"mx_issue_{name}")
                  for name, commands in phases.items())
    source = out / "mx_issue.c"
    source.write_text(c)
    obj = out / "mx_issue.o"
    cc = riscv / "bin/riscv64-unknown-elf-gcc"
    nm = riscv / "bin/riscv64-unknown-elf-nm"
    readelf = riscv / "bin/riscv64-unknown-elf-readelf"
    _run([str(cc), "-std=gnu99", "-O2", "-ffreestanding", "-fno-common",
          "-mcmodel=medany", "-march=rv64gc", "-Wa,-march=rv64gc",
          "-c", str(source), "-o", str(obj)], cwd=out, log=out / "issuer_compile.log")
    defined = subprocess.check_output(
        [str(nm), "-g", "--defined-only", str(obj)], text=True).splitlines()
    symbols = {line.split()[-1] for line in defined}
    if symbols != {f"mx_issue_{name}" for name in names} or len(defined) != 5 or \
            subprocess.check_output([str(nm), "-u", str(obj)], text=True).strip():
        raise ValueError("generated memory object has unexpected symbols")
    sections = subprocess.check_output([str(readelf), "-SW", str(obj)], text=True)
    for line in sections.splitlines():
        match = re.match(r"^\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+[0-9a-f]+\s+"
                         r"[0-9a-f]+\s+([0-9a-f]+)\s+", line)
        if match and match.group(1).startswith((".data", ".bss", ".rodata",
                                                ".sdata", ".sbss")) and int(match.group(2), 16):
            raise ValueError("generated memory object embeds runtime data")
    return obj


def _run_spike(spike: Path, so: Path, elf: Path, log: Path) -> tuple[int, str]:
    result = subprocess.run([str(spike), f"--extlib={so}",
                             "--extension=gemmini", str(elf)],
                            cwd=log.parent, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    return result.returncode, result.stdout


def _dump(log: str) -> bytes:
    match = re.search(r"MX_MEM_DUMP_BEGIN\n([0-9a-f]{32768})\nMX_MEM_DUMP_END", log)
    if match is None:
        raise ValueError("memory benchmark did not dump its complete 16 KiB output")
    return bytes.fromhex(match.group(1))


def _phase_report(log: str) -> list[tuple[str, int, int]]:
    matches = re.findall(r"^MEMBW (\S+) bytes=(\d+) cyc=\d+ .*? reqs=(\d+) ", log, re.M)
    result = [(name, int(size), int(requests)) for name, size, requests in matches]
    if [name for name, _, _ in result] != list(PHASES):
        raise ValueError("memory benchmark phase order differs from Nicolas source")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--riscv-root", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    rtl, riscv, out = args.rtl_root.resolve(), args.riscv_root.resolve(), args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    _require_gitlink(rtl, "software/gemmini-rocc-tests")
    _require_gitlink(rtl, "software/libgemmini")
    profile = load_profile(args.profile, rtl_root=rtl)
    if profile["name"] != "MxGemminiRocketConfig":
        parser.error("Nicolas memory benchmark needs the pinned MX-only profile")
    software, extension = (rtl / "software/gemmini-rocc-tests",
                           rtl / "software/libgemmini")
    source = software / "bareMetalC/mx_mem_bw.c"
    header = software / "include/matmul_fp8_128x128.h"
    if _sha(source) != SOURCE_SHA or _sha(header) != HEADER_SHA:
        parser.error("Nicolas memory benchmark source or header changed")
    cc, spike = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike"
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        parser.error("RISC-V GCC, Spike, and host g++ are required")
    original = source.read_text()
    instrumented = _instrument(original)
    compiled = _instrument(_compiler_driver(original))
    phases = _phase_commands(profile)
    out.mkdir(parents=True)
    (out / "source_full.c").write_text(instrumented)
    (out / "compiler_driver.c").write_text(compiled)
    (out / "compiler_driver.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), compiled.splitlines(keepends=True),
        fromfile="Nicolas/mx_mem_bw.c", tofile="compiler_driver.c")))
    (out / "physical_program.json").write_text(json.dumps({
        "schema": "mx_gemmini.nicolas_memory_phases.v1",
        "profile_sha256": profile_sha256(profile),
        "phases": {name: [asdict(command) for command in commands]
                   for name, commands in phases.items()},
    }, indent=2, sort_keys=True) + "\n")
    obj = _compile_issuer(out, riscv, phases)
    source_elf = _compile_program(out / "source", source, software, cc, None)
    full_elf = _compile_program(out / "source_full", out / "source_full.c", software, cc, None)
    compiled_elf = _compile_program(out / "compiled", out / "compiler_driver.c",
                                    software, cc, obj)
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = out / "libgemmini.so"
    _run(["g++", "-L", str(riscv / "lib"),
          f"-Wl,-rpath,{riscv / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(riscv / "include"), "-fPIC", "-O3",
          f"-ffile-prefix-map={extension}=software/libgemmini",
          *(str(path) for path in extension_sources)],
         cwd=out, log=out / "extension_build.log")
    baseline_code, baseline_log = _run_spike(
        spike, so, source_elf, out / "source/spike.log")
    source_code, source_log = _run_spike(
        spike, so, full_elf, out / "source_full/spike.log")
    compiled_code, compiled_log = _run_spike(
        spike, so, compiled_elf, out / "compiled/spike.log")
    if any(code != 0 for code in (baseline_code, source_code, compiled_code)) or \
            any("mx_mem_bw PASSED (0 mismatches in spot check)" not in log
                for log in (baseline_log, source_log, compiled_log)):
        raise ValueError("Nicolas memory benchmark source or compiler run failed")
    phase_expected = [
        ("A_cold_64B", 16384, 256), ("A_warm_64B", 16384, 256),
        ("B_cold_16B", 16384, 1024), ("B_warm_16B", 16384, 1024),
        ("scale_cold", 512, 64), ("scale_warm", 512, 64),
        ("mvout_16B", 16384, 1024)]
    if any(_phase_report(log) != phase_expected
           for log in (baseline_log, source_log, compiled_log)):
        raise ValueError("Nicolas memory benchmark phase geometry differs")
    source_output = _dump(source_log)
    compiled_output = _dump(compiled_log)
    if source_output != compiled_output:
        raise ValueError("compiler memory DMA output differs from Nicolas source")
    receipt = {
        "schema": "mx_gemmini.nicolas_mem_bw_physical_object_spike.v1",
        "status": "source_memory_phases_and_full_mvout_matched_on_pinned_spike",
        "scope": "seven MX memory phases; complete 16 KiB mvout bytes; timing and counters not qualified",
        "source_sha256": SOURCE_SHA, "header_sha256": HEADER_SHA,
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, [ROOT / "mx_gemmini_support/memory_phase.py",
                   ROOT / "mx_gemmini_support/command_ir.py",
                   ROOT / "tools/qualify_nicolas_mem_bw.py"]),
        "rtl_revision": _git_revision(rtl),
        "software_revision": _git_revision(software),
        "extension_revision": _git_revision(extension),
        "profile_sha256": profile_sha256(profile),
        "physical_program_sha256": _sha(out / "physical_program.json"),
        "issuer_c_sha256": _sha(out / "mx_issue.c"),
        "object_sha256": _sha(obj),
        "source_elf_sha256": _sha(source_elf),
        "source_full_elf_sha256": _sha(full_elf),
        "compiled_elf_sha256": _sha(compiled_elf),
        "source_spike_log_sha256": _sha(out / "source/spike.log"),
        "source_full_spike_log_sha256": _sha(out / "source_full/spike.log"),
        "compiled_spike_log_sha256": _sha(out / "compiled/spike.log"),
        "source_full_output_sha256": hashlib.sha256(source_output).hexdigest(),
        "compiled_output_sha256": hashlib.sha256(compiled_output).hexdigest(),
        "compared_mvout_bytes": 16384,
        "phase_count": 7,
        "source_exit_code": baseline_code,
        "source_full_exit_code": source_code,
        "compiled_exit_code": compiled_code,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{receipt['status']}: 7 phases, 16384 output bytes")


if __name__ == "__main__":
    main()
