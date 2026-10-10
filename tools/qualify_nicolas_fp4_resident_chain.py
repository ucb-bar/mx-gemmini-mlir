"""Compare a typed connected FP4 resident pair with Nicolas's source on Spike."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import difflib
import json
from pathlib import Path
import shutil

from mx_gemmini_support.command_ir import Command, Fence, emit_c
from mx_gemmini_support.fp4_plain_chain import (lower_fp4_plain_chain,
                                                render_fp4_plain_chain,
                                                source_resources)
from mx_gemmini_support.resident_pair_graph import INPUTS
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure
from tools.emit_resident_pair_object import _compile_object
from tools.qualify_nicolas_spad_requant_fp4 import _compile_program


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"
MARKER = "fp4 chain test PASSED (MM1 resident + reused as MM2 operand, scales reused)."
OUTPUTS = ("c1_scales", "c1_tiled_observed", "c2_scales", "c2_tiled")
BUFFERS = (*INPUTS, *OUTPUTS)


def compiler_driver(source: str) -> str:
    """Retain Nicolas's complete four-way reference comparison unchanged."""
    if (source.count(MARKER) != 1 or source.count("gemmini_loop_ws_spad(") != 2 or
            source.count('"include/matmul_fp4_64x64_chain.h"') != 1):
        raise ValueError("Nicolas FP4 source comparison or issue sites changed")
    start = source.index("  gemmini_flush(0);")
    first_check = source.index('  int c1_err = check_nibbles("C1"', start)
    second_start = source.index("  // Reuse MM1's output block-scales", first_check)
    second_check = source.index('  int c2_err = check_nibbles("C2"', second_start)
    issue = '''  static uint8_t c1_tiled_observed[2048] __attribute__((aligned(64)));
  static uint8_t c2_tiled[2048] __attribute__((aligned(64)));
  gemmini_flush(0);
  // The compiler issues both connected contractions and observes both C tiles.
  mx_issue(A_in_hw, A_scales_row, B_in, B_scales_col, B2_in,
           B2_scales_col, c1_scales, c1_tiled_observed, c2_scales, c2_tiled);
  for (int row = 0; row < 32; ++row)
    for (int col = 0; col < 64; ++col) {
      int tile = (((row / 16) * 4 + col / 16) * 16 + row % 16) * 16 + col % 16;
      C1_hw[row][col] = c1_tiled_observed[tile];
      C2_hw[row][col] = c2_tiled[tile];
    }
'''
    modified = (source[:start] + issue + source[first_check:second_start] +
                source[second_check:])
    prototype = "void mx_issue(" + ", ".join("const void *" for _ in BUFFERS) + ");\n"
    modified = modified.replace(
        '#include "include/matmul_fp4_64x64_chain.h"',
        '#include "include/matmul_fp4_64x64_chain.h"\n' + prototype, 1)
    main = modified[modified.index("int main()") :]
    if ("gemmini_loop_ws_spad(" in main or
            "gemmini_extended_mvin(" in main or
            "gemmini_extended_mvout(" in main or
            "gemmini_mx_load_scales(" in main):
        raise ValueError("compiler FP4 driver retained handwritten matrix commands")
    return modified


def _spike(spike: Path, so: Path, elf: Path, log: Path) -> dict:
    import subprocess

    result = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini",
                             str(elf)], cwd=log.parent, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            check=False)
    log.write_text(result.stdout)
    ok = (result.returncode == 0 and MARKER in result.stdout and
          "MM1 residency OK (C1 codes+scales exact)." in result.stdout and
          "MM2 chain OK (C2 codes+scales exact; C1 reused in place, scales reused)." in result.stdout)
    return {"matched": ok, "exit_code": result.returncode,
            "elf_sha256": _sha(elf), "spike_log_sha256": _sha(log)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "riscv-root", "mx-opt", "out-dir", "frontend-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    args = parser.parse_args()
    rtl, riscv, out = args.rtl_root.resolve(), args.riscv_root.resolve(), args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    software, extension = rtl / "software/gemmini-rocc-tests", rtl / "software/libgemmini"
    _require_gitlink(rtl, "software/gemmini-rocc-tests")
    _require_gitlink(rtl, "software/libgemmini")
    cc, spike = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike"
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        parser.error("RISC-V GCC, Spike, and host g++ are required")
    source = software / "bareMetalC/matmul_tiled_fp4_64x64_chain.c"
    header = software / "include/matmul_fp4_64x64_chain.h"
    profile = load_profile(args.profile, rtl_root=rtl)
    resources = source_resources(source.read_text(), header.read_text())
    frontend_dir = args.frontend_dir.resolve()
    frontend_path = frontend_dir / "nicolas_fp4_chain.profile_bound.mlir"
    manifest_path = frontend_dir / "quantization_manifest.json"
    capture_path = frontend_dir / "receipt.json"
    capture = json.loads(capture_path.read_text())
    if (capture.get("schema") != "mx_gemmini.nicolas_fp4_resident_model2mlir_capture.v1" or
            capture.get("status") != "two_site_frontend_handoff_only" or
            capture.get("source_sha256") != _sha(source) or
            capture.get("header_sha256") != _sha(header) or
            capture.get("profile_sha256") != profile_sha256(profile) or
            capture.get("bound_mlir_sha256") != _sha(frontend_path) or
            capture.get("manifest_sha256") != _sha(manifest_path)):
        raise ValueError("FP4 resident frontend capture or source provenance differs")
    frontend = frontend_path.read_text()
    manifest = json.loads(manifest_path.read_text())
    out.mkdir(parents=True)
    ir = out / "connected.mlir"
    ir.write_text(render_fp4_plain_chain(
        profile, resources, source_sha256=_sha(source), header_sha256=_sha(header),
        frontend_mlir=frontend, frontend_manifest=manifest))
    _run([str(args.mx_opt.resolve()), str(ir), "-o", "/dev/null"],
         cwd=out, log=out / "native_verify.log")
    commands = lower_fp4_plain_chain(ir.read_text(), profile, resources,
                                     frontend_mlir=frontend, frontend_manifest=manifest)
    physical = out / "physical_program.json"
    physical.write_text(json.dumps({
        "schema": "mx_gemmini.fp4_connected_resident_physical.v1",
        "profile_sha256": profile_sha256(profile),
        "commands": [({"kind": "command", **asdict(item)} if isinstance(item, Command)
                      else {"kind": "fence"}) for item in commands],
    }, indent=2, sort_keys=True) + "\n")
    issuer = out / "mx_issue.c"
    issuer.write_text(emit_c(commands, transport="rocket_rocc", buffers=BUFFERS))
    obj, data_bytes = _compile_object(out, riscv)
    driver = out / "compiler_driver.c"
    driver.write_text(compiler_driver(source.read_text()))
    (out / "compiler_driver.patch").write_text("".join(difflib.unified_diff(
        source.read_text().splitlines(keepends=True),
        driver.read_text().splitlines(keepends=True),
        fromfile=source.name, tofile=driver.name)))
    source_elf = _compile_program(out / "source", source, software, cc, None)
    compiler_elf = _compile_program(out / "compiled", driver, software, cc, obj)
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = out / "libgemmini.so"
    _run(["g++", "-L", str(riscv / "lib"),
          f"-Wl,-rpath,{riscv / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(riscv / "include"), "-fPIC", "-O3",
          f"-ffile-prefix-map={extension}=software/libgemmini",
          *(str(path) for path in extension_sources)],
         cwd=out, log=out / "extension_build.log")
    baseline = _spike(spike, so, source_elf, out / "source/spike.log")
    compiled = _spike(spike, so, compiler_elf, out / "compiled/spike.log")
    receipt = {
        "schema": "mx_gemmini.nicolas_fp4_connected_resident_spike.v1",
        "status": ("source_and_compiler_matched_on_pinned_spike" if baseline["matched"]
                   and compiled["matched"] else "source_or_compiler_failed_on_pinned_spike"),
        "source_spike": baseline, "compiler_spike": compiled,
        "compared_c1_fp4_codes": 4096, "compared_c2_fp4_codes": 4096,
        "compared_c1_e8m0_scales": 128, "compared_c2_e8m0_scales": 128,
        "source_sha256": _sha(source), "header_sha256": _sha(header),
        "frontend_mlir_sha256": _sha(frontend_path),
        "frontend_manifest_sha256": _sha(manifest_path),
        "model2mlir_revision": capture["model2mlir_revision"],
        "profile_sha256": profile_sha256(profile), "bound_mlir_sha256": _sha(ir),
        "issuer_sha256": _sha(issuer), "object_sha256": _sha(obj),
        "allocated_data_section_bytes": data_bytes,
        "physical_sha256": _sha(physical), "driver_sha256": _sha(driver),
        "compiler_revision": _git_revision(ROOT), "rtl_revision": _git_revision(rtl),
        "software_revision": _git_revision(software),
        "extension_revision": _git_revision(extension),
        "source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "spike_sha256": _sha(spike), "riscv_gcc_sha256": _sha(cc),
        "extension_sha256": _sha(so),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt["status"])
    if not baseline["matched"] or not compiled["matched"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
