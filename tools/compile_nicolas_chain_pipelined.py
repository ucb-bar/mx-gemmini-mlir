"""Compile the captured two-tile MX+VPU chain to a linkable RoCC object.

With --run-spike, link source-bound runtime buffers into a standalone program
and compare both tiles' complete codes and scales on Nicolas's pinned model.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.chain_pipelined_graph import (
    INPUTS, OUTPUTS, lower_chain_pipelined, render_chain_pipelined)
from mx_gemmini_support.chain_pipelined_source import audit_chain_pipelined
from mx_gemmini_support.full_chain_pipelined import (
    FULL_INPUTS, FULL_OUTPUTS, audit_full_chain_pipelined,
    lower_full_chain_pipelined, render_full_chain_pipelined)
from mx_gemmini_support.command_ir import Command, Fence, emit_c
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure
from tools.emit_resident_pair_object import _compile_object


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
MARKER = "lowered two-tile MX/VPU: C1 0 codes 0 scales, C2 0 codes 0 scales"
FULL_MARKER = "lowered full two-tile MX/VPU: C1 BF16 0, C1 0 codes 0 scales, C2 0 codes 0 scales"


def _driver(names: tuple[str, ...]) -> str:
    inputs = "\n".join(f"extern const uint8_t {name}[];" for name in
                       (*INPUTS, *(f"{stage}_ref_{tile}" for tile in (0, 1)
                                    for stage in ("c1_codes", "c1_scales",
                                                  "c2_codes", "c2_scales"))))
    outputs = "\n".join(
        f"static uint8_t {name}[{128 if 'scales' in name else 4096}] "
        "__attribute__((aligned(64)));" for name in OUTPUTS)
    comparisons = "\n".join(f'''
  for (uint32_t row = 0; row < 64; ++row)
    for (uint32_t col = 0; col < 64; ++col) {{
      uint32_t tiled = (((row / 16) * 4 + col / 16) * 16 + row % 16) * 16 + col % 16;
      c1_codes += c1_tiled_{tile}[tiled] != c1_codes_ref_{tile}[row * 64 + col];
      c2_codes += c2_tiled_{tile}[tiled] != c2_codes_ref_{tile}[row * 64 + col];
    }}
  for (uint32_t i = 0; i < 128; ++i) {{
    c1_scale_errors += c1_scales_{tile}[i] != c1_scales_ref_{tile}[i];
    c2_scale_errors += c2_scales_{tile}[i] != c2_scales_ref_{tile}[i];
  }}''' for tile in (0, 1))
    return f'''#include <stdint.h>
#include <stdio.h>
{inputs}
{outputs}
void mx_issue({", ".join(f"const void *{name}" for name in names)});

int main(void) {{
  mx_issue({", ".join(names)});
  int c1_codes = 0, c1_scale_errors = 0, c2_codes = 0, c2_scale_errors = 0;
{comparisons}
  printf("lowered two-tile MX/VPU: C1 %d codes %d scales, "
         "C2 %d codes %d scales\\n", c1_codes, c1_scale_errors,
         c2_codes, c2_scale_errors);
  return c1_codes || c1_scale_errors || c2_codes || c2_scale_errors;
}}
'''


def _full_driver(names: tuple[str, ...]) -> str:
    """Link source goldens separately from the six runtime input operands."""
    declared = "\n".join(f"extern const uint8_t {name}[];" for name in
                         (*FULL_INPUTS, "c1_bf16",
                          *(f"{stage}_ref_{tile}" for tile in (0, 1)
                            for stage in ("c1_codes", "c1_scales",
                                          "c2_codes", "c2_scales"))))
    outputs = "\n".join(
        f"static uint8_t {name}[{8192 if name == 'c1_bf16_observed' else 128 if 'scales' in name else 4096}] "
        "__attribute__((aligned(64)));" for name in FULL_OUTPUTS)
    comparisons = "\n".join(f'''
  for (uint32_t row = 0; row < 64; ++row)
    for (uint32_t col = 0; col < 64; ++col) {{
      uint32_t tiled = (((row / 16) * 4 + col / 16) * 16 + row % 16) * 16 + col % 16;
      c1_codes += c1_tiled_{tile}[tiled] != c1_codes_ref_{tile}[row * 64 + col];
      c2_codes += c2_tiled_{tile}[tiled] != c2_codes_ref_{tile}[row * 64 + col];
    }}
  for (uint32_t i = 0; i < 128; ++i) {{
    c1_scale_errors += c1_scales_{tile}[i] != c1_scales_ref_{tile}[i];
    c2_scale_errors += c2_scales_{tile}[i] != c2_scales_ref_{tile}[i];
  }}''' for tile in (0, 1))
    return f'''#include <stdint.h>
#include <stdio.h>
{declared}
{outputs}
void mx_issue({", ".join(f"const void *{name}" for name in names)});

int main(void) {{
  mx_issue({", ".join(names)});
  int bf16_errors = 0, c1_codes = 0, c1_scale_errors = 0;
  int c2_codes = 0, c2_scale_errors = 0;
  for (uint32_t i = 0; i < 8192; ++i)
    bf16_errors += c1_bf16_observed[i] != c1_bf16[i];
{comparisons}
  printf("lowered full two-tile MX/VPU: C1 BF16 %d, C1 %d codes %d scales, "
         "C2 %d codes %d scales\\n", bf16_errors, c1_codes,
         c1_scale_errors, c2_codes, c2_scale_errors);
  return bf16_errors || c1_codes || c1_scale_errors || c2_codes || c2_scale_errors;
}}
'''


def _run_spike(out: Path, rtl: Path, riscv: Path, obj: Path,
               resources: dict[str, bytes], names: tuple[str, ...], *,
               include_mm1: bool = False) -> dict:
    software, extension = (rtl / "software/gemmini-rocc-tests",
                           rtl / "software/libgemmini")
    _require_gitlink(rtl, "software/libgemmini")
    cc, spike = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike"
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        raise ValueError("RISC-V GCC, Spike, and host g++ are required")
    build = out / "spike_build"
    build.mkdir()
    assembly = [".section .rodata", ".balign 64"]
    for name, data in sorted(resources.items()):
        (build / f"{name}.bin").write_bytes(data)
        assembly.extend((f".globl {name}", f"{name}:",
                         f'.incbin "{name}.bin"', ".balign 64"))
    assembly.append('.section .note.GNU-stack,"",@progbits')
    (build / "mx_data.S").write_text("\n".join(assembly) + "\n")
    (build / "mx_driver.c").write_text(
        _full_driver(names) if include_mm1 else _driver(names))
    bench = software / "riscv-tests/benchmarks/common"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET",
             "-DBAREMETAL=1", "-mcmodel=medany", "-std=gnu99", "-O2",
             "-ffast-math", "-fno-common", "-fno-builtin-printf",
             "-fno-tree-loop-distribute-patterns", "-march=rv64gc",
             "-Wa,-march=rv64gc", f"-ffile-prefix-map={build.resolve()}=.",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [build / "mx_driver.c", build / "mx_data.S"]
    sources += sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))
    objects = []
    for index, source in enumerate(sources):
        target = build / f"mx_{index}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(target)],
             cwd=build, log=build / f"compile_{index}.log")
        objects.append(target)
    elf = build / "mx_program.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), str(obj), *(str(path) for path in objects),
          "-lm", "-lgcc", "-o", str(elf)], cwd=build, log=build / "link.log")
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = build / "libgemmini.so"
    _run(["g++", "-L", str(riscv / "lib"),
          f"-Wl,-rpath,{riscv / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(riscv / "include"), "-fPIC", "-O3",
          f"-ffile-prefix-map={extension}=software/libgemmini",
          *(str(path) for path in extension_sources)],
         cwd=build, log=build / "extension_build.log")
    result = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini",
                             str(elf)], cwd=build, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log = build / "spike.log"
    log.write_text(result.stdout)
    success = ("full_three_site_chain_matched_on_pinned_spike" if include_mm1 else
               "both_source_tiles_matched_on_pinned_spike")
    receipt = {
        "status": success if result.returncode == 0 and
                  (FULL_MARKER if include_mm1 else MARKER) in result.stdout else
                  "compiler_two_tile_chain_failed_on_pinned_spike",
        "spike_exit_code": result.returncode,
        "elf_sha256": _sha(elf), "extension_sha256": _sha(so),
        "spike_log_sha256": _sha(log),
        "riscv_gcc_sha256": _sha(cc), "spike_sha256": _sha(spike),
        "compared_fp8_codes": 4 * 4096,
        "compared_e8m0_scales": 4 * 128,
    }
    if include_mm1:
        receipt["compared_c1_bf16_values"] = 4096
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-dir", "rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--mx-opt", type=Path)
    parser.add_argument("--run-spike", action="store_true")
    parser.add_argument("--include-mm1", action="store_true",
                        help="issue the captured upstream MM1 and feed its BF16 result to both branches")
    parser.add_argument("--issue-schedule", choices=(
        "program_order_with_dependency_fences", "pipelined"),
        default="program_order_with_dependency_fences")
    args = parser.parse_args()
    out, rtl, riscv = (args.out_dir.resolve(), args.rtl_root.resolve(),
                       args.riscv_root.resolve())
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    software = rtl / "software/gemmini-rocc-tests"
    _require_gitlink(rtl, "software/gemmini-rocc-tests")
    profile = load_profile(args.profile, rtl_root=rtl)
    source = software / "bareMetalC/chain_pipelined.c"
    header = software / "include/matmul_fp8_64x64_chain.h"
    capture = args.capture_dir.resolve()
    captured = json.loads((capture / "receipt.json").read_text())
    frontend_path = capture / "chain_pipelined.profile_bound.mlir"
    manifest_path = capture / "quantization_manifest.json"
    graph_path = capture / "original_graph.json"
    if (captured.get("schema") != "mx_gemmini.nicolas_chain_pipelined_model2mlir_capture.v1" or
            captured.get("profile_sha256") != profile_sha256(profile) or
            captured.get("source_sha256") != _sha(source) or
            captured.get("header_sha256") != _sha(header) or
            captured.get("bound_mlir_sha256") != _sha(frontend_path) or
            captured.get("manifest_sha256") != _sha(manifest_path) or
            captured.get("original_graph_sha256") != _sha(graph_path)):
        raise ValueError("two-tile capture provenance differs from source or profile")
    if args.include_mm1:
        resources, facts = audit_full_chain_pipelined(
            source, header, software / "bareMetalC/matmul_tiled_fp8_64x64_chain.c",
            software / "bareMetalC/chain_vpu_spad_requant.c", profile)
    else:
        resources, facts = audit_chain_pipelined(source, header, profile)
    original = json.loads(graph_path.read_text())
    trace = {"graphs": {"original": original}}
    frontend, captured_manifest = (frontend_path.read_text(),
                                   json.loads(manifest_path.read_text()))
    if args.include_mm1:
        bound, preloaded = render_full_chain_pipelined(
            frontend, trace, captured_manifest, profile, resources, facts)
        chain = lower_full_chain_pipelined(
            bound, preloaded, profile, resources,
            issue_schedule=args.issue_schedule)
        required_inputs, required_outputs = FULL_INPUTS, FULL_OUTPUTS
    else:
        bound = render_chain_pipelined(
            frontend, trace, captured_manifest, profile, resources, facts)
        chain = lower_chain_pipelined(
            bound, profile, resources, issue_schedule=args.issue_schedule)
        required_inputs, required_outputs = INPUTS, OUTPUTS
    out.mkdir(parents=True)
    mlir = out / "connected.mlir"
    mlir.write_text(bound)
    if args.include_mm1:
        (out / "preloaded.mlir").write_text(preloaded)
    if args.mx_opt is not None:
        _run([str(args.mx_opt.resolve()), str(mlir), "-o", "/dev/null"],
             cwd=out, log=out / "native_verify.log")
    names = tuple(sorted({operand.buffer for command in chain.commands
                          if isinstance(command, Command)
                          for operand in (command.rs1, command.rs2)
                          if operand.buffer is not None}))
    if set(names) != set(required_inputs) | set(required_outputs):
        raise ValueError("two-tile command buffers differ from typed ABI")
    issuer = out / "mx_issue.c"
    issuer.write_text(emit_c(chain.commands, transport="rocket_rocc", buffers=names))
    (out / "mx_issue.h").write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n\n"
        "/* See object_manifest.json for buffer byte lengths and layouts. */\n"
        f"void mx_issue({', '.join(f'const void *{name}' for name in names)});\n"
        "#endif\n")
    physical = out / "physical_program.json"
    physical.write_text(json.dumps({
        "schema": ("mx_gemmini.full_chain_pipelined_physical.v1" if args.include_mm1 else
                   "mx_gemmini.chain_pipelined_physical.v1"),
        "issue_schedule": args.issue_schedule,
        "sites": chain.sites,
        "profile_sha256": profile_sha256(profile),
        "commands": [({"kind": "command", **asdict(item)} if isinstance(item, Command)
                      else {"kind": "fence"}) for item in chain.commands],
    }, indent=2, sort_keys=True) + "\n")
    obj, data_bytes = _compile_object(out, riscv)
    manifest = {
        "schema": ("mx_gemmini.full_chain_pipelined_linkable_object.v1" if args.include_mm1 else
                   "mx_gemmini.chain_pipelined_linkable_object.v1"),
        "status": "rv64_rocc_two_tile_object_built",
        "source_scope": facts["source_scope"],
        "first_matmul_scope": ("captured MM1 issued from Nicolas's packed A1/B1 inputs" if
                               args.include_mm1 else
                               "source C1 BF16 preload; captured shared MM1 site is not issued by this object"),
        "issue_schedule": args.issue_schedule,
        "source_sha256": facts["source_sha256"],
        "header_sha256": facts["header_sha256"],
        "source_resource_sha256": facts["resource_sha256"],
        "command_count": sum(isinstance(item, Command) for item in chain.commands),
        "fence_count": sum(isinstance(item, Fence) for item in chain.commands),
        "profile_sha256": profile_sha256(profile),
        "capture_receipt_sha256": _sha(capture / "receipt.json"),
        "bound_mlir_sha256": _sha(mlir),
        "runtime_input_sha256": {name: hashlib.sha256(resources[name]).hexdigest()
                                 for name in required_inputs},
        "buffer_abi": [
            {"name": name, "position": i, "role": "read" if name in required_inputs else "write",
             "minimum_bytes": (len(resources[name]) if name in required_inputs else
                               8192 if name == "c1_bf16_observed" else
                               128 if "scales" in name else 4096),
             "alignment_bytes": 64} for i, name in enumerate(names)],
        "embedded_operand_bytes": 0, "embedded_golden_bytes": 0,
        "allocated_data_section_bytes": data_bytes,
        "physical_program_sha256": _sha(physical),
        "issuer_c_sha256": _sha(issuer), "object_sha256": _sha(obj),
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "rtl_revision": _git_revision(rtl),
    }
    if args.include_mm1:
        manifest["first_source_sha256"] = facts["first_source_sha256"]
        manifest["seam_source_sha256"] = facts["seam_source_sha256"]
        manifest["preloaded_mlir_sha256"] = _sha(out / "preloaded.mlir")
    if args.run_spike:
        manifest["spike_qualification"] = _run_spike(
            out, rtl, riscv, obj, resources, names,
            include_mm1=args.include_mm1)
    (out / "object_manifest.json").write_text(json.dumps(
        manifest, indent=2, sort_keys=True) + "\n")
    print(f"compiled two-tile MX/VPU object: {obj}")
    if args.run_spike:
        print(manifest["spike_qualification"]["status"])
        success = ("full_three_site_chain_matched_on_pinned_spike" if args.include_mm1 else
                   "both_source_tiles_matched_on_pinned_spike")
        if manifest["spike_qualification"]["status"] != success:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
