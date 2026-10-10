"""Compile Nicolas's chained VPU and memory-ordering checks for Spike.

The PyTorch captures identify the arithmetic graph. Source-bound scheduling
places transfers and typed VPU commands in the same order as vpu_ops.c. The
C driver only generates source inputs and evaluates Nicolas's BF16 reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.command_ir import VPU_OPCODES, emit_c
from mx_gemmini_support.physical_program import _cmd, _config_ld, _config_st, _transfer
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vector_lowering import lower_vector_commands
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure
from tools.qualify_nicolas_vpu_fused import CONTRACT, PROFILE, SOURCE_RECEIPT


ROOT = Path(__file__).resolve().parents[1]
SP_A, SP_B, SP_D, SP_R = 0, 0x1000, 0x2000, 0x3000
CASES = ("chain", "war", "dual")
SOURCE_CHECKS = {"chain": ("chain",), "war": ("war mvin",),
                 "dual": ("dual X,Y", "dual Z=X*B")}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def capture(case: str, m2m, torch) -> str:
    class Model(torch.nn.Module):
        def forward(self, a, b, x):
            if case == "chain":
                return ((a + b) * 0.5).reshape(16, 4, 8).amax(dim=(1, 2))
            if case == "war":
                return a + b
            if case == "dual":
                y = a + 3.0
                return y, torch.exp(x), y * b
            raise ValueError(case)

    sample = torch.zeros((64, 8), dtype=torch.bfloat16)
    result = m2m.convert(Model().eval(), (sample, sample, sample), backend="fx_importer")
    if not result.ok:
        raise ValueError(f"model2MLIR {case} capture failed: {result.diagnostics}")
    from m2m.coverage import opaque_report
    frontend = result.mlir_text
    expected_return = {"chain": "tensor<16xbf16>",
                       "war": "tensor<64x8xbf16>",
                       "dual": "(tensor<64x8xbf16>, tensor<64x8xbf16>, tensor<64x8xbf16>)"}[case]
    signature = ("func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>, "
                 f"%2: tensor<64x8xbf16>) -> {expected_return}")
    required = {"chain": ("arith.addf", "arith.mulf", "arith.maximumf",
                          "linalg.reduce", "dimensions = [1, 2]", "5.000000e-01"),
                "war": ("arith.addf",),
                "dual": ("arith.addf", "math.exp", "arith.mulf", "3.000000e+00")}[case]
    forbidden = {"chain": ("math.exp",),
                 "war": ("arith.mulf", "linalg.reduce", "math.exp"),
                 "dual": ("linalg.reduce",)}[case]
    if (opaque_report(frontend) or "torch.operator" in frontend or
            "func.call" in frontend or signature not in frontend or
            any(token not in frontend for token in required) or
            any(token in frontend for token in forbidden) or
            frontend.count("arith.addf ") != 1 or
            (case == "dual" and (frontend.count("math.exp ") != 1 or
                                 frontend.count("arith.mulf ") != 1))):
        raise ValueError(f"model2MLIR {case} graph differs from source VPU policy")
    return frontend


def vector_specs(case: str) -> tuple[dict, ...]:
    add = dict(kind="add", src1=SP_A, src2=SP_B, dst=SP_D, rows=64, rlen=1, imm=0)
    if case == "chain":
        return (add,
                dict(kind="muls", src1=SP_D, src2=0, dst=SP_D,
                     rows=64, rlen=1, imm=0x3f00),
                dict(kind="rmax", src1=SP_D, src2=0, dst=SP_R,
                     rows=64, rlen=4, imm=0))
    if case == "war":
        return (add,)
    if case == "dual":
        return (dict(kind="adds", src1=SP_A, src2=0, dst=SP_A + 0x400,
                     rows=64, rlen=1, imm=0x4040),
                dict(kind="exp", src1=SP_R + 0x400, src2=0, dst=SP_R,
                     rows=64, rlen=1, imm=0),
                dict(kind="mul", src1=SP_A + 0x400, src2=SP_B, dst=SP_D,
                     rows=64, rlen=1, imm=0))
    raise ValueError(case)


def bind(case: str, frontend: str, source_sha: str, profile: dict) -> tuple[str, dict]:
    specs = vector_specs(case)
    policy = {"source_case": list(SOURCE_CHECKS[case]),
              "vpu_order": [spec["kind"] for spec in specs],
              "scratchpad_rows": [SP_A, SP_B, SP_D, SP_R],
              "ordering": ("VPU then overwrite source by DMA without fence" if case == "war"
                           else "two independent VPUs then dependent cross-VPU read without fence"
                           if case == "dual" else "dependent VPU RAW without fence"),
              "rounding": "Nicolas vpu_ref.h BF16 at each VPU operation"}
    manifest = {"schema": "mx_gemmini.nicolas_vpu_ordering_binding.v1",
                "source_sha256": source_sha,
                "frontend_mlir_sha256": _digest(frontend.encode()),
                "contract_sha256": _sha(CONTRACT),
                "policy_sha256": _digest(json.dumps(policy, sort_keys=True).encode()),
                "profile_sha256": profile_sha256(profile), "policy": policy}
    manifest_sha = _digest(json.dumps(manifest, sort_keys=True).encode())
    common = ", ".join(f'{field} = "{value}"' for field, value in (
        ("profile_sha256", manifest["profile_sha256"]),
        ("contract_sha256", manifest["contract_sha256"]),
        ("policy_sha256", manifest["policy_sha256"]),
        ("manifest_sha256", manifest_sha)))
    operations = []
    for index, spec in enumerate(specs):
        operations.append(f'''    "mx_gemmini.vpu_execute"() {{site_id = "vpu:{case}:{index}", kind = "{spec["kind"]}",
      src1_row = {spec["src1"]} : i32, src2_row = {spec["src2"]} : i32,
      dst_row = {spec["dst"]} : i32, rows = {spec["rows"]} : i32,
      reduction_length = {spec["rlen"]} : i32, broadcast = false,
      immediate_bf16 = {spec["imm"]} : i32, {common}}} : () -> ()''')
    mlir = f'''builtin.module attributes {{
  mx.contract_sha256 = "{manifest["contract_sha256"]}",
  mx.policy_sha256 = "{manifest["policy_sha256"]}",
  prov.quantization_manifest_sha256 = "{manifest_sha}",
  mx.source_mlir_sha256 = "{manifest["frontend_mlir_sha256"]}",
  mx.profile_sha256 = "{manifest["profile_sha256"]}"
}} {{
  func.func @vpu_{case}() {{
{chr(10).join(operations)}
    func.return
  }}
}}
'''
    vector = lower_vector_commands(mlir, profile)
    if (len(vector) != len(specs) or
            [command.funct for command in vector] != [33] * len(specs) or
            [command.rs2.immediate & 0xf for command in vector] != [
                VPU_OPCODES[spec["kind"]] for spec in specs]):
        raise ValueError(f"typed {case} VPU command order changed")
    manifest["manifest_sha256"] = manifest_sha
    manifest["bound_mlir_sha256"] = _digest(mlir.encode())
    return mlir, manifest


def commands(case: str, mlir: str, profile: dict):
    stream = [_cmd(7, 0, 0), _config_ld(16), _config_st(16)]
    inputs = (("a2", SP_A), ("b", SP_B), ("x", SP_R + 0x400)) if case == "dual" else (
        ("a", SP_A), ("b", SP_B))
    for buffer, spad in inputs:
        for row in range(0, 64, 16):
            stream.append(_transfer(2, buffer, row * 16, spad + row))
    vector = lower_vector_commands(mlir, profile)
    if case == "war":
        stream.extend(vector)
        for row in range(0, 64, 16):
            stream.append(_transfer(2, "a2", row * 16, SP_A + row))
    else:
        stream.extend(vector)
    outputs = (("out_x", SP_A + 0x400, 64),
               ("out_y", SP_R, 64),
               ("output", SP_D, 64)) if case == "dual" else (
                   ("output", SP_R, 16) if case == "chain" else
                   ("output", SP_D, 64),)
    for buffer, spad, count in outputs:
        for row in range(0, count, 16):
            stream.append(_transfer(3, buffer, row * 16, spad + row))
    return stream


def driver(case: str) -> str:
    if case == "chain":
        reference = '''
  vpu_ref_exec(VPU_ADD, ref, a, b, 64, 1, 0, 0);
  vpu_ref_exec(VPU_MULS, ref, ref, 0, 64, 1, 0, 0x3f00);
  vpu_ref_exec(VPU_RMAX, ref, ref, 0, 64, 4, 0, 0);
  int bad = 0;
  for (int r = 0; r < 16; r++)
    for (int l = 0; l < VPU_LANES; l++)
      if (output[r][l] != ref[r][l]) bad++;
  printf("compiled ordering chain: %d mismatches\\n", bad);
  return bad != 0;'''
    elif case == "war":
        reference = '''
  vpu_ref_exec(VPU_ADD, ref, a, b, 64, 1, 0, 0);
  int bad = 0;
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++)
      if (output[r][l] != ref[r][l]) bad++;
  printf("compiled ordering war: %d mismatches\\n", bad);
  return bad != 0;'''
    elif case == "dual":
        reference = '''
  vpu_ref_exec(VPU_ADDS, ref_x, a2, 0, 64, 1, 0, 0x4040);
  vpu_ref_exec(VPU_EXP, ref_y, x, 0, 64, 1, 0, 0);
  vpu_ref_exec(VPU_MUL, ref, ref_x, b, 64, 1, 0, 0);
  int xy_bad = 0, z_bad = 0;
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++) {
      xy_bad += (out_x[r][l] != ref_x[r][l]) + (out_y[r][l] != ref_y[r][l]);
      z_bad += output[r][l] != ref[r][l];
    }
  printf("compiled ordering dual: %d XY mismatches, %d Z mismatches\\n",
         xy_bad, z_bad);
  return xy_bad != 0 || z_bad != 0;'''
    else:
        raise ValueError(case)
    return f'''#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"

static uint16_t a[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t a2[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t b[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t x[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t output[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t out_x[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t out_y[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t ref[64][VPU_LANES];
static uint16_t ref_x[64][VPU_LANES];
static uint16_t ref_y[64][VPU_LANES];
static uint32_t lcg = 12345;
static uint32_t rnd(void) {{ lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }}
static uint16_t rand_bf16(int elo, int ehi, int sign) {{
  uint16_t s = (sign && (rnd() & 1)) ? 0x8000 : 0;
  return (uint16_t)(s | ((elo + rnd() % (ehi - elo + 1)) << 7) | (rnd() & 0x7f));
}}
void mx_issue(const void *a, const void *a2, const void *b, const void *x,
              const void *out_x, const void *out_y, const void *output);

int main(void) {{
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++) {{
      a[r][l] = rand_bf16(110, 140, 1);
      a2[r][l] = rand_bf16(110, 140, 1);
      b[r][l] = rand_bf16(110, 140, 1);
      (void)rand_bf16(100, 154, 0);  // P in Nicolas's source
      uint16_t v;
      do v = rand_bf16(100, 133, 1);
      while (vpu_bf16_to_f(v) > 80.0f || vpu_bf16_to_f(v) < -80.0f);
      x[r][l] = v;
    }}
  x[0][0] = 0x0000; x[0][1] = 0x8000; x[0][2] = 0x7f80;
  x[0][3] = 0xff80; x[0][4] = 0x7fc1; x[0][5] = 0x0001;
  mx_issue(a, a2, b, x, out_x, out_y, output);
  gemmini_fence();
{reference}
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    model2mlir, rtl, riscv, out = (args.model2mlir_root.resolve(),
                                   args.rtl_root.resolve(), args.riscv_root.resolve(),
                                   args.out_dir.resolve())
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    software = rtl / "software/gemmini-rocc-tests"
    extension = rtl / "software/libgemmini"
    _require_gitlink(rtl, "software/gemmini-rocc-tests")
    _require_gitlink(rtl, "software/libgemmini")
    profile = load_profile(PROFILE, rtl_root=rtl)
    if profile["resources"].get("vpu_config") != {
            "units": 2, "exp_sub": True, "exp_sum": True}:
        parser.error("selected Nicolas profile lacks the qualified two-unit VPU")
    source, oracle = software / "bareMetalC/vpu_ops.c", software / "include/vpu_ref.h"
    baseline = json.loads(SOURCE_RECEIPT.read_text())
    if (_sha(source) != baseline["source_sha256"] or
            _sha(oracle) != baseline["reference_sha256"] or
            baseline["check_count"] != 29):
        parser.error("Nicolas VPU source or oracle differs from qualified baseline")
    source_text = source.read_text()
    if any(token not in source_text for token in (
            'check("chain",', 'check("war mvin",',
            '"dual X,Y"', 'check("dual Z=X*B",',
            'gemmini_vpu(VPU_ADD,  SP_D, SP_A, SP_B, N, 1, 0, 0);',
            'mvin_rows(A2, SP_A, N);')):
        parser.error("Nicolas VPU ordering source cases changed")
    cc, spike = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike"
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        parser.error("RISC-V GCC, Spike, and host g++ are required")
    sys.path.insert(0, str(model2mlir))
    import m2m
    import torch
    if Path(m2m.__file__).resolve().parents[1] != model2mlir:
        parser.error("model2MLIR resolved to a different checkout")

    out.mkdir(parents=True)
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = out / "libgemmini.so"
    _run(["g++", "-L", str(riscv / "lib"), f"-Wl,-rpath,{riscv / 'lib'}",
          "-shared", "-o", str(so), "-std=c++17", "-I", str(riscv / "include"),
          "-fPIC", "-O3", f"-ffile-prefix-map={extension}=software/libgemmini",
          *(str(path) for path in extension_sources)], cwd=out, log=out / "extension_build.log")
    bench = software / "riscv-tests/benchmarks/common"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc", f"-ffile-prefix-map={out}=.",
             f"-ffile-prefix-map={software}=software/gemmini-rocc-tests",
             "-I", str(software / "riscv-tests"), "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    common_objects = []
    for index, path in enumerate(sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))):
        obj = out / f"common_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(obj)],
             cwd=out, log=out / f"common_{index}.log")
        common_objects.append(obj)
    rows = []
    for case in CASES:
        directory = out / case
        directory.mkdir()
        frontend = capture(case, m2m, torch)
        mlir, binding = bind(case, frontend, _sha(source), profile)
        (directory / "frontend.mlir").write_text(frontend)
        (directory / "bound.mlir").write_text(mlir)
        (directory / "binding.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
        stream = commands(case, mlir, profile)
        expected = {"chain": [7, 0, 0, *([2] * 8), 33, 33, 33, 3],
                    "war": [7, 0, 0, *([2] * 8), 33, *([2] * 4), *([3] * 4)],
                    "dual": [7, 0, 0, *([2] * 12), 33, 33, 33, *([3] * 12)]}[case]
        if [command.funct for command in stream] != expected:
            raise ValueError(f"{case} physical schedule differs from source ordering")
        (directory / "mx_issue.c").write_text(emit_c(
            stream, transport="rocket_rocc",
            buffers=("a", "a2", "b", "x", "out_x", "out_y", "output")))
        (directory / "mx_driver.c").write_text(driver(case))
        objects = []
        for index, name in enumerate(("mx_issue.c", "mx_driver.c")):
            obj = directory / f"program_{index}.o"
            _run([str(cc), *flags, "-c", str(directory / name), "-o", str(obj)],
                 cwd=directory, log=directory / f"compile_{index}.log")
            objects.append(obj)
        elf = directory / "ordering.elf"
        _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
              str(bench / "test.ld"), *(str(path) for path in objects + common_objects),
              "-lm", "-lgcc", "-o", str(elf)], cwd=directory,
             log=directory / "link.log")
        result = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini",
                                 str(elf)], cwd=directory, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                check=False)
        log = directory / "spike.log"
        log.write_text(result.stdout)
        expected_log = (f"compiled ordering {case}: 0 XY mismatches, 0 Z mismatches"
                        if case == "dual" else f"compiled ordering {case}: 0 mismatches")
        if result.returncode or expected_log not in result.stdout:
            raise RuntimeError(f"compiler-issued {case} differs from source oracle: {log}")
        compared = {"chain": [128], "war": [512], "dual": [1024, 512]}[case]
        rows.append({"name": case, "source_checks": SOURCE_CHECKS[case],
                     "status": "source_vpu_reference_matched_on_pinned_spike",
                     "frontend_mlir_sha256": _sha(directory / "frontend.mlir"),
                     "bound_mlir_sha256": _sha(directory / "bound.mlir"),
                     "binding_sha256": _sha(directory / "binding.json"),
                     "issuer_sha256": _sha(directory / "mx_issue.c"),
                     "driver_sha256": _sha(directory / "mx_driver.c"),
                     "elf_sha256": _sha(elf), "spike_log_sha256": _sha(log),
                     "ordered_functs": expected, "compared_bf16": compared})
    index = {"schema": "mx_gemmini.nicolas_vpu_ordering_compiled_spike.v1",
             "status": "compiler_vpu_ordering_source_checks_matched_reference",
             "source_qualification_sha256": _sha(SOURCE_RECEIPT),
             "source_sha256": _sha(source), "reference_sha256": _sha(oracle),
             "source_revision": _git_revision(software), "rtl_revision": _git_revision(rtl),
             "extension_revision": _git_revision(extension),
             "model2mlir_revision": _git_revision(model2mlir),
             "profile_sha256": profile_sha256(profile),
             "compiler_revision": _git_revision(ROOT),
             "compiler_source_closure_sha256": _source_closure(
                 ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
                 sorted((ROOT / "tools").glob("*.py"))),
             "riscv_gcc_sha256": _sha(cc), "spike_sha256": _sha(spike),
             "extension_sha256": _sha(so), "rows": rows}
    (out / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print("compiler-issued all four remaining Nicolas VPU source checks matched 2176 BF16 values")


if __name__ == "__main__":
    main()
