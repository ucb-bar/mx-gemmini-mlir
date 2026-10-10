"""Compile Nicolas's unfused VPU op classes from model2MLIR and run on Spike.

The C driver generates source inputs and evaluates vpu_ref.h. All accelerator
configuration, transfers, VPU commands, and readout come from bound MLIR.
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
KINDS = ("add", "sub", "mul", "max", "adds", "muls", "exp", "rcp",
         "rsqrt", "rmax", "ramax", "rsum")
BINARY = {"add", "sub", "mul", "max"}
REDUCING = {"rmax", "ramax", "rsum"}
SRC1 = {"exp": "x", "rsqrt": "p"}
IMMEDIATE = {"adds": 0x4040, "muls": 0x3f00}
ATEN = {"add": "aten.add.Tensor", "sub": "aten.sub.Tensor",
        "mul": "aten.mul.Tensor", "max": "aten.maximum.default",
        "adds": "aten.add.Tensor", "muls": "aten.mul.Tensor",
        "exp": "aten.exp.default", "rcp": "aten.reciprocal.default",
        "rsqrt": "aten.rsqrt.default", "rmax": "aten.amax.default",
        "ramax": "aten.amax.default", "rsum": "aten.sum.dim_IntList"}
PRIMITIVE = {"add": "arith.addf", "sub": "arith.subf",
             "mul": "arith.mulf", "max": "arith.maximumf",
             "adds": "arith.addf", "muls": "arith.mulf",
             "exp": "math.exp", "rcp": "arith.divf",
             "rsqrt": "math.rsqrt", "rmax": "arith.maximumf",
             "ramax": "math.absf", "rsum": "arith.addf"}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def capture(kind: str, m2m, torch) -> str:
    class Model(torch.nn.Module):
        def forward(self, a, b):
            if kind == "add": return a + b
            if kind == "sub": return a - b
            if kind == "mul": return a * b
            if kind == "max": return torch.maximum(a, b)
            if kind == "adds": return a + 3.0
            if kind == "muls": return a * 0.5
            if kind == "exp": return torch.exp(a)
            if kind == "rcp": return torch.reciprocal(a)
            if kind == "rsqrt": return torch.rsqrt(a)
            if kind == "rmax": return a.reshape(16, 4, 8).amax(dim=(1, 2))
            if kind == "ramax": return a.abs().reshape(16, 4, 8).amax(dim=(1, 2))
            if kind == "rsum": return a.reshape(16, 4, 8).sum(dim=(1, 2))
            raise ValueError(kind)

    sample = torch.zeros((64, 8), dtype=torch.bfloat16)
    result = m2m.convert(Model().eval(), (sample, sample), backend="fx_importer")
    if not result.ok:
        raise ValueError(f"model2MLIR {kind} capture failed: {result.diagnostics}")
    from m2m.coverage import opaque_report
    frontend = result.mlir_text
    if (opaque_report(frontend) or "torch.operator" in frontend or
            "func.call" in frontend or
            "func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>)" not in frontend or
            f'prov.aten = "{ATEN[kind]}"' not in frontend or
            PRIMITIVE[kind] not in frontend or
            ("linalg.reduce" in frontend) != (kind in REDUCING) or
            ("dimensions = [1, 2]" in frontend) != (kind in REDUCING) or
            ("output_shape [16, 4, 8]" in frontend) != (kind in REDUCING) or
            ("tensor<16xbf16>" in frontend) != (kind in REDUCING)):
        raise ValueError(f"model2MLIR {kind} decomposition differs from VPU binding policy")
    if kind == "ramax" and "math.absf" not in frontend:
        raise ValueError("RAMAX capture lacks absolute value")
    if kind in IMMEDIATE:
        literal = "3.000000e+00" if kind == "adds" else "5.000000e-01"
        if literal not in frontend or "tensor.splat" not in frontend:
            raise ValueError(f"{kind} capture lacks expected BF16 scalar")
    return frontend


def bind(kind: str, frontend: str, source_sha: str, profile: dict) -> tuple[str, dict]:
    policy = {"kind": kind, "source_shape": [64, 8],
              "result_shape": [16] if kind in REDUCING else [64, 8],
              "source_array": SRC1.get(kind, "a"),
              "source2_array": "b" if kind in BINARY else None,
              "reduction_length": 4 if kind in REDUCING else 1,
              "immediate_bf16": IMMEDIATE.get(kind, 0),
              "rounding": "Nicolas vpu_ref.h BF16; reduction result replicated across 8 lanes"}
    manifest = {"schema": "mx_gemmini.nicolas_elementwise_vpu_binding.v1",
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
    mlir = f'''builtin.module attributes {{
  mx.contract_sha256 = "{manifest["contract_sha256"]}",
  mx.policy_sha256 = "{manifest["policy_sha256"]}",
  prov.quantization_manifest_sha256 = "{manifest_sha}",
  mx.source_mlir_sha256 = "{manifest["frontend_mlir_sha256"]}",
  mx.profile_sha256 = "{manifest["profile_sha256"]}"
}} {{
  func.func @vpu_{kind}() {{
    "mx_gemmini.vpu_execute"() {{site_id = "vpu:{kind}", kind = "{kind}",
      src1_row = 0 : i32, src2_row = {4096 if kind in BINARY else 0} : i32,
      dst_row = 8192 : i32, rows = 64 : i32,
      reduction_length = {4 if kind in REDUCING else 1} : i32,
      broadcast = false, immediate_bf16 = {IMMEDIATE.get(kind, 0)} : i32,
      {common}}} : () -> ()
    func.return
  }}
}}
'''
    vector = lower_vector_commands(mlir, profile)
    if len(vector) != 1 or vector[0].funct != 33 or vector[0].rs2.immediate & 0xf != VPU_OPCODES[kind]:
        raise ValueError(f"typed {kind} op did not lower to VPU opcode")
    manifest["manifest_sha256"] = manifest_sha
    manifest["bound_mlir_sha256"] = _digest(mlir.encode())
    return mlir, manifest


def commands(kind: str, mlir: str, profile: dict):
    stream = [_cmd(7, 0, 0), _config_ld(16), _config_st(16)]
    for row in range(0, 64, 16):
        stream.append(_transfer(2, "src1", row * 16, row))
    if kind in BINARY:
        for row in range(0, 64, 16):
            stream.append(_transfer(2, "src2", row * 16, 4096 + row))
    stream.extend(lower_vector_commands(mlir, profile))
    for row in range(0, 16 if kind in REDUCING else 64, 16):
        stream.append(_transfer(3, "output", row * 16, 8192 + row))
    return stream


def driver(kind: str) -> str:
    op = f"VPU_{kind.upper()}"
    src1 = SRC1.get(kind, "a")
    src2 = "b" if kind in BINARY else "0"
    nout = 16 if kind in REDUCING else 64
    return f'''#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"

static uint16_t a[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t b[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t p[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t x[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t output[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t ref[64][VPU_LANES];
static uint32_t lcg = 12345;
static uint32_t rnd(void) {{ lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }}
static uint16_t rand_bf16(int elo, int ehi, int sign) {{
  uint16_t s = (sign && (rnd() & 1)) ? 0x8000 : 0;
  return (uint16_t)(s | ((elo + rnd() % (ehi - elo + 1)) << 7) | (rnd() & 0x7f));
}}
void mx_issue(const void *src1, const void *src2, const void *output);

int main(void) {{
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++) {{
      a[r][l] = rand_bf16(110, 140, 1);
      (void)rand_bf16(110, 140, 1);  // A2 in Nicolas's source
      b[r][l] = rand_bf16(110, 140, 1);
      p[r][l] = rand_bf16(100, 154, 0);
      uint16_t v;
      do v = rand_bf16(100, 133, 1);
      while (vpu_bf16_to_f(v) > 80.0f || vpu_bf16_to_f(v) < -80.0f);
      x[r][l] = v;
    }}
  x[0][0] = 0x0000; x[0][1] = 0x8000; x[0][2] = 0x7f80;
  x[0][3] = 0xff80; x[0][4] = 0x7fc1; x[0][5] = 0x0001;
  p[0][0] = 0x0000; p[0][1] = 0x7f80; p[0][2] = 0x0001;
  mx_issue({src1}, {src2}, output);
  gemmini_fence();
  vpu_ref_exec({op}, ref, {src1}, {src2}, 64,
               {4 if kind in REDUCING else 1}, 0, {IMMEDIATE.get(kind, 0)});
  int bad = 0;
  for (int r = 0; r < {nout}; r++)
    for (int l = 0; l < VPU_LANES; l++)
      if (output[r][l] != ref[r][l]) bad++;
  printf("compiled VPU {kind}: %d mismatches\\n", bad);
  return bad != 0;
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
        parser.error("selected Nicolas profile lacks the qualified VPU")
    source, oracle = software / "bareMetalC/vpu_ops.c", software / "include/vpu_ref.h"
    baseline = json.loads(SOURCE_RECEIPT.read_text())
    if (_sha(source) != baseline["source_sha256"] or
            _sha(oracle) != baseline["reference_sha256"] or
            baseline["check_count"] != 29):
        parser.error("Nicolas VPU source or oracle differs from qualified baseline")
    source_text = source.read_text()
    source_lines = ("static uint32_t lcg = 12345;",
                    "A[r][l] = rand_bf16(110, 140, 1);",
                    "A2[r][l] = rand_bf16(110, 140, 1);",
                    "B[r][l] = rand_bf16(110, 140, 1);",
                    "P[r][l] = rand_bf16(100, 154, 0);",
                    "do x = rand_bf16(100, 133, 1);",
                    "X[0][0] = 0x0000;", "P[0][0] = 0x0000;")
    if any(source_text.count(line) != 1 for line in source_lines):
        parser.error("Nicolas VPU source inputs changed")
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
    for kind in KINDS:
        directory = out / kind
        directory.mkdir()
        frontend = capture(kind, m2m, torch)
        mlir, binding = bind(kind, frontend, _sha(source), profile)
        (directory / "frontend.mlir").write_text(frontend)
        (directory / "bound.mlir").write_text(mlir)
        (directory / "binding.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
        stream = commands(kind, mlir, profile)
        expected_functs = [7, 0, 0, *([2] * (8 if kind in BINARY else 4)),
                           33, *([3] * (1 if kind in REDUCING else 4))]
        if [command.funct for command in stream] != expected_functs:
            raise ValueError(f"{kind} physical command schedule differs from policy")
        (directory / "mx_issue.c").write_text(emit_c(
            stream, transport="rocket_rocc", buffers=("src1", "src2", "output")))
        (directory / "mx_driver.c").write_text(driver(kind))
        objects = []
        for index, name in enumerate(("mx_issue.c", "mx_driver.c")):
            obj = directory / f"program_{index}.o"
            _run([str(cc), *flags, "-c", str(directory / name), "-o", str(obj)],
                 cwd=directory, log=directory / f"compile_{index}.log")
            objects.append(obj)
        elf = directory / "vpu.elf"
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
        if result.returncode or f"compiled VPU {kind}: 0 mismatches" not in result.stdout:
            raise RuntimeError(f"compiler-issued {kind} differs from source oracle: {log}")
        rows.append({"kind": kind,
                     "status": "source_vpu_reference_matched_on_pinned_spike",
                     "frontend_mlir_sha256": _sha(directory / "frontend.mlir"),
                     "bound_mlir_sha256": _sha(directory / "bound.mlir"),
                     "binding_sha256": _sha(directory / "binding.json"),
                     "issuer_sha256": _sha(directory / "mx_issue.c"),
                     "driver_sha256": _sha(directory / "mx_driver.c"),
                     "elf_sha256": _sha(elf), "spike_log_sha256": _sha(log),
                     "ordered_functs": expected_functs,
                     "compared_output_bf16": (16 if kind in REDUCING else 64) * 8})
    index = {"schema": "mx_gemmini.nicolas_elementwise_vpu_compiled_spike.v1",
             "status": "compiler_vpu_base_ops_matched_source_reference",
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
    print(f"compiler-issued {len(rows)} base VPU classes matched source BF16 oracle on Spike")


if __name__ == "__main__":
    main()
