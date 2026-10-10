"""Compile Nicolas's VPU broadcast and same-bank cases from model2MLIR.

This qualifies each source case as an independent Rocket program. The C
driver retains Nicolas's inputs and BF16 oracle, not accelerator commands.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from tools.qualify_nicolas_vpu_elementwise import capture as capture_base
from tools.qualify_nicolas_vpu_fused import (
    CONTRACT, PROFILE, SOURCE_RECEIPT, capture as capture_fused,
)


ROOT = Path(__file__).resolve().parents[1]
SP_A, SP_A2, SP_B, SP_D, SP_R = 0, 0x800, 0x1000, 0x2000, 0x3000


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    second: str | None
    broadcast: bool
    reduction_length: int

    @property
    def second_row(self) -> int:
        return {"a2": SP_A2, "b": SP_B, None: 0}[self.second]

    @property
    def destination_row(self) -> int:
        return SP_R if self.kind == "rsum" else SP_D

    @property
    def output_rows(self) -> int:
        return 64 // self.reduction_length if self.kind == "rsum" else 64


CASES = (
    Case("mul_same_bank", "mul", "a2", False, 1),
    Case("mul_bcast", "mul", "b", True, 4),
    Case("max_same_bank", "max", "a2", False, 1),
    Case("max_bcast", "max", "b", True, 4),
    Case("sub_bcast_same_bank", "sub", "a2", True, 4),
    Case("expsub_plain", "expsub", "b", False, 1),
    Case("expsub_bcast", "expsub", "b", True, 4),
    Case("expsub_same_bank", "expsub", "a2", True, 4),
    Case("expsum_bcast", "expsum", "b", True, 4),
    Case("expsum_same_bank", "expsum", "a2", True, 4),
    Case("rsum_rlen1", "rsum", None, False, 1),
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def capture(case: Case, m2m, torch) -> str:
    if not case.broadcast and case.kind in {"mul", "max"}:
        return capture_base(case.kind, m2m, torch)
    if not case.broadcast and case.kind == "expsub":
        return capture_fused("expsub", m2m, torch)

    class Model(torch.nn.Module):
        def forward(self, a, b):
            if case.kind == "rsum":
                return a.sum(dim=1)
            lhs = a.reshape(16, 4, 8)
            rhs = b.reshape(16, 1, 8)
            if case.kind == "mul": value = lhs * rhs
            elif case.kind == "max": value = torch.maximum(lhs, rhs)
            else: value = lhs - rhs
            if case.kind in {"expsub", "expsum"}:
                value = torch.exp(value)
            if case.kind == "expsum":
                return value.reshape(64, 8), value.sum(dim=(1, 2))
            return value.reshape(64, 8)

    lhs = torch.zeros((64, 8), dtype=torch.bfloat16)
    rhs = torch.zeros((16 if case.broadcast else 64, 8), dtype=torch.bfloat16)
    result = m2m.convert(Model().eval(), (lhs, rhs), backend="fx_importer")
    if not result.ok:
        raise ValueError(f"model2MLIR {case.name} capture failed: {result.diagnostics}")
    from m2m.coverage import opaque_report
    frontend = result.mlir_text
    primitive = {"mul": "arith.mulf", "max": "arith.maximumf",
                 "sub": "arith.subf", "expsub": "arith.subf",
                 "expsum": "arith.subf", "rsum": "arith.addf"}[case.kind]
    aten = {"mul": "aten.mul.Tensor", "max": "aten.maximum.default",
            "sub": "aten.sub.Tensor", "expsub": "aten.sub.Tensor",
            "expsum": "aten.sub.Tensor", "rsum": "aten.sum.dim_IntList"}[case.kind]
    expected_input = ("tensor<16x8xbf16>" if case.broadcast else "tensor<64x8xbf16>")
    expected_return = ("tensor<64xbf16>" if case.kind == "rsum" else
                       "(tensor<64x8xbf16>, tensor<16xbf16>)" if case.kind == "expsum" else
                       "tensor<64x8xbf16>")
    signature = (f"func.func @forward(%0: tensor<64x8xbf16>, %1: {expected_input}) "
                 f"-> {expected_return}")
    if (opaque_report(frontend) or "torch.operator" in frontend or
            "func.call" in frontend or signature not in frontend or
            f'prov.aten = "{aten}"' not in frontend or primitive not in frontend or
            ("math.exp" in frontend) != (case.kind in {"expsub", "expsum"}) or
            ("linalg.reduce" in frontend) != (case.kind in {"expsum", "rsum"}) or
            ("dimensions = [1, 2]" in frontend) != (case.kind == "expsum") or
            ("dimensions = [1]" in frontend) != (case.kind == "rsum") or
            ("output_shape [16, 4, 8]" in frontend) != case.broadcast):
        raise ValueError(f"model2MLIR {case.name} decomposition differs from VPU policy")
    return frontend


def bind(case: Case, frontend: str, source_sha: str, profile: dict) -> tuple[str, dict]:
    policy = {"source_case": case.name, "kind": case.kind,
              "source_shape": [64, 8],
              "second_shape": [16 if case.broadcast else 64, 8] if case.second else None,
              "source2_array": case.second, "src2_row": case.second_row,
              "destination_row": case.destination_row,
              "broadcast": case.broadcast, "reduction_length": case.reduction_length,
              "rounding": "Nicolas vpu_ref.h BF16; reduction result replicated across 8 lanes"}
    manifest = {"schema": "mx_gemmini.nicolas_vpu_variant_binding.v1",
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
    second_dst = f", second_dst_row = {SP_R} : i32" if case.kind == "expsum" else ""
    mlir = f'''builtin.module attributes {{
  mx.contract_sha256 = "{manifest["contract_sha256"]}",
  mx.policy_sha256 = "{manifest["policy_sha256"]}",
  prov.quantization_manifest_sha256 = "{manifest_sha}",
  mx.source_mlir_sha256 = "{manifest["frontend_mlir_sha256"]}",
  mx.profile_sha256 = "{manifest["profile_sha256"]}"
}} {{
  func.func @vpu_{case.name}() {{
    "mx_gemmini.vpu_execute"() {{site_id = "vpu:{case.name}", kind = "{case.kind}",
      src1_row = {SP_A} : i32, src2_row = {case.second_row} : i32,
      dst_row = {case.destination_row} : i32, rows = 64 : i32,
      reduction_length = {case.reduction_length} : i32,
      broadcast = {str(case.broadcast).lower()}, immediate_bf16 = 0 : i32{second_dst},
      {common}}} : () -> ()
    func.return
  }}
}}
'''
    vector = lower_vector_commands(mlir, profile)
    if (len(vector) != 1 or vector[0].funct != 33 or
            vector[0].rs2.immediate & 0xf != VPU_OPCODES[case.kind] or
            bool(vector[0].rs2.immediate & 0x10) != case.broadcast):
        raise ValueError(f"typed {case.name} did not lower to its selected VPU command")
    manifest["manifest_sha256"] = manifest_sha
    manifest["bound_mlir_sha256"] = _digest(mlir.encode())
    return mlir, manifest


def commands(case: Case, mlir: str, profile: dict):
    stream = [_cmd(7, 0, 0), _config_ld(16), _config_st(16)]
    for row in range(0, 64, 16):
        stream.append(_transfer(2, "a", row * 16, SP_A + row))
    if case.second:
        for row in range(0, 16 if case.broadcast else 64, 16):
            stream.append(_transfer(2, case.second, row * 16, case.second_row + row))
    stream.extend(lower_vector_commands(mlir, profile))
    for row in range(0, case.output_rows, 16):
        stream.append(_transfer(3, "output", row * 16, case.destination_row + row))
    if case.kind == "expsum":
        stream.append(_transfer(3, "sums", 0, SP_R))
    return stream


def driver(case: Case) -> str:
    second = case.second or "0"
    sums_check = '''
  vpu_ref_exec(VPU_RSUM, sum_ref, ref, 0, 64, 4, 0, 0);
  for (int r = 0; r < 16; r++)
    for (int l = 0; l < VPU_LANES; l++)
      if (sums[r][l] != sum_ref[r][l]) sum_bad++;
''' if case.kind == "expsum" else ""
    return f'''#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"

static uint16_t a[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t a2[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t b[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t output[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t sums[16][VPU_LANES] __attribute__((aligned(64)));
static uint16_t ref[64][VPU_LANES];
static uint16_t sum_ref[16][VPU_LANES];
static uint32_t lcg = 12345;
static uint32_t rnd(void) {{ lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }}
static uint16_t rand_bf16(int elo, int ehi, int sign) {{
  uint16_t s = (sign && (rnd() & 1)) ? 0x8000 : 0;
  return (uint16_t)(s | ((elo + rnd() % (ehi - elo + 1)) << 7) | (rnd() & 0x7f));
}}
void mx_issue(const void *a, const void *a2, const void *b,
              const void *output, const void *sums);

int main(void) {{
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++) {{
      a[r][l] = rand_bf16(110, 140, 1);
      a2[r][l] = rand_bf16(110, 140, 1);
      b[r][l] = rand_bf16(110, 140, 1);
      (void)rand_bf16(100, 154, 0);  // P in Nicolas's source
      uint16_t x;
      do x = rand_bf16(100, 133, 1);
      while (vpu_bf16_to_f(x) > 80.0f || vpu_bf16_to_f(x) < -80.0f);
    }}
  mx_issue(a, a2, b, output, sums);
  gemmini_fence();
  vpu_ref_exec(VPU_{case.kind.upper()}, ref, a, {second}, 64,
               {case.reduction_length}, {int(case.broadcast)}, 0);
  int out_bad = 0, sum_bad = 0;
  for (int r = 0; r < {case.output_rows}; r++)
    for (int l = 0; l < VPU_LANES; l++)
      if (output[r][l] != ref[r][l]) out_bad++;
{sums_check}  printf("compiled variant {case.name}: %d output mismatches, %d sum mismatches\\n",
         out_bad, sum_bad);
  return out_bad != 0 || sum_bad != 0;
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
    for label in ('"mul same-bank"', '"mul bcast"', '"max same-bank"',
                  '"max bcast"', '"sub bcast sb"', '"expsub"', '"expsub bcast"',
                  '"expsub sb"', '"expsum sb"', '"expsum bcast"',
                  '"rsum rlen1"'):
        if source_text.count(label) != 1:
            parser.error(f"Nicolas VPU source case {label} changed")
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
        directory = out / case.name
        directory.mkdir()
        frontend = capture(case, m2m, torch)
        mlir, binding = bind(case, frontend, _sha(source), profile)
        (directory / "frontend.mlir").write_text(frontend)
        (directory / "bound.mlir").write_text(mlir)
        (directory / "binding.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
        stream = commands(case, mlir, profile)
        expected_functs = ([7, 0, 0] + [2] * (4 + (1 if case.broadcast else 4) if case.second else 4) +
                           [33] + [3] * (case.output_rows // 16 + int(case.kind == "expsum")))
        if [command.funct for command in stream] != expected_functs:
            raise ValueError(f"{case.name} physical schedule differs from policy")
        (directory / "mx_issue.c").write_text(emit_c(
            stream, transport="rocket_rocc", buffers=("a", "a2", "b", "output", "sums")))
        (directory / "mx_driver.c").write_text(driver(case))
        objects = []
        for index, name in enumerate(("mx_issue.c", "mx_driver.c")):
            obj = directory / f"program_{index}.o"
            _run([str(cc), *flags, "-c", str(directory / name), "-o", str(obj)],
                 cwd=directory, log=directory / f"compile_{index}.log")
            objects.append(obj)
        elf = directory / "variant.elf"
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
        expected = f"compiled variant {case.name}: 0 output mismatches, 0 sum mismatches"
        if result.returncode or expected not in result.stdout:
            raise RuntimeError(f"compiler-issued {case.name} differs from source oracle: {log}")
        rows.append({"name": case.name, "kind": case.kind,
                     "status": "source_vpu_reference_matched_on_pinned_spike",
                     "frontend_mlir_sha256": _sha(directory / "frontend.mlir"),
                     "bound_mlir_sha256": _sha(directory / "bound.mlir"),
                     "binding_sha256": _sha(directory / "binding.json"),
                     "issuer_sha256": _sha(directory / "mx_issue.c"),
                     "driver_sha256": _sha(directory / "mx_driver.c"),
                     "elf_sha256": _sha(elf), "spike_log_sha256": _sha(log),
                     "ordered_functs": expected_functs,
                     "compared_output_bf16": case.output_rows * 8,
                     "compared_sum_bf16": 128 if case.kind == "expsum" else 0})
    index = {"schema": "mx_gemmini.nicolas_vpu_variants_compiled_spike.v1",
             "status": "compiler_vpu_source_variants_matched_reference",
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
    total = sum(row["compared_output_bf16"] + row["compared_sum_bf16"] for row in rows)
    print(f"compiler-issued {len(rows)} VPU variants matched {total} BF16 values on Spike")


if __name__ == "__main__":
    main()
