"""Capture and compile EXPSUB/EXPSUM, then compare against Nicolas's BF16 VPU reference.

The accelerator command is lowered from typed, profile-bound MLIR. The C
driver supplies the source test's deterministic inputs and bit-exact oracle;
it contains no Gemmini configuration, transfer, or VPU issue calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.physical_program import _cmd, _config_ld, _config_st, _transfer
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vector_lowering import lower_vector_commands
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json"
SOURCE_RECEIPT = ROOT / "docs/evidence/nicolas_vpu_source_all_ops_266c593/receipt.json"
CONTRACT = ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
ROWS, RLEN, SP_A, SP_B, SP_D, SP_SUM = 64, 4, 0, 0x1000, 0x2000, 0x3000


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def capture(kind: str, m2m, torch) -> str:
    class Fused(torch.nn.Module):
        def forward(self, a, b):
            exp = torch.exp(a - b)
            if kind == "expsum":
                return exp, exp.reshape(16, 4, 8).sum(dim=1)
            return exp

    inputs = (torch.zeros((64, 8), dtype=torch.bfloat16),) * 2
    result = m2m.convert(Fused().eval(), inputs, backend="fx_importer")
    if not result.ok:
        raise ValueError(f"model2MLIR {kind} capture failed: {result.diagnostics}")
    from m2m.coverage import opaque_report
    frontend = result.mlir_text
    if (opaque_report(frontend) or
            "func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>)" not in frontend or
            frontend.count('prov.aten = "aten.sub.Tensor"') != 1 or
            frontend.count('prov.aten = "aten.exp.default"') != 1 or
            frontend.count("arith.subf") != 1 or frontend.count("math.exp") != 1 or
            (frontend.count("linalg.reduce") == 1) != (kind == "expsum") or
            ("output_shape [16, 4, 8]" in frontend) != (kind == "expsum") or
            ("dimensions = [1]" in frontend) != (kind == "expsum") or
            "torch.operator" in frontend or "func.call" in frontend):
        raise ValueError(f"model2MLIR {kind} decomposition differs from selected VPU policy")
    return frontend


def bind(kind: str, frontend: str, source_sha: str, profile: dict) -> tuple[str, dict]:
    policy = {"kind": kind, "shape": [64, 8], "reduction_length": RLEN,
              "broadcast": True, "scratchpad_rows": [SP_A, SP_B, SP_D, SP_SUM],
              "rounding": "BF16 after subtract, exp, and reduction as vpu_ref.h"}
    manifest = {"schema": "mx_gemmini.nicolas_fused_vpu_binding.v1",
                "source_sha256": source_sha,
                "frontend_mlir_sha256": digest(frontend.encode()),
                "contract_sha256": _sha(CONTRACT),
                "policy_sha256": digest(json.dumps(policy, sort_keys=True).encode()),
                "profile_sha256": profile_sha256(profile), "policy": policy}
    manifest_sha = digest(json.dumps(manifest, sort_keys=True).encode())
    common = (f'profile_sha256 = "{manifest["profile_sha256"]}", '
              f'contract_sha256 = "{manifest["contract_sha256"]}", '
              f'policy_sha256 = "{manifest["policy_sha256"]}", '
              f'manifest_sha256 = "{manifest_sha}"')
    second = f", second_dst_row = {SP_SUM} : i32" if kind == "expsum" else ""
    mlir = f'''builtin.module attributes {{
  mx.contract_sha256 = "{manifest["contract_sha256"]}",
  mx.policy_sha256 = "{manifest["policy_sha256"]}",
  prov.quantization_manifest_sha256 = "{manifest_sha}",
  mx.source_mlir_sha256 = "{manifest["frontend_mlir_sha256"]}",
  mx.profile_sha256 = "{manifest["profile_sha256"]}"
}} {{
  func.func @fused_vpu() {{
    "mx_gemmini.vpu_execute"() {{site_id = "vpu:{kind}", kind = "{kind}",
      src1_row = {SP_A} : i32, src2_row = {SP_B} : i32,
      dst_row = {SP_D} : i32, rows = {ROWS} : i32,
      reduction_length = {RLEN} : i32, broadcast = true,
      immediate_bf16 = 0 : i32{second}, {common}}} : () -> ()
    func.return
  }}
}}
'''
    commands = lower_vector_commands(mlir, profile)
    if len(commands) != 1 or commands[0].funct != 33 or commands[0].rs2.immediate & 0xf != {
            "expsub": 12, "expsum": 13}[kind]:
        raise ValueError("typed fused operation did not lower to the selected VPU command")
    manifest["manifest_sha256"] = manifest_sha
    manifest["bound_mlir_sha256"] = digest(mlir.encode())
    return mlir, manifest


def commands(kind: str, mlir: str, profile: dict) -> list[Command]:
    vector = lower_vector_commands(mlir, profile)
    stream = [_cmd(7, 0, 0), _config_ld(16), _config_st(16)]
    for name, spad in (("a", SP_A), ("b", SP_B)):
        for row in range(0, ROWS, 16):
            stream.append(_transfer(2, name, row * 16, spad + row))
    stream.extend(vector)
    for row in range(0, ROWS, 16):
        stream.append(_transfer(3, "output", row * 16, SP_D + row))
    if kind == "expsum":
        stream.append(_transfer(3, "sums", 0, SP_SUM))
    return stream


def driver(kind: str) -> str:
    op = "VPU_EXPSUM" if kind == "expsum" else "VPU_EXPSUB"
    sums_check = '''
  vpu_ref_exec(VPU_RSUM, sum_ref, ref, 0, 64, 4, 0, 0);
  for (int r = 0; r < 16; r++)
    for (int l = 0; l < VPU_LANES; l++)
      if (sums[r][l] != sum_ref[r][l]) sum_bad++;
''' if kind == "expsum" else ""
    return f'''#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"

static uint16_t a[64][VPU_LANES] __attribute__((aligned(64)));
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
void mx_issue(const void *a, const void *b, const void *output, const void *sums);

int main(void) {{
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++) {{
      a[r][l] = rand_bf16(110, 140, 1);
      (void)rand_bf16(110, 140, 1);  // A2 in Nicolas's source test
      b[r][l] = rand_bf16(110, 140, 1);
      (void)rand_bf16(100, 154, 0);  // P in Nicolas's source test
      uint16_t x;
      do x = rand_bf16(100, 133, 1);
      while (vpu_bf16_to_f(x) > 80.0f || vpu_bf16_to_f(x) < -80.0f);
    }}
  mx_issue(a, b, output, sums);
  gemmini_fence();
  vpu_ref_exec({op}, ref, a, b, 64, 4, 1, 0);
  int out_bad = 0, sum_bad = 0;
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++)
      if (output[r][l] != ref[r][l]) out_bad++;
{sums_check}  printf("compiled fused {kind}: %d output mismatches, %d sum mismatches\\n",
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
        parser.error("selected Nicolas profile lacks fused EXPSUB/EXPSUM")
    source = software / "bareMetalC/vpu_ops.c"
    oracle = software / "include/vpu_ref.h"
    baseline = json.loads(SOURCE_RECEIPT.read_text())
    if (_sha(source) != baseline["source_sha256"] or
            _sha(oracle) != baseline["reference_sha256"] or
            baseline["check_count"] != 29):
        parser.error("Nicolas VPU source or oracle differs from qualified baseline")
    source_text = source.read_text()
    input_lines = (
        "static uint32_t lcg = 12345;",
        "A[r][l] = rand_bf16(110, 140, 1);",
        "A2[r][l] = rand_bf16(110, 140, 1);",
        "B[r][l] = rand_bf16(110, 140, 1);",
        "P[r][l] = rand_bf16(100, 154, 0);",
        "do x = rand_bf16(100, 133, 1);",
        'run("expsub bcast",',
        'run(t ? "expsum sb" : "expsum bcast",',
    )
    if any(source_text.count(line) != 1 for line in input_lines):
        parser.error("Nicolas fused VPU inputs or selected calls changed")
    cc, spike = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike"
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        parser.error("RISC-V GCC, Spike, and host g++ are required")
    bench = software / "riscv-tests/benchmarks/common"
    sys.path.insert(0, str(model2mlir))
    import m2m
    import torch

    if Path(m2m.__file__).resolve().parents[1] != model2mlir:
        parser.error("model2MLIR resolved to a different checkout")
    out.mkdir(parents=True)
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = out / "libgemmini.so"
    _run(["g++", "-L", str(riscv / "lib"),
          f"-Wl,-rpath,{riscv / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(riscv / "include"), "-fPIC", "-O3",
          f"-ffile-prefix-map={extension}=software/libgemmini",
          *(str(path) for path in extension_sources)],
         cwd=out, log=out / "extension_build.log")
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             f"-ffile-prefix-map={out}=.",
             f"-ffile-prefix-map={software}=software/gemmini-rocc-tests",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    common_sources = sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))
    common_objects = []
    for index, path in enumerate(common_sources):
        obj = out / f"common_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(obj)],
             cwd=out, log=out / f"common_{index}.log")
        common_objects.append(obj)
    rows = []
    for kind in ("expsub", "expsum"):
        directory = out / kind
        directory.mkdir()
        frontend = capture(kind, m2m, torch)
        mlir, binding = bind(kind, frontend, _sha(source), profile)
        (directory / "frontend.mlir").write_text(frontend)
        (directory / "bound.mlir").write_text(mlir)
        (directory / "binding.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
        stream = commands(kind, mlir, profile)
        if [command.funct for command in stream] != [7, 0, 0, *([2] * 8), 33,
                                                      *([3] * (5 if kind == "expsum" else 4))]:
            raise ValueError("fused VPU physical stream differs from selected source schedule")
        (directory / "mx_issue.c").write_text(emit_c(
            stream, transport="rocket_rocc", buffers=("a", "b", "output", "sums")))
        (directory / "mx_driver.c").write_text(driver(kind))
        objects = []
        for index, name in enumerate(("mx_issue.c", "mx_driver.c")):
            obj = directory / f"program_{index}.o"
            _run([str(cc), *flags, "-c", str(directory / name), "-o", str(obj)],
                 cwd=directory, log=directory / f"compile_{index}.log")
            objects.append(obj)
        elf = directory / "fused.elf"
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
        expected = f"compiled fused {kind}: 0 output mismatches, 0 sum mismatches"
        if result.returncode or expected not in result.stdout:
            raise RuntimeError(f"compiler-issued {kind} differs from source oracle: {log}")
        rows.append({
            "kind": kind, "status": "source_vpu_reference_matched_on_pinned_spike",
            "frontend_mlir_sha256": _sha(directory / "frontend.mlir"),
            "bound_mlir_sha256": _sha(directory / "bound.mlir"),
            "binding_sha256": _sha(directory / "binding.json"),
            "issuer_sha256": _sha(directory / "mx_issue.c"),
            "driver_sha256": _sha(directory / "mx_driver.c"),
            "elf_sha256": _sha(elf), "spike_log_sha256": _sha(log),
            "ordered_functs": [command.funct for command in stream],
            "compared_output_bf16": 512,
            "compared_sum_bf16": 128 if kind == "expsum" else 0,
        })
    index = {
        "schema": "mx_gemmini.nicolas_fused_vpu_compiled_spike.v1",
        "status": "compiler_fused_expsub_expsum_matched_source_reference",
        "source_qualification_sha256": _sha(SOURCE_RECEIPT),
        "source_sha256": _sha(source), "reference_sha256": _sha(oracle),
        "source_revision": _git_revision(software),
        "rtl_revision": _git_revision(rtl),
        "extension_revision": _git_revision(extension),
        "model2mlir_revision": _git_revision(model2mlir),
        "profile_sha256": profile_sha256(profile),
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "riscv_gcc_sha256": _sha(cc), "spike_sha256": _sha(spike),
        "extension_sha256": _sha(so), "rows": rows,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print("compiler-issued EXPSUB: 512/512 BF16; EXPSUM: 512/512 BF16 + 128/128 sums")


if __name__ == "__main__":
    main()
