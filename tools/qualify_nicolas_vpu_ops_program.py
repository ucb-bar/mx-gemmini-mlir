"""Compile Nicolas's entire VPU operation schedule into one RV64 object.

The driver reproduces his deterministic inputs and calls his bit-exact
``vpu_ref.h`` oracle. All accelerator commands come from typed MLIR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha
from tools.qualify_nicolas_vpu_public_objects import _build_extension, _flags


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/evidence/nicolas_vpu_source_all_ops_266c593/receipt.json"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json"
FP4_PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
N = 64
SP_A, SP_A2, SP_B, SP_D, SP_R = 0, 0x800, 0x1000, 0x2000, 0x3000


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _schedule() -> tuple[list[dict], list[dict], list[str], list[dict]]:
    operations: list[dict] = []
    captures: list[dict] = []
    references: list[str] = []

    def capture(label: str, row: int, rows: int, statement: str) -> None:
        index = len(captures)
        captures.append({"name": f"out_{index}", "label": label, "row": row,
                         "rows": rows, "after_operation": len(operations) - 1})
        references.append(statement.format(out=f"expected[{index}]") + ";")

    def add(label: str, kind: str, dst: int, src1: int, src2: int = 0,
            *, rlen: int = 1, bcast: bool = False, imm: int = 0,
            second: int | None = None, ref1: str | None = None,
            ref2: str = "0", save: bool = True) -> None:
        operations.append({"label": label, "kind": kind, "dst_row": dst,
                           "src1_row": src1, "src2_row": src2, "rows": N,
                           "reduction_length": rlen, "broadcast": bcast,
                           "immediate_bf16": imm, "second_dst_row": second})
        if save:
            output_rows = N // rlen if kind in ("rmax", "ramax", "rsum") else N
            source = ref1 if ref1 is not None else "A"
            capture(label, dst, output_rows,
                    f"vpu_ref_exec(VPU_{kind.upper()}, {{out}}, {source}, {ref2}, "
                    f"{N}, {rlen}, {int(bcast)}, {imm if kind != 'expsum' else second})")

    for label, kind, src2, ref2, rlen, bcast in (
            ("add", "add", SP_B, "B", 1, False),
            ("sub", "sub", SP_B, "B", 1, False),
            ("mul", "mul", SP_B, "B", 1, False),
            ("mul same-bank", "mul", SP_A2, "A2", 1, False),
            ("mul bcast", "mul", SP_B, "B", 4, True),
            ("max", "max", SP_B, "B", 1, False),
            ("max same-bank", "max", SP_A2, "A2", 1, False),
            ("max bcast", "max", SP_B, "B", 4, True),
            ("sub bcast sb", "sub", SP_A2, "A2", 4, True),
            ("expsub", "expsub", SP_B, "B", 1, False),
            ("expsub bcast", "expsub", SP_B, "B", 4, True),
            ("expsub sb", "expsub", SP_A2, "A2", 4, True)):
        add(label, kind, SP_D, SP_A, src2, rlen=rlen, bcast=bcast,
            ref1="A", ref2=ref2)
    for label, src2, ref2 in (("expsum bcast", SP_B, "B"),
                              ("expsum sb", SP_A2, "A2")):
        add(label, "expsum", SP_D, SP_A, src2, rlen=4, bcast=True,
            second=SP_R, ref1="A", ref2=ref2)
        capture("expsum sums" if src2 == SP_B else "expsum sb sums",
                SP_R, 16,
                f"vpu_ref_exec(VPU_RSUM, {{out}}, expected[{len(captures) - 1}], "
                "0, 64, 4, 0, 0)")
    for label, kind, dst, src, rlen, imm, source in (
            ("adds", "adds", SP_D, SP_A, 1, 0x4040, "A"),
            ("muls", "muls", SP_D, SP_A, 1, 0x3f00, "A"),
            ("exp", "exp", SP_D, SP_R + 0x400, 1, 0, "X"),
            ("rcp", "rcp", SP_D, SP_A, 1, 0, "A"),
            ("rsqrt", "rsqrt", SP_R, SP_D + 0x400, 1, 0, "P"),
            ("rmax", "rmax", SP_R, SP_A, 4, 0, "A"),
            ("ramax", "ramax", SP_R, SP_A, 4, 0, "A"),
            ("rsum", "rsum", SP_R, SP_A, 4, 0, "A"),
            ("rsum rlen1", "rsum", SP_R, SP_A, 1, 0, "A")):
        add(label, kind, dst, src, rlen=rlen, imm=imm, ref1=source)
    add("chain add", "add", SP_D, SP_A, SP_B, save=False)
    add("chain muls", "muls", SP_D, SP_D, imm=0x3f00, save=False)
    add("chain", "rmax", SP_R, SP_D, rlen=4, save=False)
    chain_index = len(captures)
    capture("chain", SP_R, 16,
            "vpu_ref_exec(VPU_ADD, scratch, A, B, 64, 1, 0, 0);\n"
            "  vpu_ref_exec(VPU_MULS, scratch, scratch, 0, 64, 1, 0, 0x3f00);\n"
            "  vpu_ref_exec(VPU_RMAX, {out}, scratch, 0, 64, 4, 0, 0)")
    assert len(captures) == chain_index + 1
    add("war mvin", "add", SP_D, SP_A, SP_B, ref1="A", ref2="B")
    war_index = len(operations) - 1
    add("dual X", "adds", SP_A + 0x400, SP_A, imm=0x4040,
        ref1="A2", save=False)
    add("dual Y", "exp", SP_R, SP_R + 0x400, ref1="X", save=False)
    add("dual Z=X*B", "mul", SP_D, SP_A + 0x400, SP_B,
        ref1="A2", ref2="B", save=False)
    x = len(captures)
    capture("dual X", SP_A + 0x400, N,
            "vpu_ref_exec(VPU_ADDS, {out}, A2, 0, 64, 1, 0, 0x4040)")
    capture("dual Y", SP_R, N,
            "vpu_ref_exec(VPU_EXP, {out}, X, 0, 64, 1, 0, 0)")
    capture("dual Z=X*B", SP_D, N,
            f"vpu_ref_exec(VPU_MUL, {{out}}, expected[{x}], B, 64, 1, 0, 0)")
    if len(operations) != 30 or len(captures) != 30 or war_index != 26:
        raise AssertionError("Nicolas VPU source schedule size differs")
    reloads = [{"input": "a2", "row": SP_A, "after_operation": war_index}]
    return operations, captures, references, reloads


def _mlir(operations: list[dict], source_sha: str, oracle_sha: str,
          schedule_sha: str, profile_sha: str) -> str:
    body = []
    for index, op in enumerate(operations):
        second = (f', second_dst_row = {op["second_dst_row"]} : i32'
                  if op["second_dst_row"] is not None else "")
        body.append(
            f'    "mx_gemmini.vpu_execute"() {{site_id = "vpu_ops:{index}", '
            f'kind = "{op["kind"]}", src1_row = {op["src1_row"]} : i32, '
            f'src2_row = {op["src2_row"]} : i32, dst_row = {op["dst_row"]} : i32, '
            f'rows = {N} : i32, reduction_length = {op["reduction_length"]} : i32, '
            f'broadcast = {str(op["broadcast"]).lower()}, '
            f'immediate_bf16 = {op["immediate_bf16"]} : i32{second}, '
            f'profile_sha256 = "{profile_sha}", contract_sha256 = "{source_sha}", '
            f'policy_sha256 = "{oracle_sha}", manifest_sha256 = "{schedule_sha}"}} '
            ': () -> ()')
    return (
        f'builtin.module attributes {{mx.profile_sha256 = "{profile_sha}", '
        f'mx.contract_sha256 = "{source_sha}", mx.policy_sha256 = "{oracle_sha}", '
        f'prov.quantization_manifest_sha256 = "{schedule_sha}", '
        'mx.vpu_sequence_schema = "mx_gemmini.vpu_sequence.v1"} {\n'
        '  func.func @nicolas_vpu_ops() {\n' + '\n'.join(body) +
        '\n    func.return\n  }\n}\n')


def _abi(captures: list[dict], reloads: list[dict]) -> dict:
    return {"schema": "mx_gemmini.vpu_sequence_buffer_map.v2",
            "inputs": [{"name": name, "row": row, "rows": N} for name, row in
                       (("a", SP_A), ("a2", SP_A2), ("b", SP_B),
                        ("p", SP_D + 0x400), ("x", SP_R + 0x400))],
            "outputs": [{key: value for key, value in item.items() if key != "label"}
                        for item in captures], "reloads": reloads}


def _driver(captures: list[dict], references: list[str]) -> str:
    outputs = ', '.join(f'const void *out_{index}' for index in range(len(captures)))
    call = ', '.join(f'observed[{index}]' for index in range(len(captures)))
    refs = '\n  '.join(references)
    checks = '\n'.join(
        f'  for (int r = 0; r < {capture["rows"]}; ++r) '
        f'for (int l = 0; l < VPU_LANES; ++l) '
        f'bad[{index}] += observed[{index}][r][l] != expected[{index}][r][l];\n'
        f'  printf("{capture["label"]}: %d mismatches\\n", bad[{index}]);'
        for index, capture in enumerate(captures))
    return f'''#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "include/vpu_ref.h"
#define N 64
static uint16_t A[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t A2[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t B[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t P[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t X[N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t observed[{len(captures)}][N][VPU_LANES] __attribute__((aligned(64)));
static uint16_t expected[{len(captures)}][N][VPU_LANES];
static uint16_t scratch[N][VPU_LANES];
static uint32_t lcg = 12345;
static uint32_t rnd(void) {{ lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }}
static uint16_t rand_bf16(int elo, int ehi, int sign) {{
  uint16_t s = (sign && (rnd() & 1)) ? 0x8000 : 0;
  return (uint16_t)(s | ((elo + rnd() % (ehi - elo + 1)) << 7) | (rnd() & 0x7f));
}}
void mx_issue(const void *a, const void *a2, const void *b, const void *p,
              const void *x, {outputs});
int main(void) {{
  for (int r = 0; r < N; r++)
    for (int l = 0; l < VPU_LANES; l++) {{
      A[r][l] = rand_bf16(110, 140, 1);
      A2[r][l] = rand_bf16(110, 140, 1);
      B[r][l] = rand_bf16(110, 140, 1);
      P[r][l] = rand_bf16(100, 154, 0);
      uint16_t x;
      do x = rand_bf16(100, 133, 1); while (vpu_bf16_to_f(x) > 80.0f || vpu_bf16_to_f(x) < -80.0f);
      X[r][l] = x;
    }}
  X[0][0] = 0x0000; X[0][1] = 0x8000; X[0][2] = 0x7f80;
  X[0][3] = 0xff80; X[0][4] = 0x7fc1; X[0][5] = 0x0001;
  P[0][0] = 0x0000; P[0][1] = 0x7f80; P[0][2] = 0x0001;
  memset(observed, 0xa5, sizeof(observed));
  mx_issue(A, A2, B, P, X, {call});
  {refs}
  int bad[{len(captures)}] = {{0}};
{checks}
  int total = 0;
  for (int i = 0; i < {len(captures)}; ++i) total += bad[i];
  printf("compiled vpu_ops: %d mismatches across {len(captures)} output snapshots\\n", total);
  return total != 0;
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    args = parser.parse_args()
    rtl, riscv, out = (args.rtl_root.resolve(), args.riscv_root.resolve(),
                       args.out_dir.resolve())
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    profile_path = args.profile.resolve()
    if profile_path not in {PROFILE, FP4_PROFILE}:
        parser.error("VPU program needs one of Nicolas's two pinned VPU Rocket profiles")
    if not args.mx_opt.is_file():
        parser.error("native MX verifier is absent")
    baseline = json.loads(BASELINE.read_text())
    software = rtl / "software/gemmini-rocc-tests"
    source = software / "bareMetalC/vpu_ops.c"
    oracle = software / "include/vpu_ref.h"
    if (_git_revision(rtl) != baseline["rtl_revision"] or
            _sha(source) != baseline["source_sha256"] or
            _sha(oracle) != baseline["reference_sha256"] or
            baseline["check_count"] != 29 or baseline["spike_exit_code"] != 0):
        raise ValueError("Nicolas VPU source baseline differs")
    _require_gitlink(rtl, "software/gemmini-rocc-tests")
    _require_gitlink(rtl, "software/libgemmini")
    profile = load_profile(profile_path, rtl_root=rtl)
    ops, captures, references, reloads = _schedule()
    source_labels = [name for label in baseline["checks"] for name in
                     (["dual X", "dual Y"] if label == "dual X,Y" else [label])]
    if [item["label"] for item in captures] != source_labels:
        raise ValueError("compiler VPU snapshots differ from Nicolas's 29 checks")
    schedule = {"operations": ops, "captures": captures, "reloads": reloads}
    schedule_sha = _digest(json.dumps(schedule, sort_keys=True,
                                      separators=(",", ":")).encode())
    bound = _mlir(ops, baseline["source_sha256"], baseline["reference_sha256"],
                  schedule_sha, profile_sha256(profile))
    abi = _abi(captures, reloads)
    cc, spike = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike"
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        raise ValueError("VPU replay needs RISC-V GCC, Spike, and host g++")
    bench = software / "riscv-tests/benchmarks/common"
    if not (bench / "test.ld").is_file():
        raise ValueError("Nicolas benchmark linker script is absent")
    out.mkdir(parents=True)
    mlir_file, abi_file, driver = (out / "vpu_ops.mlir", out / "abi.json",
                                   out / "mx_driver.c")
    mlir_file.write_text(bound)
    abi_file.write_text(json.dumps(abi, indent=2, sort_keys=True) + "\n")
    driver.write_text(_driver(captures, references))
    object_dir = out / "object"
    compile_run = subprocess.run(
        [sys.executable, "-m", "tools.compile_object", "--mlir", str(mlir_file),
         "--profile", str(profile_path), "--rtl-root", str(rtl),
         "--riscv-root", str(riscv), "--mx-opt", str(args.mx_opt.resolve()),
         "--abi-json", str(abi_file), "--out-dir", str(object_dir)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "compiler.log").write_text(compile_run.stdout)
    if compile_run.returncode:
        raise RuntimeError("public VPU program object compilation failed")
    object_manifest = json.loads((object_dir / "object_manifest.json").read_text())
    if (object_manifest.get("operation_count") != 30 or
            object_manifest.get("buffer_map_schema") != abi["schema"] or
            object_manifest.get("allocated_data_section_bytes") != 0):
        raise ValueError("VPU program object is not data-free and complete")
    so = _build_extension(rtl / "software/libgemmini", riscv, out)
    flags = _flags(software, bench, out)
    sources = [driver, *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S"))]
    objects = []
    for index, path in enumerate(sources):
        obj = out / f"driver_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(obj)],
             cwd=out, log=out / f"driver_{index}.log")
        objects.append(obj)
    elf = out / "vpu_ops.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), str(object_dir / "mx_issue.o"),
          *(str(path) for path in objects), "-lm", "-lgcc", "-o", str(elf)],
         cwd=out, log=out / "link.log")
    result = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini",
                             str(elf)], cwd=out, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = out / "spike.log"
    log.write_text(result.stdout)
    count = sum(item["rows"] * 8 for item in captures)
    if (result.returncode or
            "compiled vpu_ops: 0 mismatches across 30 output snapshots" not in result.stdout or
            sum(line.endswith(": 0 mismatches") for line in
                result.stdout.splitlines()) != 30):
        raise RuntimeError(f"compiler VPU program differs from source oracle; see {log}")
    receipt = {"schema": "mx_gemmini.nicolas_vpu_program_spike.v1",
               "status": "compiler_vpu_program_matched_source_reference_on_pinned_spike",
               "rtl_revision": _git_revision(rtl),
               "source_sha256": baseline["source_sha256"],
               "reference_sha256": baseline["reference_sha256"],
               "baseline_receipt_sha256": _sha(BASELINE),
               "profile_sha256": profile_sha256(profile),
               "compiler_revision": _git_revision(ROOT),
               "bound_mlir_sha256": _sha(mlir_file),
               "abi_json_sha256": _sha(abi_file),
               "driver_sha256": _sha(driver),
               "object_sha256": _sha(object_dir / "mx_issue.o"),
               "object_manifest_sha256": _sha(object_dir / "object_manifest.json"),
               "physical_program_sha256": _sha(object_dir / "physical_program.json"),
               "elf_sha256": _sha(elf), "spike_log_sha256": _sha(log),
               "extension_sha256": _sha(so), "spike_sha256": _sha(spike),
               "operation_count": len(ops), "snapshot_count": len(captures),
               "source_check_count": baseline["check_count"],
               "bf16_values_checked": count, "mismatches": 0,
               "scope": "one typed 30-command VPU stream and 30 source-oracle output snapshots"}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(out / "receipt.json")


if __name__ == "__main__":
    main()
