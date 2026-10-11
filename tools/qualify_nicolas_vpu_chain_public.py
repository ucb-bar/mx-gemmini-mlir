"""Capture and replay Nicolas's dependent ADD→MULS→RMAX VPU source chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vpu_sequence_program import lower_vpu_sequence
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure
from tools.qualify_nicolas_vpu_public_objects import (
    PROFILE, FP4_VPU_PROFILE, _build_extension, _flags,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RECEIPT = ROOT / "docs/evidence/nicolas_vpu_source_all_ops_266c593/receipt.json"
MODEL2MLIR_REVISION = "e9ded36eb85abf2d9097ac4dc11457c825853388"
MODEL2MLIR_CLOSURE = "559efcf9232c1375df1fb12bd131edc0e3daba7d1364df25725bb2ef7a0e2027"
CHAIN_MARKERS = (
    "#define N 64",
    "#define RL 4",
    "#define SP_A   0x0000",
    "#define SP_B   0x1000",
    "#define SP_D   0x2000",
    "#define SP_R   0x3000",
    "const uint16_t bf_half = 0x3f00",
    "gemmini_vpu(VPU_ADD,  SP_D, SP_A, SP_B, N, 1, 0, 0);",
    "gemmini_vpu(VPU_MULS, SP_D, SP_D, 0,    N, 1, 0, bf_half);",
    "gemmini_vpu(VPU_RMAX, SP_R, SP_D, 0,    N, RL, 0, 0);",
    "vpu_ref_exec(VPU_ADD,  out_ref, A, B, N, 1, 0, 0);",
    "vpu_ref_exec(VPU_MULS, out_ref, out_ref, 0, N, 1, 0, bf_half);",
    "vpu_ref_exec(VPU_RMAX, out_ref, out_ref, 0, N, RL, 0, 0);",
)
ABI = {
    "schema": "mx_gemmini.vpu_sequence_buffer_map.v1",
    "inputs": [{"name": "a", "row": 0, "rows": 64},
               {"name": "b", "row": 4096, "rows": 64}],
    "outputs": [{"name": "middle", "row": 8192, "rows": 64},
                {"name": "result", "row": 12288, "rows": 16}],
}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _capture(model2mlir: Path) -> str:
    if _source_closure(model2mlir, list((model2mlir / "m2m").rglob("*.py"))) != MODEL2MLIR_CLOSURE:
        raise ValueError("model2MLIR source differs from pinned upstream main")
    sys.path.insert(0, str(model2mlir))
    import m2m
    from m2m.coverage import opaque_report
    import torch
    if not Path(m2m.__file__).resolve().is_relative_to(model2mlir):
        raise ValueError("model2MLIR capture imported a different source tree")

    class Chain(torch.nn.Module):
        def forward(self, a, b):
            return ((a + b) * 0.5).reshape(16, 4, 8).amax(dim=(1, 2))

    sample = torch.zeros((64, 8), dtype=torch.bfloat16)
    captured = m2m.convert(Chain().eval(), (sample, sample), backend="fx_importer")
    if not captured.ok:
        raise ValueError(f"model2MLIR VPU chain capture failed: {captured.diagnostics}")
    frontend = captured.mlir_text
    if (opaque_report(frontend) or "torch.operator" in frontend or
            "func.call" in frontend or
            "func.func @forward(%0: tensor<64x8xbf16>, %1: tensor<64x8xbf16>)" not in frontend or
            not all(marker in frontend for marker in (
                'prov.aten = "aten.add.Tensor"',
                'prov.aten = "aten.mul.Tensor"',
                'prov.aten = "aten.amax.default"',
                "5.000000e-01", "linalg.reduce", "dimensions = [1, 2]",
                "tensor<16xbf16>"))):
        raise ValueError("model2MLIR VPU chain differs from ADD→MULS→RMAX policy")
    return frontend


def _bind(frontend: str, source_sha: str, reference_sha: str, profile: dict) -> str:
    profile_sha = profile_sha256(profile)
    contract = _digest(bytes.fromhex(source_sha) + bytes.fromhex(reference_sha))
    policy = _digest(b"nicolas_vpu_ops_chain:add;muls_3f00;rmax_rlen4;rows64")
    manifest = _digest(frontend.encode() + bytes.fromhex(source_sha))
    common = (f'profile_sha256 = "{profile_sha}", contract_sha256 = "{contract}", '
              f'policy_sha256 = "{policy}", manifest_sha256 = "{manifest}"')
    ops = (("add", 0, 4096, 8192, 1, 0),
           ("muls", 8192, 0, 8192, 1, 0x3f00),
           ("rmax", 8192, 0, 12288, 4, 0))
    body = "\n".join(
        f'    "mx_gemmini.vpu_execute"() {{site_id = "vpu:chain:{index}", '
        f'kind = "{kind}", src1_row = {src1} : i32, src2_row = {src2} : i32, '
        f'dst_row = {dst} : i32, rows = 64 : i32, '
        f'reduction_length = {rlen} : i32, broadcast = false, '
        f'immediate_bf16 = {imm} : i32, {common}}} : () -> ()'
        for index, (kind, src1, src2, dst, rlen, imm) in enumerate(ops))
    mlir = (
        f'builtin.module attributes {{mx.profile_sha256 = "{profile_sha}", '
        f'mx.contract_sha256 = "{contract}", mx.policy_sha256 = "{policy}", '
        f'prov.quantization_manifest_sha256 = "{manifest}", '
        f'mx.source_mlir_sha256 = "{_digest(frontend.encode())}", '
        f'mx.source_sha256 = "{source_sha}", '
        'mx.vpu_sequence_schema = "mx_gemmini.vpu_sequence.v1"} {\n'
        '  func.func @nicolas_vpu_chain() {\n' + body +
        '\n    func.return\n  }\n}\n')
    plan = lower_vpu_sequence(mlir, profile, ABI)
    if [item["kind"] for item in plan.operations] != ["add", "muls", "rmax"]:
        raise ValueError("Nicolas VPU chain binding lost an operation")
    return mlir


def _driver() -> str:
    return """#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"

static uint16_t a[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t a2[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t b[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t p[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t x[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t middle[64][VPU_LANES] __attribute__((aligned(64)));
static uint16_t result[16][VPU_LANES] __attribute__((aligned(64)));
static uint16_t middle_ref[64][VPU_LANES];
static uint16_t result_ref[64][VPU_LANES];
static uint32_t lcg = 12345;
static uint32_t rnd(void) { lcg = lcg * 1103515245u + 12345u; return lcg >> 8; }
static uint16_t rand_bf16(int elo, int ehi, int sign) {
  uint16_t s = (sign && (rnd() & 1)) ? 0x8000 : 0;
  return (uint16_t)(s | ((elo + rnd() % (ehi - elo + 1)) << 7) | (rnd() & 0x7f));
}
void mx_issue(const void *a, const void *b, const void *middle, const void *result);

int main(void) {
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++) {
      a[r][l] = rand_bf16(110, 140, 1);
      a2[r][l] = rand_bf16(110, 140, 1);
      b[r][l] = rand_bf16(110, 140, 1);
      p[r][l] = rand_bf16(100, 154, 0);
      uint16_t v;
      do v = rand_bf16(100, 133, 1);
      while (vpu_bf16_to_f(v) > 80.0f || vpu_bf16_to_f(v) < -80.0f);
      x[r][l] = v;
    }
  mx_issue(a, b, middle, result);
  gemmini_fence();
  vpu_ref_exec(VPU_ADD, middle_ref, a, b, 64, 1, 0, 0);
  vpu_ref_exec(VPU_MULS, middle_ref, middle_ref, 0, 64, 1, 0, 0x3f00);
  vpu_ref_exec(VPU_RMAX, result_ref, middle_ref, 0, 64, 4, 0, 0);
  int middle_bad = 0, result_bad = 0;
  for (int r = 0; r < 64; r++)
    for (int l = 0; l < VPU_LANES; l++)
      middle_bad += middle[r][l] != middle_ref[r][l];
  for (int r = 0; r < 16; r++)
    for (int l = 0; l < VPU_LANES; l++)
      result_bad += result[r][l] != result_ref[r][l];
  printf("compiled VPU chain: %d middle mismatches, %d result mismatches\\n",
         middle_bad, result_bad);
  return middle_bad || result_bad;
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--mx-opt", type=Path)
    args = parser.parse_args()
    model2mlir, rtl, riscv, out = (
        args.model2mlir_root.resolve(), args.rtl_root.resolve(),
        args.riscv_root.resolve(), args.out_dir.resolve())
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    selected_profile = args.profile.resolve()
    if selected_profile not in {PROFILE, FP4_VPU_PROFILE}:
        parser.error("source chain supports only Nicolas's two VPU Rocket profiles")
    software, extension = (rtl / "software/gemmini-rocc-tests",
                           rtl / "software/libgemmini")
    _require_gitlink(rtl, "software/gemmini-rocc-tests")
    _require_gitlink(rtl, "software/libgemmini")
    source, reference = (software / "bareMetalC/vpu_ops.c",
                         software / "include/vpu_ref.h")
    baseline = json.loads(SOURCE_RECEIPT.read_text())
    source_text = source.read_text()
    if (_git_revision(rtl) != baseline["rtl_revision"] or
            _sha(source) != baseline["source_sha256"] or
            _sha(reference) != baseline["reference_sha256"] or
            baseline.get("status") !=
            "all_29_source_vpu_checks_matched_on_pinned_spike" or
            "chain" not in baseline.get("checks", []) or
            any(marker not in source_text for marker in CHAIN_MARKERS)):
        parser.error("Nicolas VPU source or pinned chain markers differ")
    profile = load_profile(selected_profile, rtl_root=rtl)
    if (profile.get("transport") != "rocket_rocc" or
            profile["resources"].get("vpu_config") !=
            {"units": 2, "exp_sub": True, "exp_sum": True}):
        parser.error("selected VPU profile lacks Nicolas's two-unit VPU")
    cc, spike = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike"
    bench = software / "riscv-tests/benchmarks/common"
    if not cc.is_file() or not spike.is_file() or not (bench / "test.ld").is_file():
        parser.error("RISC-V tools or benchmark linker script are absent")
    if shutil.which("g++") is None:
        parser.error("host g++ is required")
    out.mkdir(parents=True)
    frontend = _capture(model2mlir)
    (out / "frontend.mlir").write_text(frontend)
    bound = _bind(frontend, baseline["source_sha256"],
                  baseline["reference_sha256"], profile)
    (out / "bound.mlir").write_text(bound)
    (out / "abi.json").write_text(json.dumps(ABI, indent=2, sort_keys=True) + "\n")
    object_dir = out / "object"
    command = [sys.executable, "-m", "tools.compile_object",
               "--mlir", str(out / "bound.mlir"),
               "--profile", str(selected_profile),
               "--rtl-root", str(rtl), "--riscv-root", str(riscv),
               "--abi-json", str(out / "abi.json"), "--out-dir", str(object_dir)]
    if args.mx_opt is not None:
        command += ["--mx-opt", str(args.mx_opt.resolve())]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "TMPDIR": str(out)},
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "compiler.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError("public VPU sequence object compilation failed")
    object_manifest = json.loads((object_dir / "object_manifest.json").read_text())
    if (object_manifest.get("schema") !=
            "mx_gemmini.vpu_sequence_linkable_object.v1" or
            object_manifest.get("operation_count") != 3 or
            object_manifest.get("allocated_data_section_bytes") != 0):
        raise ValueError("public VPU sequence object manifest differs")
    (out / "mx_driver.c").write_text(_driver())
    flags = _flags(software, bench, out)
    driver_obj = out / "driver.o"
    _run([str(cc), *flags, "-c", str(out / "mx_driver.c"), "-o",
          str(driver_obj)], cwd=out, log=out / "driver_compile.log")
    common = []
    for index, path in enumerate(sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))):
        obj = out / f"common_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(obj)],
             cwd=out, log=out / f"common_{index}.log")
        common.append(obj)
    elf = out / "vpu_chain.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), str(object_dir / "mx_issue.o"),
          str(driver_obj), *(str(path) for path in common),
          "-lm", "-lgcc", "-o", str(elf)], cwd=out, log=out / "link.log")
    so = _build_extension(extension, riscv, out)
    sim = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini",
                          str(elf)], cwd=out, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / "spike.log").write_text(sim.stdout)
    if (sim.returncode != 0 or
            "compiled VPU chain: 0 middle mismatches, 0 result mismatches" not in sim.stdout):
        raise RuntimeError("compiler VPU chain differs from Nicolas's source reference")
    receipt = {
        "schema": "mx_gemmini.nicolas_vpu_chain_public_spike.v1",
        "status": "source_chain_matched_by_public_object_on_pinned_spike",
        "scope": "Nicolas VPU ADD→MULS→RMAX chain; full intermediate and final BF16 outputs",
        "source_sha256": baseline["source_sha256"],
        "reference_sha256": baseline["reference_sha256"],
        "source_receipt_sha256": _sha(SOURCE_RECEIPT),
        "rtl_revision": _git_revision(rtl),
        "model2mlir_revision": MODEL2MLIR_REVISION,
        "model2mlir_source_closure_sha256": MODEL2MLIR_CLOSURE,
        "compiler_revision": _git_revision(ROOT),
        "profile_sha256": profile_sha256(profile),
        "frontend_mlir_sha256": _sha(out / "frontend.mlir"),
        "bound_mlir_sha256": _sha(out / "bound.mlir"),
        "abi_sha256": _sha(out / "abi.json"),
        "driver_sha256": _sha(out / "mx_driver.c"),
        "object_sha256": _sha(object_dir / "mx_issue.o"),
        "object_manifest_sha256": _sha(object_dir / "object_manifest.json"),
        "physical_program_sha256": _sha(object_dir / "physical_program.json"),
        "elf_sha256": _sha(elf), "spike_log_sha256": _sha(out / "spike.log"),
        "extension_sha256": _sha(so), "spike_sha256": _sha(spike),
        "riscv_gcc_sha256": _sha(cc), "spike_exit_code": sim.returncode,
        "compared_middle_bf16": 512, "compared_result_bf16": 128,
        "mismatches": 0,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(out / "receipt.json")


if __name__ == "__main__":
    main()
