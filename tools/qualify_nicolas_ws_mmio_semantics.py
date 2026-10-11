"""Compile Nicolas's legacy FP6 MMIO debug data through current model2MLIR.

This qualifies the header's numerical matmul and output projection on Rocket
Spike. The original C program's fixed MMIO issue path is a separate target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.bind_payload import bind_payload
from mx_gemmini_support.quant_reference import quantize_bf16_radiance_header_fp6
from mx_gemmini_support.source_gemm import SourceGemm
from mx_gemmini_support.source_payload import (NICOLAS_SOURCE_HEADER_ORIGIN,
                                               read_source_payload, write_bundle)
from mx_gemmini_support.standard_matmul_handoff import bind_standard_matmul
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure
from tools.qualify_nicolas_vpu_public_objects import _build_extension, _flags


ROOT = Path(__file__).resolve().parents[1]
RTL_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
MODEL2MLIR_REVISION = "a042643e31366724ca0482389c4ee85a9f88e343"
MODEL2MLIR_CLOSURE = "6eb6648cf2eebd72cd481dff084b0e154e034fbe50b7bce31935091b1a237e39"
SOURCE_SHA256 = "dcbf940fbe7c40b135cb1f8e656196d0187aac326cfef673ed01012f45d9fb47"
HEADER_SHA256 = "127e962daebcbd890b4d76c9f884453e88c37aa5e1d9034954e7bb1cdfd42751"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE3M2OnlyGemminiRocketConfig.json"
SUMMARY = ("compiled Nicolas generic FP6 data: 0 bf16 / 16384, "
           "0 packed / 8192, 0 scale / 512 mismatches")


def source_kernel(rtl: Path) -> SourceGemm:
    if _git_revision(rtl) != RTL_REVISION:
        raise ValueError("generic FP6 qualification needs Nicolas's pinned RTL")
    for gitlink in ("software/gemmini-rocc-tests", "software/libgemmini"):
        _require_gitlink(rtl, gitlink)
    software = rtl / "software/gemmini-rocc-tests"
    driver = software / "bareMetalC/matmul_ws_mx_generic.c"
    header = software / "include/matmul_data_mx_lut_hw.h"
    if _sha(driver) != SOURCE_SHA256 or _sha(header) != HEADER_SHA256:
        raise ValueError("Nicolas generic FP6 source or header differs")
    source = driver.read_text()
    for marker in ('#define GEMMINI_CTRL 0x40084000',
                   '#define GEMMINI_RS1_ADDR (GEMMINI_CTRL + 0x10)',
                   '#define GEMMINI_RS2_ADDR (GEMMINI_CTRL + 0x18)',
                   'gemmini_loop_ws_spad(', 'C_proj_hw[i][j]'):
        if marker not in source:
            raise ValueError("Nicolas generic FP6 MMIO or checker path changed")
    kernel = SourceGemm(driver, header, (128, 128, 128), (128, 128, 128),
                        "FP6", True, False, True)
    resources = read_source_payload(kernel)
    codes, scales = quantize_bf16_radiance_header_fp6(
        resources["golden_bf16"].data, 128, 128, resources["output_lut"].data)
    if (codes != resources["source_fp6_packed"].data or
            scales != resources["golden_output_scales"].data):
        raise ValueError("generic FP6 source header output projection is inconsistent")
    return kernel


def capture(model2mlir_root: Path, directory: Path) -> str:
    import torch

    closure = _source_closure(model2mlir_root,
                             list((model2mlir_root / "m2m").rglob("*.py")))
    if closure != MODEL2MLIR_CLOSURE:
        raise ValueError("current model2MLIR source closure differs from pinned revision")
    sys.path.insert(0, str(model2mlir_root))
    import m2m
    if Path(m2m.__file__).resolve().parents[1] != model2mlir_root.resolve():
        raise ValueError("model2MLIR import differs from selected source root")

    class Matmul(torch.nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        inputs = (torch.randn(128, 128), torch.randn(128, 128))
    result = m2m.convert(Matmul().eval(), inputs, backend="fx_importer")
    if not result.ok or result.path_taken != "fx_importer":
        raise ValueError(f"current model2MLIR matmul capture failed: {result.diagnostics}")
    (directory / "model2mlir.mlir").write_text(result.mlir_text)
    (directory / "model2mlir_diagnostics.json").write_text(json.dumps(
        {"revision": MODEL2MLIR_REVISION, "source_closure_sha256": closure,
         "path_taken": result.path_taken, "diagnostics": result.diagnostics},
        indent=2, sort_keys=True) + "\n")
    return result.mlir_text


def _driver(names: list[str]) -> str:
    args = {"activation": "A_in_hw", "activation_lut": "A_lut",
            "activation_scales": "A_scales_row", "output_bf16": "C_bf16",
            "output_lut": "C_lut", "output_quantized": "C_packed",
            "scratch_output_scales": "C_scales", "weight": "B_in",
            "weight_lut": "B_lut", "weight_scales": "B_scales_col"}
    if set(names) != set(args):
        raise ValueError("generic FP6 object buffer ABI differs from selected header")
    return '''#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_data_mx_lut_hw.h"
#include "mx_issue.h"
static uint16_t C_bf16[128][128] __attribute__((aligned(64)));
static uint8_t C_packed[64][128] __attribute__((aligned(64)));
static uint8_t C_scales[512] __attribute__((aligned(64)));
int main(void) {
  mx_issue(''' + ", ".join(args[name] for name in names) + ''');
  gemmini_fence();
  int bf16 = 0, packed = 0, scale = 0;
  for (int i = 0; i < 128; ++i)
    for (int j = 0; j < 128; ++j)
      if (C_bf16[i][j] != C_out_bf16[i][j]) ++bf16;
  for (int i = 0; i < 64; ++i)
    for (int j = 0; j < 128; ++j)
      if (C_packed[i][j] != C_proj_hw[i][j]) ++packed;
  for (int i = 0; i < 128; ++i)
    for (int g = 0; g < 4; ++g)
      if (C_scales[i * 4 + g] != C_scales_row[g][i]) ++scale;
  printf("compiled Nicolas generic FP6 data: %d bf16 / 16384, %d packed / 8192, %d scale / 512 mismatches\\n",
         bf16, packed, scale);
  return bf16 || packed || scale;
}
'''


def run_spike(rtl: Path, riscv: Path, object_dir: Path, out: Path) -> tuple[Path, Path]:
    software = rtl / "software/gemmini-rocc-tests"
    bench = software / "riscv-tests/benchmarks/common"
    cc, spike = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/spike"
    if not cc.is_file() or not spike.is_file() or not (bench / "test.ld").is_file():
        raise ValueError("generic FP6 qualification lacks RISC-V GCC, Spike, or linker")
    build = out / "run"
    build.mkdir()
    abi = json.loads((object_dir / "object_manifest.json").read_text())["buffer_abi"]
    names = [slot["name"] for slot in abi]
    if [slot["position"] for slot in abi] != list(range(len(abi))):
        raise ValueError("generic FP6 object buffer positions differ")
    (build / "mx_driver.c").write_text(_driver(names))
    flags = [*_flags(software, bench, build), "-I", str(object_dir)]
    sources = [build / "mx_driver.c", *sorted(bench.glob("*.c")),
               *sorted(bench.glob("*.S"))]
    objects = []
    for index, source in enumerate(sources):
        obj = build / f"driver_{index}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(obj)],
             cwd=build, log=build / f"compile_{index}.log")
        objects.append(obj)
    elf = build / "mx_program.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), str(object_dir / "mx_issue.o"),
          *(str(obj) for obj in objects), "-lm", "-lgcc", "-o", str(elf)],
         cwd=build, log=build / "link.log")
    extension = _build_extension(rtl / "software/libgemmini", riscv, build)
    command = [str(spike), f"--extlib={extension}", "--extension=gemmini", str(elf)]
    run = subprocess.run(command, cwd=build, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=False)
    log = build / "spike.log"
    log.write_text(run.stdout)
    if run.returncode or SUMMARY not in run.stdout:
        raise ValueError("compiler-generated generic FP6 result differs from source header")
    return elf, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "rtl-root", "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    args = parser.parse_args()
    for name in ("model2mlir_root", "rtl_root", "riscv_root", "mx_opt",
                 "out_dir", "profile"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    kernel = source_kernel(args.rtl_root)
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if profile["name"] != "MxE3M2OnlyGemminiRocketConfig":
        raise ValueError("generic FP6 source needs Nicolas's E3M2 Rocket profile")
    args.out_dir.mkdir(parents=True)
    frontend = capture(args.model2mlir_root, args.out_dir)
    bundle = args.out_dir / "bundle"
    manifest = write_bundle(bundle, kernel, site_id="functional:matmul",
                            profile_sha256=profile_sha256(profile),
                            source_origin=NICOLAS_SOURCE_HEADER_ORIGIN)
    bound, binding = bind_standard_matmul(frontend, profile, manifest)
    (args.out_dir / "profile_bound.mlir").write_text(bound)
    (args.out_dir / "frontend_binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n")
    payload = bind_payload(bound, profile, manifest, source_header_quantized=True)
    mlir = args.out_dir / "payload_bound.mlir"
    mlir.write_text(payload)
    object_dir = args.out_dir / "object"
    _run([sys.executable, "-m", "tools.compile_object", "--mlir", str(mlir),
          "--bundle", str(bundle), "--profile", str(args.profile),
          "--rtl-root", str(args.rtl_root), "--riscv-root", str(args.riscv_root),
          "--mx-opt", str(args.mx_opt), "--out-dir", str(object_dir)],
         cwd=ROOT, log=args.out_dir / "object_compile.log")
    object_manifest = json.loads((object_dir / "object_manifest.json").read_text())
    if (object_manifest["embedded_operand_bytes"] != 0 or
            object_manifest["embedded_golden_bytes"] != 0 or
            object_manifest["allocated_data_section_bytes"] != 0 or
            object_manifest["transport"] != "rocket_rocc"):
        raise ValueError("generic FP6 issuer is not a data-free Rocket object")
    elf, log = run_spike(args.rtl_root, args.riscv_root, object_dir, args.out_dir)
    receipt = {"schema": "mx_gemmini.nicolas_ws_generic_portable_spike.v1",
               "status": "source_header_selected_outputs_matched_on_pinned_spike",
               "source_transport": "fixed_mmio", "compiled_transport": "rocket_rocc",
               "source_mmio_issue_qualification": "not_tested",
               "source_header_projection_policy": "radiance_header_fp6_lut_v1",
               "source_driver_sha256": SOURCE_SHA256, "source_header_sha256": HEADER_SHA256,
               "rtl_revision": RTL_REVISION, "model2mlir_revision": MODEL2MLIR_REVISION,
               "model2mlir_source_closure_sha256": MODEL2MLIR_CLOSURE,
               "compiler_revision": _git_revision(ROOT),
               "profile_sha256": profile_sha256(profile),
               "frontend_mlir_sha256": _sha(args.out_dir / "model2mlir.mlir"),
               "frontend_binding_sha256": _sha(args.out_dir / "frontend_binding.json"),
               "bound_mlir_sha256": _sha(mlir),
               "object_sha256": _sha(object_dir / "mx_issue.o"),
               "elf_sha256": _sha(elf), "spike_log_sha256": _sha(log),
               "spike_extension_sha256": _sha(args.out_dir / "run/libgemmini.so"),
               "spike_binary_sha256": _sha(args.riscv_root / "bin/spike"),
               "riscv_gcc_sha256": _sha(args.riscv_root / "bin/riscv64-unknown-elf-gcc"),
               "native_mx_opt_sha256": _sha(args.mx_opt),
               "bf16_values_checked": 16384, "packed_bytes_checked": 8192,
               "e8m0_scales_checked": 512, "mismatches": 0}
    (args.out_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(SUMMARY)


if __name__ == "__main__":
    main()
