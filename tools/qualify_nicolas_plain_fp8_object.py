"""Compile Nicolas's pinned 128-cubed FP8 source data through typed MX MLIR.

The structural matmul comes from a fresh PyTorch -> model2MLIR capture. The
packed operands and BF16 reference come from Nicolas's checked-in C header.
The driver only calls the data-free compiled issuer object and checks output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from mx_gemmini_support.bind_payload import bind_payload, select_bf16_output_layout
from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.source_gemm import SourceGemm
from mx_gemmini_support.source_payload import NICOLAS_SOURCE_HEADER_ORIGIN, write_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _require_gitlink, _run


ROOT = Path(__file__).resolve().parents[1]
RTL_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
SOURCE_SHA256 = "1a0016d2ca9ebcca5840b31486ed6bda2df4cf3756aec0922781db398c98d5ce"
HEADER_SHA256 = "16241671c4df2d4f738e77225063caac4895cdcba4d4495941e599d0db185bcd"
MODEL2MLIR_REVISION = "e9ded36eb85abf2d9097ac4dc11457c825853388"
MXQ_REVISION = "b4af5430bac147f4a16126931cc0177367cc3982"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def source_kernel(rtl_root: Path) -> SourceGemm:
    """Admit exactly the Nicolas test whose header and golden were audited."""
    if _revision(rtl_root) != RTL_REVISION:
        raise ValueError("Nicolas FP8 qualifier requires the pinned MX RTL revision")
    for submodule in ("software/gemmini-rocc-tests", "software/libgemmini"):
        _require_gitlink(rtl_root, submodule)
    software = rtl_root / "software/gemmini-rocc-tests"
    driver = software / "bareMetalC/matmul_tiled_fp8_128x128.c"
    header = software / "include/matmul_fp8_128x128.h"
    if _sha(driver) != SOURCE_SHA256 or _sha(header) != HEADER_SHA256:
        raise ValueError("Nicolas FP8 driver/header differs from pinned source")
    source = driver.read_text()
    required = ("#include \"include/matmul_fp8_128x128.h\"", "gemmini_mx_load_scales",
                "gemmini_loop_ws_spad", "gemmini_extended_mvout", "C_out_bf16")
    if any(needle not in source for needle in required):
        raise ValueError("Nicolas FP8 source no longer has the audited compute/check path")
    header_text = header.read_text()
    for axis in ("M", "N", "K"):
        if re.search(rf"^#define MATMUL_{axis} 128$", header_text, re.M) is None:
            raise ValueError("Nicolas FP8 source header shape changed")
    return SourceGemm(driver, header, (128, 128, 128), (128, 128, 128),
                      "FP8", False, False, True)


def capture_handoff(model2mlir: Path, mxq_root: Path) -> tuple[str, str, dict]:
    sys.path[:0] = [str(model2mlir), str(mxq_root)]
    import m2m
    import mxq
    import torch
    from m2m.capture.external_quantization import ExternalQuantizationConfig
    from m2m.coverage import opaque_report
    from mx_gemmini_support.handoff import render_handoff, validate_handoff

    if (Path(m2m.__file__).resolve().parents[1] != model2mlir or
            Path(mxq.__file__).resolve().parents[1] != mxq_root or
            _revision(model2mlir) != MODEL2MLIR_REVISION or
            _revision(mxq_root) != MXQ_REVISION):
        raise ValueError("model2MLIR/MXQuant checkout differs from selected frontend")

    class Matmul(torch.nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        example = (torch.randn((128, 128), dtype=torch.float32),
                   torch.randn((128, 128), dtype=torch.float32))
    contract = ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
    policy = ROOT / "examples/default-policy.yaml"
    result = m2m.convert(
        Matmul().eval(), example,
        quantization=ExternalQuantizationConfig("mx_gemmini", contract, policy),
        backend="fx_importer", capture_trace=True)
    if not result.ok or opaque_report(result.mlir_text):
        raise ValueError(f"model2MLIR matmul capture failed: {result.diagnostics}")
    sites = result.quantization_manifest["sites"]
    if [(site["site_id"], site["status"], site["format"], site["shape"])
            for site in sites] != [
                ("functional:matmul", "quantized", "mxfp8", [128, 128, 128])]:
        raise ValueError(f"model2MLIR did not capture the selected FP8 matmul: {sites}")
    contract_bytes, policy_bytes = contract.read_bytes(), policy.read_bytes()
    validate_handoff(result, contract_bytes, policy_bytes)
    return (result.mlir_text, render_handoff(result, contract_bytes, policy_bytes),
            result.quantization_manifest)


def _driver() -> str:
    return """#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/matmul_fp8_128x128.h"
#include "mx_issue.h"
static uint16_t C_hw[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));
static uint8_t scratch_output_scales[2048] __attribute__((aligned(64)));
int main(void) {
  mx_issue(A_in, A_scales_row, C_hw, scratch_output_scales, B_in, B_scales_col);
  gemmini_fence();
  int errors = 0;
  for (int i = 0; i < MATMUL_M; ++i)
    for (int j = 0; j < MATMUL_N; ++j)
      if (C_hw[i][j] != C_out_bf16[i][j]) {
        if (errors < 8) printf("mismatch %d,%d got %x want %x\\n",
                               i,j,C_hw[i][j],C_out_bf16[i][j]);
        ++errors;
      }
  printf("compiled Nicolas FP8 128 cubed: %d mismatches / %d BF16 values\\n",
         errors, MATMUL_M * MATMUL_N);
  return errors != 0;
}
"""


def run_spike(rtl_root: Path, riscv_root: Path, object_dir: Path,
              out_dir: Path) -> tuple[int, str, Path]:
    software = rtl_root / "software/gemmini-rocc-tests"
    bench = software / "riscv-tests/benchmarks/common"
    build = out_dir / "run"
    build.mkdir()
    abi = json.loads((object_dir / "object_manifest.json").read_text())["buffer_abi"]
    if ([slot["name"] for slot in abi] != ["activation", "activation_scales",
        "output_bf16", "scratch_output_scales", "weight", "weight_scales"] or
            [slot["position"] for slot in abi] != list(range(6)) or
            abi[3]["minimum_bytes"] > 2048):
        raise ValueError("compiled MX object buffer ABI differs from checked driver")
    (build / "mx_driver.c").write_text(_driver())
    cc = riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = riscv_root / "bin/spike"
    if not all(path.is_file() for path in (cc, spike, bench / "test.ld")):
        raise ValueError("RISC-V GCC, Spike, or Nicolas benchmark linker is missing")
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench), "-I", str(object_dir)]
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
    extension = rtl_root / "software/libgemmini"
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc",
                         *sorted((extension / "perf").rglob("*.cc"))]
    so = build / "libgemmini.so"
    _run(["g++", "-L", str(riscv_root / "lib"),
          f"-Wl,-rpath,{riscv_root / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(riscv_root / "include"), "-fPIC", "-O3",
          *(str(path) for path in extension_sources)],
         cwd=build, log=build / "extension.log")
    run = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini",
                          str(elf)], cwd=build, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=False)
    (build / "spike.log").write_text(run.stdout)
    return run.returncode, run.stdout, elf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "rtl-root", "profile",
                 "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    rtl_root = args.rtl_root.resolve()
    kernel = source_kernel(rtl_root)
    profile = load_profile(args.profile, rtl_root=rtl_root)
    if profile["name"] != "MxGemminiRocketConfig":
        raise ValueError("Nicolas plain FP8 source needs the MX-only Rocket profile")
    source_mlir, handoff, quant_manifest = capture_handoff(
        args.model2mlir_root.resolve(), args.mxq_root.resolve())
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "model2mlir.mlir").write_text(source_mlir)
    (args.out_dir / "handoff.mlir").write_text(handoff)
    (args.out_dir / "quantization_manifest.json").write_text(
        json.dumps(quant_manifest, indent=2, sort_keys=True) + "\n")
    bundle = args.out_dir / "bundle"
    manifest = write_bundle(bundle, kernel, site_id="functional:matmul",
                            profile_sha256=profile_sha256(profile),
                            source_origin=NICOLAS_SOURCE_HEADER_ORIGIN)
    bound = bind_handoff(handoff, profile)
    bound = bind_payload(bound, profile, manifest)
    bound = select_bf16_output_layout(bound, profile, manifest)
    mlir = args.out_dir / "payload_bound.mlir"
    mlir.write_text(bound)
    object_dir = args.out_dir / "object"
    _run([sys.executable, "-m", "tools.compile_object", "--mlir", str(mlir),
          "--bundle", str(bundle), "--profile", str(args.profile.resolve()),
          "--rtl-root", str(rtl_root), "--riscv-root", str(args.riscv_root.resolve()),
          "--mx-opt", str(args.mx_opt.resolve()), "--out-dir", str(object_dir)],
         cwd=ROOT, log=args.out_dir / "object_compile.log")
    returncode, output, elf = run_spike(rtl_root, args.riscv_root.resolve(),
                                        object_dir, args.out_dir)
    passed = (returncode == 0 and
              "compiled Nicolas FP8 128 cubed: 0 mismatches / 16384 BF16 values" in output)
    dispatch = json.loads((object_dir / "compile_manifest.json").read_text())
    receipt = {
        "schema": "mx_gemmini.nicolas_plain_fp8_typed_object_spike.v1",
        "status": "source_golden_matched_on_pinned_spike" if passed else
                  "source_golden_failed_on_pinned_spike",
        "scope": "Nicolas matmul_tiled_fp8_128x128: packed FP8 source arrays, 128x128x128 BF16 output; PyTorch model2MLIR capture, typed source binding, public object compiler, full Spike comparison",
        "source_driver_sha256": _sha(kernel.driver),
        "source_header_sha256": _sha(kernel.data_header),
        "rtl_revision": _revision(rtl_root),
        "software_revision": _revision(rtl_root / "software/gemmini-rocc-tests"),
        "model2mlir_revision": _revision(args.model2mlir_root),
        "mxq_revision": _revision(args.mxq_root),
        "compiler_revision": dispatch["compiler_revision"],
        "compiler_source_closure_sha256": dispatch["compiler_source_closure_sha256"],
        "profile_name": profile["name"], "profile_sha256": profile_sha256(profile),
        "source_mlir_sha256": _sha(args.out_dir / "model2mlir.mlir"),
        "handoff_mlir_sha256": _sha(args.out_dir / "handoff.mlir"),
        "payload_manifest_sha256": _sha(bundle / "manifest.json"),
        "bound_mlir_sha256": _sha(mlir),
        "object_sha256": _sha(object_dir / "mx_issue.o"),
        "object_dispatch_manifest_sha256": _sha(object_dir / "compile_manifest.json"),
        "object_manifest_sha256": _sha(object_dir / "object_manifest.json"),
        "elf_sha256": _sha(elf), "spike_log_sha256": _sha(args.out_dir / "run/spike.log"),
        "spike_sha256": _sha(args.riscv_root.resolve() / "bin/spike"),
        "spike_extension_sha256": _sha(args.out_dir / "run/libgemmini.so"),
        "driver_sha256": _sha(args.out_dir / "run/mx_driver.c"),
        "outputs_checked": 16384, "mismatches": 0 if passed else None,
    }
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                           sort_keys=True) + "\n")
    print(output)
    if not passed:
        raise SystemExit("compiled Nicolas FP8 source did not match its BF16 golden")


if __name__ == "__main__":
    main()
