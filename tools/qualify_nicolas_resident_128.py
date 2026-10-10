"""Lower Nicolas's typed 128³ resident MM2 and check all outputs on Spike.

This source-bound stage preloads the checked C1 codes and transposed C1 scales.
It proves the resident MM2 command lowering, not compilation of MM1. The
source's complete MM1→MM2 chain is a separate connected-graph gate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.command_ir import Command, Fence, Operand, emit_c
from mx_gemmini_support.physical_program import _cmd, _config_ld, _config_st, _transfer
from mx_gemmini_support.plain_chain_128 import (lower_plain_chain_128,
                                                render_plain_chain_128)
from mx_gemmini_support.resident_lowering import lower_single_resident_contract
from mx_gemmini_support.source_fp6 import _array
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"


def _source_resources(source: Path, header: Path, *, with_mm1: bool = False,
                      source_rows: int = 128
                      ) -> dict[str, bytes]:
    if source_rows not in (64, 128) or (source_rows != 128 and not with_mm1):
        raise ValueError("row-prefix specialization requires connected MM1 and MM2")
    text = source.read_text()
    if (source.name != "matmul_tiled_fp8_128x128_chain.c" or
            header.name != "matmul_fp8_128x128_chain.h" or
            '"include/matmul_fp8_128x128_chain.h"' not in text or
            "SPAD_DEST1 = 2048" not in text or "SPAD_DEST2 = 4096" not in text or
            "#define CHAIN_FLAGS (0x38 | LOOP_WS_REQUANT_TILED)" not in text or
            text.count("gemmini_loop_ws_spad(") != 2 or
            "gemmini_mx_load_scales((uint64_t)&A_scales_row" not in text or
            "gemmini_mx_load_scales((uint64_t)&B_scales_col" not in text or
            "gemmini_mxquant_config_mvout_resident" not in text or
            "gemmini_mx_load_scales((uint64_t)&B2_scales_col" not in text):
        raise ValueError("Nicolas 128³ resident source schedule differs")
    data = header.read_text()
    for marker in ("#define MATMUL_M 128", "#define MATMUL_K 128",
                   "#define MATMUL_N 128", "#define MATMUL_GK 4",
                   "#define MATMUL_GN 4"):
        if marker not in data:
            raise ValueError("Nicolas 128³ resident source shape differs")

    def codes(name: str, dims: str, count: int) -> bytes:
        return bytes(_array(data, name=name, ctype="uint8_t",
                            dimensions=dims, count=count, maximum=255))

    c1 = codes("C1_out", "[MATMUL_M][MATMUL_N]", 16384)
    c1_scales = codes("C1_scales_out", "[MATMUL_M][MATMUL_GN]", 512)
    tiled = bytes(c1[(i * 16 + r) * 128 + j * 16 + c]
                  for i in range(8) for j in range(8)
                  for r in range(16) for c in range(16))
    transposed = bytes(c1_scales[row * 4 + block]
                       for block in range(4) for row in range(128))
    if len(tiled) != 16384 or len(transposed) != 512:
        raise ValueError("Nicolas C1 resident operand layout differs")
    result = {
        "c1_tiled": tiled, "c1_act_scales": transposed,
        "b2_weight": codes("B2_in", "[MATMUL_K][MATMUL_N]", 16384),
        "b2_scales": codes("B2_scales_col", "[MATMUL_GK][MATMUL_N]", 512),
        "c2_codes_ref": codes("C2_out", "[MATMUL_M][MATMUL_N]", 16384),
        "c2_scales_ref": codes("C2_scales_out", "[MATMUL_M][MATMUL_GN]", 512),
    }
    if with_mm1:
        del result["c1_tiled"]
        del result["c1_act_scales"]
        a1 = codes("A_in", "[MATMUL_M][MATMUL_K]", 16384)
        a1_scales = codes("A_scales_row", "[MATMUL_GK][MATMUL_M]", 512)
        if source_rows != 128:
            c1 = c1[:source_rows * 128]
            c1_scales = c1_scales[:source_rows * 4]
            result["c2_codes_ref"] = result["c2_codes_ref"][:source_rows * 128]
            result["c2_scales_ref"] = result["c2_scales_ref"][:source_rows * 4]
            a1 = a1[:source_rows * 128]
            a1_scales = b"".join(
                a1_scales[group * 128:group * 128 + source_rows]
                for group in range(4))
        result.update({
            "a1_activation": a1,
            "a1_scales": a1_scales,
            "b1_weight": codes("B_in", "[MATMUL_K][MATMUL_N]", 16384),
            "b1_scales": codes("B_scales_col", "[MATMUL_GK][MATMUL_N]", 512),
            "c1_codes_ref": c1, "c1_scales_ref": c1_scales,
        })
    return result


def _render_mlir(profile: dict, source: Path, header: Path) -> str:
    contract = _sha(source)
    manifest = _sha(header)
    policy = hashlib.sha256(b"nicolas_fp8_128_resident_mm2_tiled_v1").hexdigest()
    digest = profile_sha256(profile)
    binding = (f'contract_sha256 = "{contract}", policy_sha256 = "{policy}", '
               f'manifest_sha256 = "{manifest}", profile_sha256 = "{digest}"')
    return f'''module attributes {{mx.profile_sha256 = "{digest}",
  mx.contract_sha256 = "{contract}", mx.policy_sha256 = "{policy}",
  prov.quantization_manifest_sha256 = "{manifest}"}} {{
  func.func @nicolas_resident_mm2(
      %c1: tensor<128x128xi8>, %c1s: tensor<128x4xi8>,
      %b2: tensor<128x128xi8>, %b2s: tensor<4x128xi8>)
      -> (tensor<128x128xi8>, tensor<128x4xi8>) {{
    %c2, %c2s = "mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s) {{
      site_id = "nicolas:chain:mm2", activation_row = 2048 : i32,
      weight_row = 15360 : i32, output_row = 4096 : i32,
      m = 128 : i32, n = 128 : i32, k = 128 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      {binding}}}
      : (tensor<128x128xi8>, tensor<128x4xi8>, tensor<128x128xi8>, tensor<4x128xi8>)
      -> (tensor<128x128xi8>, tensor<128x4xi8>)
    func.return %c2, %c2s : tensor<128x128xi8>, tensor<128x4xi8>
  }}
}}
'''


def _commands(mlir: str, profile: dict) -> tuple[Command | Fence, ...]:
    # The first two transfers materialize the checked C1 output as a resident
    # operand; the typed op itself lowers only MM2 and never moves C1 to DRAM.
    commands: list[Command | Fence] = [
        _cmd(7, 0, 0), _config_ld(16),
        _cmd(27, Operand(buffer="c1_act_scales"), 512), Fence(),
    ]
    for row in range(0, 1024, 16):
        commands.append(_transfer(2, "c1_tiled", row * 16, 2048 + row))
    commands.append(Fence())
    commands.extend(lower_single_resident_contract(mlir, profile))
    commands.append(_config_st(16))
    for row in range(0, 1024, 16):
        commands.append(_transfer(3, "c2_tiled", row * 16, 4096 + row))
    commands.append(Fence())
    return tuple(commands)


def _connected_commands(mlir: str, frontend: str, manifest: dict,
                        profile: dict, resources: dict[str, bytes],
                        source: Path, header: Path) -> tuple[Command | Fence, ...]:
    return lower_plain_chain_128(
        mlir, frontend, manifest, profile, resources,
        source_sha256=_sha(source), header_sha256=_sha(header))


def _write_sources(directory: Path, commands: tuple[Command | Fence, ...],
                   resources: dict[str, bytes], *, with_mm1: bool = False,
                   source_rows: int = 128) -> dict:
    directory.mkdir()
    referenced = {operand.buffer for command in commands if isinstance(command, Command)
                  for operand in (command.rs1, command.rs2) if operand.buffer is not None}
    outputs = ({"c1_scales", "c1_tiled_observed", "c2_scales", "c2_tiled"}
               if with_mm1 else {"c2_scales", "c2_tiled"})
    if referenced - set(resources) != outputs:
        raise ValueError("resident MM2 command references an unexpected buffer")
    names = tuple(sorted(referenced))
    (directory / "mx_issue.c").write_text(
        emit_c(commands, transport="rocket_rocc", buffers=names))
    if with_mm1:
        code_count, scale_count = source_rows * 128, source_rows * 4
        declarations = "\n".join(f"extern const uint8_t {name}[];" for name in sorted(resources))
        driver = f'''#include <stdint.h>
#include <stdio.h>
{declarations}
static uint8_t c1_scales[{scale_count}] __attribute__((aligned(64)));
static uint8_t c1_tiled_observed[{code_count}] __attribute__((aligned(64)));
static uint8_t c2_scales[{scale_count}] __attribute__((aligned(64)));
static uint8_t c2_tiled[{code_count}] __attribute__((aligned(64)));
void mx_issue({", ".join(f"const void *{name}" for name in names)});

int main(void) {{
  mx_issue({", ".join(names)});
  int c1_codes = 0, c1_scale_errors = 0, c2_codes = 0, c2_scale_errors = 0;
  for (uint32_t row = 0; row < {source_rows}; ++row)
    for (uint32_t col = 0; col < 128; ++col) {{
      uint32_t tiled = (((row / 16) * 8 + col / 16) * 16 + row % 16) * 16 + col % 16;
      c1_codes += c1_tiled_observed[tiled] != c1_codes_ref[row * 128 + col];
      c2_codes += c2_tiled[tiled] != c2_codes_ref[row * 128 + col];
    }}
  for (uint32_t i = 0; i < {scale_count}; ++i) {{
    c1_scale_errors += c1_scales[i] != c1_scales_ref[i];
    c2_scale_errors += c2_scales[i] != c2_scales_ref[i];
  }}
  printf("lowered connected {source_rows}x128: C1 %d codes %d scales; C2 %d codes %d scales\\n",
         c1_codes, c1_scale_errors, c2_codes, c2_scale_errors);
  return c1_codes || c1_scale_errors || c2_codes || c2_scale_errors;
}}
'''
    else:
        driver = f'''#include <stdint.h>
#include <stdio.h>
extern const uint8_t b2_scales[];
extern const uint8_t b2_weight[];
extern const uint8_t c1_act_scales[];
extern const uint8_t c1_tiled[];
extern const uint8_t c2_codes_ref[];
extern const uint8_t c2_scales_ref[];
static uint8_t c2_scales[512] __attribute__((aligned(64)));
static uint8_t c2_tiled[16384] __attribute__((aligned(64)));
void mx_issue({", ".join(f"const void *{name}" for name in names)});

int main(void) {{
  mx_issue({", ".join(names)});
  int codes = 0, scales = 0;
  for (uint32_t row = 0; row < 128; ++row)
    for (uint32_t col = 0; col < 128; ++col) {{
      uint32_t tiled = (((row / 16) * 8 + col / 16) * 16 + row % 16) * 16 + col % 16;
      codes += c2_tiled[tiled] != c2_codes_ref[row * 128 + col];
    }}
  for (uint32_t i = 0; i < 512; ++i)
    scales += c2_scales[i] != c2_scales_ref[i];
  printf("lowered resident MM2 128x128: %d FP8 code mismatches, %d E8M0 scale mismatches\\n",
         codes, scales);
  return codes || scales;
}}
'''
    (directory / "mx_driver.c").write_text(driver)
    assembly = [".section .rodata", ".balign 64"]
    for name, data in sorted(resources.items()):
        (directory / f"{name}.bin").write_bytes(data)
        assembly.extend((f".globl {name}", f"{name}:",
                         f'.incbin "{name}.bin"', ".balign 64"))
    assembly.append('.section .note.GNU-stack,"",@progbits')
    (directory / "mx_data.S").write_text("\n".join(assembly) + "\n")
    physical = {
        "schema": (("mx_gemmini.connected_plain_chain_64x128_physical.v1"
                    if source_rows != 128 else
                    "mx_gemmini.connected_plain_chain_128_physical.v1") if with_mm1 else
                   "mx_gemmini.resident_mm2_128_physical.v1"),
        "ordered_functs": [item.funct for item in commands if isinstance(item, Command)],
        "command_count": sum(isinstance(item, Command) for item in commands),
        "fence_count": sum(isinstance(item, Fence) for item in commands),
    }
    (directory / "physical_program.json").write_text(
        json.dumps(physical, indent=2, sort_keys=True) + "\n")
    return {path.name: _sha(path) for path in sorted(directory.iterdir()) if path.is_file()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--mx-opt", type=Path)
    parser.add_argument("--connected-frontend-dir", type=Path,
                        help="capture_nicolas_chain --matrix-dim 128 output or checked-in archive")
    parser.add_argument("--source-rows", type=int, choices=(64, 128), default=128,
                        help="row prefix of Nicolas's 128³ source; 64 requires a connected capture")
    parser.add_argument("--baseline-manifest", type=Path,
                        help="require identical generated program and Spike output")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if (profile["name"] != "MxGemminiRocketConfig" or
            profile["geometry"]["mesh_columns"] != 16 or
            not profile["resources"]["requantizer"]):
        raise ValueError("selected Nicolas plain MX profile differs")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    extension = args.rtl_root / "software/libgemmini"
    _require_gitlink(args.rtl_root, "software/gemmini-rocc-tests")
    _require_gitlink(args.rtl_root, "software/libgemmini")
    source = software / "bareMetalC/matmul_tiled_fp8_128x128_chain.c"
    header = software / "include/matmul_fp8_128x128_chain.h"
    connected = args.connected_frontend_dir is not None
    if not connected and args.source_rows != 128:
        parser.error("--source-rows 64 requires --connected-frontend-dir")
    resources = _source_resources(source, header, with_mm1=connected,
                                  source_rows=args.source_rows)
    if connected:
        frontend_dir = args.connected_frontend_dir

        def frontend_bytes(name: str) -> bytes:
            path = frontend_dir / name
            if path.is_file():
                return path.read_bytes()
            return gzip.decompress((frontend_dir / f"{name}.gz").read_bytes())

        frontend = frontend_bytes("nicolas_chain.profile_bound.mlir").decode()
        manifest_bytes = frontend_bytes("quantization_manifest.json")
        frontend_manifest = json.loads(manifest_bytes)
        capture_receipt = json.loads(frontend_bytes("receipt.json"))
        if (capture_receipt.get("schema") !=
                "mx_gemmini.nicolas_chain_128_model2mlir_capture.v1" or
                capture_receipt.get("matrix_dim") != 128 or
                capture_receipt.get("output_rows", 128) != args.source_rows or
                capture_receipt.get("profile_sha256") != profile_sha256(profile) or
                capture_receipt.get("source_sha256") != _sha(source) or
                capture_receipt.get("header_sha256") != _sha(header) or
                capture_receipt.get("bound_mlir_sha256") !=
                hashlib.sha256(frontend.encode()).hexdigest() or
                capture_receipt.get("manifest_sha256") != hashlib.sha256(
                    manifest_bytes).hexdigest() or
                capture_receipt.get("source_mlir_sha256") != hashlib.sha256(
                    frontend_bytes("nicolas_chain.model2mlir.mlir")).hexdigest() or
                capture_receipt.get("handoff_mlir_sha256") != hashlib.sha256(
                    frontend_bytes("nicolas_chain.handoff.mlir")).hexdigest()):
            raise ValueError("connected frontend capture or source provenance differs")
        mlir = render_plain_chain_128(
            frontend, frontend_manifest, profile, resources,
            source_sha256=_sha(source), header_sha256=_sha(header))
        commands = _connected_commands(
            mlir, frontend, frontend_manifest, profile, resources, source, header)
    else:
        mlir = _render_mlir(profile, source, header)
        commands = _commands(mlir, profile)
    args.out_dir.mkdir(parents=True)
    mlir_path = args.out_dir / ("connected_chain.mlir" if connected else "resident_mm2.mlir")
    mlir_path.write_text(mlir)
    if args.mx_opt:
        _run([str(args.mx_opt.resolve()), str(mlir_path), "-o", "/dev/null"],
             cwd=args.out_dir, log=args.out_dir / "native_verify.log")
    build = args.out_dir / "build"
    files = _write_sources(build, commands, resources, with_mm1=connected,
                           source_rows=args.source_rows)
    riscv_cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if not riscv_cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        parser.error("selected RISC-V toolchain or host C++ compiler is absent")
    bench = software / "riscv-tests/benchmarks/common"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             f"-ffile-prefix-map={build.resolve()}=.",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [build / name for name in ("mx_issue.c", "mx_driver.c", "mx_data.S")]
    sources += sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))
    objects = []
    for index, path in enumerate(sources):
        obj = build / f"mx_{index}.o"
        _run([str(riscv_cc), *flags, "-c", str(path), "-o", str(obj)],
             cwd=build, log=build / f"compile_{index}.log")
        objects.append(obj)
    elf = build / "mx_program.elf"
    _run([str(riscv_cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(path) for path in objects), "-lm", "-lgcc",
          "-o", str(elf)], cwd=build, log=build / "link.log")
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = build / "libgemmini.so"
    _run(["g++", "-L", str(args.riscv_root / "lib"),
          f"-Wl,-rpath,{args.riscv_root / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(args.riscv_root / "include"), "-fPIC", "-O3",
          *(str(path) for path in extension_sources)],
         cwd=build, log=build / "extension_build.log")
    result = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini", str(elf)],
                            cwd=build, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log = build / "spike.log"
    log.write_text(result.stdout)
    marker = (f"lowered connected {args.source_rows}x128: C1 0 codes 0 scales; "
              "C2 0 codes 0 scales"
              if connected else
              "lowered resident MM2 128x128: 0 FP8 code mismatches, 0 E8M0 scale mismatches")
    passed = result.returncode == 0 and marker in result.stdout
    receipt = {
        "schema": (("mx_gemmini.nicolas_connected_plain_chain_64x128.v1"
                    if args.source_rows != 128 else
                    "mx_gemmini.nicolas_connected_plain_chain_128.v1") if connected else
                   "mx_gemmini.nicolas_resident_mm2_128.v1"),
        "status": (("source_prefix_connected_chain_matched_on_pinned_spike" if passed else
                    "source_prefix_connected_chain_failed_on_pinned_spike")
                   if args.source_rows != 128 else
                   ("source_connected_chain_matched_on_pinned_spike" if passed else
                    "source_connected_chain_failed_on_pinned_spike")) if connected else (
                   "source_resident_mm2_matched_on_pinned_spike" if passed else
                   "source_resident_mm2_failed_on_pinned_spike"),
        "scope": ("typed MM1 quantized C1 and scales remain resident for typed MM2" if connected
                  else "source C1 codes/scales preloaded; typed resident MM2 lowered; excludes MM1"),
        "source_sha256": _sha(source), "header_sha256": _sha(header),
        "profile_sha256": profile_sha256(profile), "bound_mlir_sha256": _sha(mlir_path),
        "source_revision": _git_revision(software),
        "rtl_revision": _git_revision(args.rtl_root),
        "gemmini_extension_revision": _git_revision(extension),
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "riscv_gcc_sha256": _sha(riscv_cc), "spike_sha256": _sha(spike),
        "elf_sha256": _sha(elf), "extension_sha256": _sha(so),
        "object_sha256": {path.name: _sha(path) for path in objects},
        "spike_log_sha256": _sha(log), "spike_exit_code": result.returncode,
        "compared_fp8_codes": args.source_rows * 128,
        "compared_e8m0_scales": args.source_rows * 4,
        "files_sha256": files,
    }
    if connected:
        receipt["frontend_mlir_sha256"] = hashlib.sha256(frontend.encode()).hexdigest()
        receipt["compared_c1_fp8_codes"] = args.source_rows * 128
        receipt["compared_c1_e8m0_scales"] = args.source_rows * 4
        if args.source_rows != 128:
            receipt["source_rows"] = args.source_rows
            receipt["scope"] = (
                "first 64 independent output rows of Nicolas's 128³ packed source; "
                "typed MM1 C1 and scales remain resident for typed MM2")
    (build / "artifact_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if args.baseline_manifest:
        baseline = json.loads(args.baseline_manifest.read_text())
        stable = ("source_sha256", "header_sha256", "profile_sha256",
                  "bound_mlir_sha256", "source_revision", "rtl_revision",
                  "gemmini_extension_revision", "elf_sha256", "extension_sha256",
                  "spike_log_sha256", "spike_exit_code", "compared_fp8_codes",
                  "compared_e8m0_scales", "files_sha256", "object_sha256")
        if connected:
            stable += ("frontend_mlir_sha256", "compared_c1_fp8_codes",
                       "compared_c1_e8m0_scales")
        if any(receipt[key] != baseline.get(key) for key in stable):
            raise ValueError("resident MM2 program or Spike result differs from baseline")
    print(f"{receipt['status']}: {elf}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
