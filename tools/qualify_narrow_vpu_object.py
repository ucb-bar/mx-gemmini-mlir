"""Link the typed narrow MX/VPU object and check every source-derived output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.command_ir import emit_c
from mx_gemmini_support.connected_fp8_model import (
    MODEL_SHA256, load_nicolas_fp8_model, model_c2)
from mx_gemmini_support.narrow_vpu_chain import render_narrow_vpu_chain
from mx_gemmini_support.narrow_vpu_source import derive_narrow_vpu_resources
from mx_gemmini_support.quant_reference import (
    bf16_add_scalar, exact_bf16_x2, quantize_bf16_fp8_output)
from mx_gemmini_support.resident_pair_graph import INPUTS
from mx_gemmini_support.resident_vpu_graph import OUTPUTS, lower_connected_fp8_vpu_pair
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure
from tools.emit_resident_vpu_object import _readout_commands


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
MARKER = "lowered narrow MX/VPU: C1 BF16 0, C1 0 codes 0 scales, C2 0 codes 0 scales"


def _driver(names: tuple[str, ...], second_width: int = 32) -> str:
    if second_width < 32 or second_width % 32:
        raise ValueError("connected VPU driver needs complete E8M0 output blocks")
    declarations = "\n".join(
        f"extern const uint8_t {name}[];" for name in
        (*INPUTS, "c1_bf16", "c1_codes_ref", "c1_scales_ref",
         "c2_codes_ref", "c2_scales_ref"))
    return f'''#include <stdint.h>
#include <stdio.h>
{declarations}
static uint8_t c1_scales[128] __attribute__((aligned(64)));
static uint8_t c1_bf16_observed[8192] __attribute__((aligned(64)));
static uint8_t c1_tiled[4096] __attribute__((aligned(64)));
static uint8_t c2_scales[{2 * second_width}] __attribute__((aligned(64)));
static uint8_t c2_tiled[{64 * second_width}] __attribute__((aligned(64)));
void mx_issue({", ".join(f"const void *{name}" for name in names)});

int main(void) {{
  mx_issue({", ".join(names)});
  int bf16_errors = 0, c1_codes = 0, c1_scale_errors = 0;
  int c2_codes = 0, c2_scale_errors = 0;
  for (uint32_t i = 0; i < 8192; ++i)
    bf16_errors += c1_bf16_observed[i] != c1_bf16[i];
  for (uint32_t row = 0; row < 64; ++row) {{
    for (uint32_t col = 0; col < 64; ++col) {{
      uint32_t tiled = (((row / 16) * 4 + col / 16) * 16 + row % 16) * 16 + col % 16;
      c1_codes += c1_tiled[tiled] != c1_codes_ref[row * 64 + col];
    }}
    for (uint32_t col = 0; col < {second_width}; ++col) {{
      uint32_t tiled = (((row / 16) * {second_width // 16} + col / 16) * 16 + row % 16) * 16 + col % 16;
      c2_codes += c2_tiled[tiled] != c2_codes_ref[row * {second_width} + col];
    }}
  }}
  for (uint32_t i = 0; i < 128; ++i)
    c1_scale_errors += c1_scales[i] != c1_scales_ref[i];
  for (uint32_t i = 0; i < {2 * second_width}; ++i)
    c2_scale_errors += c2_scales[i] != c2_scales_ref[i];
  printf("lowered narrow MX/VPU: C1 BF16 %d, C1 %d codes %d scales, "
         "C2 %d codes %d scales\\n", bf16_errors, c1_codes,
         c1_scale_errors, c2_codes, c2_scale_errors);
  return bf16_errors || c1_codes || c1_scale_errors || c2_codes || c2_scale_errors;
}}
'''


def _append_scalar_adds(bound: str, scalar_bf16: int) -> str:
    """Derive an SSA-linked scalar ADDS after Nicolas's source MULS ×2."""
    if not 0 <= scalar_bf16 <= 0xffff or scalar_bf16 & 0x7f80 == 0x7f80:
        raise ValueError("derived VPU scalar must be finite BF16 bits")
    marker = '    %vpu = "mx_gemmini.vpu_execute"'
    if bound.count(marker) != 1 or bound.count('"mx_gemmini.spad_requant"(%vpu)') != 1:
        raise ValueError("narrow source graph lacks its single checked VPU edge")
    start = bound.index(marker)
    end = bound.index('    %c1, %c1s =', start)
    original = bound[start:end]
    if (original.count('"mx_gemmini.vpu_execute"(%bf16)') != 1 or
            original.count('kind = "muls"') != 1 or
            original.count('immediate_bf16 = 16384') != 1):
        raise ValueError("narrow source VPU differs from the qualified BF16 ×2")
    second = (original.replace('%vpu =', '%vpu2 =', 1)
              .replace('"mx_gemmini.vpu_execute"(%bf16)',
                       '"mx_gemmini.vpu_execute"(%vpu)', 1)
              .replace('kind = "muls"', 'kind = "adds"', 1)
              .replace('immediate_bf16 = 16384',
                       f'immediate_bf16 = {scalar_bf16}', 1))
    return (bound[:end] + second + bound[end:]).replace(
        '"mx_gemmini.spad_requant"(%vpu)',
        '"mx_gemmini.spad_requant"(%vpu2)', 1)


def _append_zero_adds(bound: str) -> str:
    return _append_scalar_adds(bound, 0)


def _model_c2(resources: dict[str, bytes], model) -> tuple[bytes, bytes]:
    _, codes, scales = model_c2(resources, model, width=32)
    return codes, scales


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-dir", "bound-dir", "object-dir", "rtl-root",
                 "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--mx-opt", type=Path)
    parser.add_argument("--derived-zero-adds-mlir", type=Path,
                        help="check an SSA-linked ADDS +0 after the source MULS ×2")
    parser.add_argument("--derived-adds-mlir", type=Path,
                        help="check a derived SSA-linked nonzero scalar ADDS")
    parser.add_argument("--derived-adds-bf16", type=lambda s: int(s, 0),
                        help="BF16 bits of the derived nonzero ADDS immediate")
    args = parser.parse_args()
    for name in ("capture_dir", "bound_dir", "object_dir", "rtl_root",
                 "riscv_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    software = args.rtl_root / "software/gemmini-rocc-tests"
    extension = args.rtl_root / "software/libgemmini"
    _require_gitlink(args.rtl_root, "software/gemmini-rocc-tests")
    _require_gitlink(args.rtl_root, "software/libgemmini")
    source = software / "bareMetalC/chain_vpu_spad_requant.c"
    first_source = software / "bareMetalC/matmul_tiled_fp8_64x64_chain.c"
    header = software / "include/matmul_fp8_64x64_chain.h"
    capture_receipt = json.loads((args.capture_dir / "receipt.json").read_text())
    frontend = args.capture_dir / "nicolas_chain.profile_bound.mlir"
    manifest_path = args.capture_dir / "quantization_manifest.json"
    if (capture_receipt.get("schema") !=
            "mx_gemmini.nicolas_narrow_vpu_chain_model2mlir_capture.v1" or
            capture_receipt.get("source_sha256") != _sha(source) or
            capture_receipt.get("header_sha256") != _sha(header) or
            capture_receipt.get("profile_sha256") != profile_sha256(profile) or
            capture_receipt.get("bound_mlir_sha256") != _sha(frontend) or
            capture_receipt.get("manifest_sha256") != _sha(manifest_path)):
        raise ValueError("narrow MX/VPU capture provenance differs")
    resources, facts = derive_narrow_vpu_resources(
        source, header, first_source, profile)
    bound = render_narrow_vpu_chain(
        frontend.read_text(), json.loads(manifest_path.read_text()),
        profile, resources, facts)
    binding_path = args.bound_dir / "binding_manifest.json"
    binding = json.loads(binding_path.read_text())
    bound_path = args.bound_dir / "connected.mlir"
    if (binding.get("schema") != "mx_gemmini.nicolas_narrow_vpu_chain_binding.v1" or
            binding.get("capture_receipt_sha256") != _sha(args.capture_dir / "receipt.json") or
            binding.get("bound_mlir_sha256") != hashlib.sha256(bound.encode()).hexdigest() or
            binding.get("profile_sha256") != profile_sha256(profile) or
            binding.get("source_sha256") != _sha(source) or
            binding.get("first_source_sha256") != _sha(first_source) or
            binding.get("header_sha256") != _sha(header) or
            bound_path.read_text() != bound or
            binding.get("resources_sha256") != {
                name: hashlib.sha256(data).hexdigest() for name, data in sorted(resources.items())} or
            any((args.bound_dir / f"{name}.bin").read_bytes() != data
                for name, data in resources.items())):
        raise ValueError("narrow MX/VPU bound resources differ from source")
    if args.derived_zero_adds_mlir and args.derived_adds_mlir:
        parser.error("select one derived VPU candidate")
    if (args.derived_adds_mlir is None) != (args.derived_adds_bf16 is None):
        parser.error("nonzero ADDS graph and BF16 immediate must be supplied together")
    bound_for_object = bound_path
    reference_resources = resources.copy()
    model_sha256 = None
    if args.derived_zero_adds_mlir is not None:
        bound_for_object = args.derived_zero_adds_mlir.resolve()
        first_vpu_output = exact_bf16_x2(resources["c1_bf16"])
        if (bound_for_object.read_text() != _append_zero_adds(bound) or
                bf16_add_scalar(first_vpu_output, 0) != first_vpu_output):
            raise ValueError("derived ADDS +0 graph or BF16 oracle differs from source")
    if args.derived_adds_mlir is not None:
        scalar = args.derived_adds_bf16
        if scalar == 0:
            parser.error("use --derived-zero-adds-mlir for the identity case")
        bound_for_object = args.derived_adds_mlir.resolve()
        if bound_for_object.read_text() != _append_scalar_adds(bound, scalar):
            raise ValueError("derived nonzero ADDS graph differs from source")
        model_path = software / "fp8_matmul_model.py"
        model_sha256 = _sha(model_path)
        if model_sha256 != MODEL_SHA256:
            raise ValueError("Nicolas's pinned FP8 model changed")
        model = load_nicolas_fp8_model(software)
        baseline_c2 = _model_c2(resources, model)
        if baseline_c2 != (resources["c2_codes_ref"], resources["c2_scales_ref"]):
            raise ValueError("Nicolas's model no longer matches the source C2 reference")
        derived_bf16 = bf16_add_scalar(exact_bf16_x2(resources["c1_bf16"]), scalar)
        c1_codes, c1_scales = quantize_bf16_fp8_output(derived_bf16, 64, 64)
        reference_resources["c1_codes_ref"] = c1_codes
        reference_resources["c1_scales_ref"] = c1_scales
        c2_codes, c2_scales = _model_c2(reference_resources, model)
        reference_resources["c2_codes_ref"] = c2_codes
        reference_resources["c2_scales_ref"] = c2_scales
        if any(reference_resources[name] == resources[name] for name in
               ("c1_codes_ref", "c1_scales_ref", "c2_codes_ref",
                "c2_scales_ref")):
            raise ValueError("derived nonzero case did not change every output class")
    pair = lower_connected_fp8_vpu_pair(
        bound_for_object.read_text(), profile, resources,
        buffers={name: name for name in INPUTS},
        outputs={name: name for name in OUTPUTS})
    object_manifest_path = args.object_dir / "object_manifest.json"
    object_manifest = json.loads(object_manifest_path.read_text())
    obj = args.object_dir / "mx_issue.o"
    abi = object_manifest.get("buffer_abi", [])
    names = tuple(entry.get("name") for entry in abi)
    expected_bytes = {name: len(resources[name]) for name in INPUTS} | {
        "c1_scales": 128, "c1_bf16_observed": 8192, "c1_tiled": 4096,
        "c2_scales": 64, "c2_tiled": 2048}
    if (object_manifest.get("schema") !=
            "mx_gemmini.resident_vpu_linkable_object.v1" or
            object_manifest.get("shape_mnk") != [64, 32, 64] or
            object_manifest.get("profile_sha256") != profile_sha256(profile) or
            object_manifest.get("bound_mlir_sha256") != _sha(bound_for_object) or
            object_manifest.get("object_sha256") != _sha(obj) or
            object_manifest.get("issuer_c_sha256") != hashlib.sha256(
                emit_c(_readout_commands(pair, {name: name for name in OUTPUTS}),
                       transport="rocket_rocc", buffers=names).encode()).hexdigest() or
            any(object_manifest.get(key) != 0 for key in
                ("allocated_data_section_bytes", "embedded_operand_bytes",
                 "embedded_golden_bytes")) or
            {entry.get("slot"): entry.get("name") for entry in abi} !=
            {name: name for name in (*INPUTS, *OUTPUTS)} or
            {entry.get("slot"): entry.get("minimum_bytes") for entry in abi} !=
            expected_bytes or
            object_manifest.get("input_sha256") != {
                name: hashlib.sha256(resources[name]).hexdigest() for name in INPUTS}):
        raise ValueError("narrow MX/VPU object differs from checked typed source")
    args.out_dir.mkdir(parents=True)
    if args.mx_opt is not None:
        _run([str(args.mx_opt.resolve()), str(bound_for_object), "-o", "/dev/null"],
             cwd=args.out_dir, log=args.out_dir / "native_verify.log")
    build = args.out_dir / "build"
    build.mkdir()
    assembly = [".section .rodata", ".balign 64"]
    for name, data in sorted(reference_resources.items()):
        (build / f"{name}.bin").write_bytes(data)
        assembly.extend((f".globl {name}", f"{name}:",
                         f'.incbin "{name}.bin"', ".balign 64"))
    assembly.append('.section .note.GNU-stack,"",@progbits')
    (build / "mx_data.S").write_text("\n".join(assembly) + "\n")
    (build / "mx_driver.c").write_text(_driver(names))
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
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
    sources = [build / "mx_driver.c", build / "mx_data.S"]
    sources += sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))
    objects = []
    for index, path in enumerate(sources):
        target = build / f"mx_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(target)],
             cwd=build, log=build / f"compile_{index}.log")
        objects.append(target)
    elf = build / "mx_program.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), str(obj), *(str(path) for path in objects),
          "-lm", "-lgcc", "-o", str(elf)], cwd=build, log=build / "link.log")
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = build / "libgemmini.so"
    _run(["g++", "-L", str(args.riscv_root / "lib"),
          f"-Wl,-rpath,{args.riscv_root / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(args.riscv_root / "include"),
          "-fPIC", "-O3", *(str(path) for path in extension_sources)],
         cwd=build, log=build / "extension_build.log")
    result = subprocess.run(
        [str(spike), f"--extlib={so}", "--extension=gemmini", str(elf)],
        cwd=build, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    log = build / "spike.log"
    log.write_text(result.stdout)
    passed = result.returncode == 0 and MARKER in result.stdout
    receipt = {
        "schema": "mx_gemmini.nicolas_narrow_vpu_pair_spike.v1",
        "status": (("derived_nonzero_adds_vpu_chain_matched_on_pinned_spike" if
                    args.derived_adds_mlir else
                    "derived_zero_adds_vpu_chain_matched_on_pinned_spike" if
                    args.derived_zero_adds_mlir else
                    "source_mx_vpu_and_narrow_mm2_matched_on_pinned_spike") if passed else
                   "narrow_mx_vpu_pair_failed_on_pinned_spike"),
        "first_shape_mnk": [64, 64, 64], "second_shape_mnk": [64, 32, 64],
        "reference_kind": ("source_chain_plus_bf16_adds_nonzero_model_derived" if
                           args.derived_adds_mlir else
                           "source_chain_plus_bf16_adds_zero_identity" if
                           args.derived_zero_adds_mlir else
                           "unchanged_nicolas_vpu_chain_left_output_block"),
        "compared_c1_bf16_values": 4096,
        "compared_c1_fp8_codes": 4096, "compared_c1_e8m0_scales": 128,
        "compared_c2_fp8_codes": 2048, "compared_c2_e8m0_scales": 64,
        "capture_receipt_sha256": _sha(args.capture_dir / "receipt.json"),
        "binding_manifest_sha256": _sha(binding_path),
        "source_sha256": _sha(source), "first_source_sha256": _sha(first_source),
        "header_sha256": _sha(header),
        "bound_mlir_sha256": _sha(bound_for_object),
        "object_manifest_sha256": _sha(object_manifest_path),
        "object_sha256": _sha(obj), "elf_sha256": _sha(elf),
        "spike_log_sha256": _sha(log), "spike_exit_code": result.returncode,
        "profile_sha256": profile_sha256(profile),
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "rtl_revision": _git_revision(args.rtl_root),
        "riscv_gcc_sha256": _sha(cc), "spike_sha256": _sha(spike),
        "extension_sha256": _sha(so),
        "resources_sha256": {name: hashlib.sha256(data).hexdigest()
                             for name, data in sorted(resources.items())},
    }
    if args.derived_adds_mlir is not None:
        receipt["derived_adds_bf16"] = args.derived_adds_bf16
        receipt["model_sha256"] = model_sha256
        receipt["source_to_derived_byte_differences"] = {
            name: sum(a != b for a, b in zip(resources[name], reference_resources[name]))
            for name in ("c1_codes_ref", "c1_scales_ref", "c2_codes_ref",
                         "c2_scales_ref")}
        receipt["reference_resources_sha256"] = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(reference_resources.items())}
    (args.out_dir / "qualification_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{receipt['status']}: {elf}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
