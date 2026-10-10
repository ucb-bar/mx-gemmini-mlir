"""Link the typed narrow MX/VPU object and check every source-derived output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.command_ir import emit_c
from mx_gemmini_support.narrow_vpu_chain import render_narrow_vpu_chain
from mx_gemmini_support.narrow_vpu_source import derive_narrow_vpu_resources
from mx_gemmini_support.resident_pair_graph import INPUTS
from mx_gemmini_support.resident_vpu_graph import OUTPUTS, lower_connected_fp8_vpu_pair
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure
from tools.emit_resident_vpu_object import _readout_commands


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
MARKER = "lowered narrow MX/VPU: C1 BF16 0, C1 0 codes 0 scales, C2 0 codes 0 scales"


def _driver(names: tuple[str, ...]) -> str:
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
static uint8_t c2_scales[64] __attribute__((aligned(64)));
static uint8_t c2_tiled[2048] __attribute__((aligned(64)));
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
    for (uint32_t col = 0; col < 32; ++col) {{
      uint32_t tiled = (((row / 16) * 2 + col / 16) * 16 + row % 16) * 16 + col % 16;
      c2_codes += c2_tiled[tiled] != c2_codes_ref[row * 32 + col];
    }}
  }}
  for (uint32_t i = 0; i < 128; ++i)
    c1_scale_errors += c1_scales[i] != c1_scales_ref[i];
  for (uint32_t i = 0; i < 64; ++i)
    c2_scale_errors += c2_scales[i] != c2_scales_ref[i];
  printf("lowered narrow MX/VPU: C1 BF16 %d, C1 %d codes %d scales, "
         "C2 %d codes %d scales\\n", bf16_errors, c1_codes,
         c1_scale_errors, c2_codes, c2_scale_errors);
  return bf16_errors || c1_codes || c1_scale_errors || c2_codes || c2_scale_errors;
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-dir", "bound-dir", "object-dir", "rtl-root",
                 "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--mx-opt", type=Path)
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
    pair = lower_connected_fp8_vpu_pair(
        bound, profile, resources, buffers={name: name for name in INPUTS},
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
            object_manifest.get("bound_mlir_sha256") != _sha(bound_path) or
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
        _run([str(args.mx_opt.resolve()), str(bound_path), "-o", "/dev/null"],
             cwd=args.out_dir, log=args.out_dir / "native_verify.log")
    build = args.out_dir / "build"
    build.mkdir()
    assembly = [".section .rodata", ".balign 64"]
    for name, data in sorted(resources.items()):
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
        "status": ("source_mx_vpu_and_narrow_mm2_matched_on_pinned_spike" if passed else
                   "narrow_mx_vpu_pair_failed_on_pinned_spike"),
        "first_shape_mnk": [64, 64, 64], "second_shape_mnk": [64, 32, 64],
        "reference_kind": "unchanged_nicolas_vpu_chain_left_output_block",
        "compared_c1_bf16_values": 4096,
        "compared_c1_fp8_codes": 4096, "compared_c1_e8m0_scales": 128,
        "compared_c2_fp8_codes": 2048, "compared_c2_e8m0_scales": 64,
        "capture_receipt_sha256": _sha(args.capture_dir / "receipt.json"),
        "binding_manifest_sha256": _sha(binding_path),
        "source_sha256": _sha(source), "first_source_sha256": _sha(first_source),
        "header_sha256": _sha(header),
        "bound_mlir_sha256": _sha(bound_path),
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
    (args.out_dir / "qualification_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{receipt['status']}: {elf}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
