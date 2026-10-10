"""Execute one source-bound FP6 Rocket object with two runtime payloads.

The second payload permutes 64-row M halves and 64-column N halves in the
generated Radiance FP6 128x128x1024 fixture. Packed codes, E8M0 scales,
and A/B/C per-pair LUT banks move together. The expected BF16 matrix is the
same permutation of the independent source golden. No data enters mx_issue.o.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.source_payload import load_bundle, manifest_sha256
from tools.compile_mx import _require_gitlink
from tools.qualify_runtime_fp4_tilewise_object import _revision, _run, _sha, _swap_halves


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/evidence/radiance_fp6_fullout_266c593/fp6_128x128x1024"
RTL_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
ABI_NAMES = ("activation", "activation_lut", "activation_scales",
             "output_bf16", "output_lut", "scratch_output_scales",
             "weight", "weight_lut", "weight_scales")


def derive_second_payload(first: dict[str, bytes]) -> dict[str, bytes]:
    a, a_scales = first["activation"], first["activation_scales"]
    b, b_scales = first["weight"], first["weight_scales"]
    golden = first["golden_bf16"]
    if (len(a) != 64 * 1024 or len(a_scales) != 32 * 128 or
            len(b) != 1024 * 64 or len(b_scales) != 32 * 128 or
            any(len(first[name]) != 64 * 12 for name in
                ("activation_lut", "weight_lut", "output_lut")) or
            len(golden) != 128 * 128 * 2):
        raise ValueError("runtime permutation requires the generated FP6 source shape")
    return {
        "activation": a[len(a) // 2:] + a[:len(a) // 2],
        "activation_lut": first["activation_lut"][384:] +
                          first["activation_lut"][:384],
        "activation_scales": _swap_halves(a_scales, 32, 128),
        "output_lut": first["output_lut"][384:] + first["output_lut"][:384],
        "weight": _swap_halves(b, 1024, 64),
        "weight_lut": first["weight_lut"][384:] + first["weight_lut"][:384],
        "weight_scales": _swap_halves(b_scales, 32, 128),
        "golden_bf16": _swap_halves(golden[len(golden) // 2:] +
                                     golden[:len(golden) // 2], 128, 128 * 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("object-dir", "rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--baseline-index", type=Path)
    args = parser.parse_args()
    for name in ("object_dir", "rtl_root", "riscv_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    if _revision(args.rtl_root) != RTL_REVISION:
        parser.error("selected MX RTL differs from Nicolas's pinned branch")
    receipt = json.loads((args.object_dir / "object_manifest.json").read_text())
    obj = args.object_dir / "mx_issue.o"
    if (receipt.get("status") != "rv64_rocc_issuer_object_built" or
            receipt.get("shape_mnk") != [128, 128, 1024] or
            receipt.get("transport") != "rocket_rocc" or
            receipt.get("object_sha256") != _sha(obj) or
            receipt.get("allocated_data_section_bytes") != 0 or
            receipt.get("embedded_operand_bytes") != 0 or
            receipt.get("embedded_golden_bytes") != 0 or
            tuple(entry["name"] for entry in receipt.get("buffer_abi", [])) != ABI_NAMES or
            [entry["minimum_bytes"] for entry in receipt["buffer_abi"]] !=
            [65536, 768, 4096, 32768, 768, 2048, 65536, 768, 4096] or
            receipt["buffer_abi"][3]["layout"] != "row_major_bf16"):
        raise ValueError("selected object lacks the FP6 source pointer ABI")
    manifest, first = load_bundle(SOURCE / "bundle")
    if (manifest.get("precision") != "FP6" or
            manifest.get("shape_mnk") != [128, 128, 1024] or
            manifest.get("tile_mnk") != [128, 128, 512] or
            receipt["bound_mlir_sha256"] != _sha(SOURCE / "bound.mlir") or
            receipt["payload_manifest_sha256"] != manifest_sha256(manifest) or
            receipt["source_bundle_manifest_sha256"] != _sha(SOURCE / "bundle/manifest.json") or
            receipt["profile_sha256"] != manifest["profile_sha256"]):
        raise ValueError("FP6 source fixture differs from compiled object")
    second = derive_second_payload(first)
    if any(first[name] == second[name] for name in (
            "activation", "activation_lut", "activation_scales", "output_lut",
            "weight", "weight_lut", "weight_scales", "golden_bf16")):
        raise ValueError("runtime payloads must have different codes, scales, LUTs, and golden")

    software = args.rtl_root / "software/gemmini-rocc-tests"
    extension = args.rtl_root / "software/libgemmini"
    _require_gitlink(args.rtl_root, "software/gemmini-rocc-tests")
    _require_gitlink(args.rtl_root, "software/libgemmini")
    bench = software / "riscv-tests/benchmarks/common"
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if not cc.is_file() or not spike.is_file() or not (bench / "test.ld").is_file():
        parser.error("selected RISC-V toolchain or benchmark runtime is incomplete")
    args.out_dir.mkdir(parents=True)
    shutil.copyfile(args.object_dir / "mx_issue.h", args.out_dir / "mx_issue.h")
    data = args.out_dir / "data"
    data.mkdir()
    source_names = ("activation", "activation_lut", "activation_scales",
                    "output_lut", "weight", "weight_lut", "weight_scales",
                    "golden_bf16")
    assembly = [".section .rodata", ""]
    for label, resources in (("first", first), ("second", second)):
        for name in source_names:
            filename = f"{label}_{name}.bin"
            (data / filename).write_bytes(resources[name])
            assembly.extend((".p2align 6", f".globl {label}_{name}",
                             f"{label}_{name}:", f'.incbin "data/{filename}"', ""))
    (args.out_dir / "mx_runtime_data.S").write_text("\n".join(assembly))
    externs = "".join(f"extern const uint8_t {label}_{name}[];\n"
                      for label in ("first", "second") for name in source_names)
    (args.out_dir / "mx_runtime_driver.c").write_text(f'''#include <stdint.h>
#include <stdio.h>
#include "mx_issue.h"
{externs}
static uint8_t first_output[32768] __attribute__((aligned(64)));
static uint8_t second_output[32768] __attribute__((aligned(64)));
static uint8_t first_scratch[2048] __attribute__((aligned(64)));
static uint8_t second_scratch[2048] __attribute__((aligned(64)));

static int compare(const uint8_t *output, const uint8_t *golden, const char *label) {{
  const uint16_t *got = (const uint16_t *)output;
  const uint16_t *want = (const uint16_t *)golden;
  int errors = 0;
  for (int i = 0; i < 16384; ++i) {{
    if (got[i] != want[i]) {{
      if (errors < 8)
        printf("%s mismatch %d: got=0x%04x expected=0x%04x\\n",
               label, i, got[i], want[i]);
      ++errors;
    }}
  }}
  return errors;
}}

int main(void) {{
  mx_issue(first_activation, first_activation_lut, first_activation_scales,
           first_output, first_output_lut, first_scratch, first_weight,
           first_weight_lut, first_weight_scales);
  int first_errors = compare(first_output, first_golden_bf16, "first");
  mx_issue(second_activation, second_activation_lut, second_activation_scales,
           second_output, second_output_lut, second_scratch, second_weight,
           second_weight_lut, second_weight_scales);
  int second_errors = compare(second_output, second_golden_bf16, "second");
  first_errors += compare(first_output, first_golden_bf16, "first_after_second");
  printf("runtime source FP6: %d/32768 BF16 mismatches\\n",
         first_errors + second_errors);
  return first_errors || second_errors;
}}
''')
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             f"-ffile-prefix-map={args.out_dir}=.",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [args.out_dir / "mx_runtime_driver.c", args.out_dir / "mx_runtime_data.S",
               *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S"))]
    objects = [obj]
    for index, source in enumerate(sources):
        compiled_obj = args.out_dir / f"runtime_{index}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(compiled_obj)],
             cwd=args.out_dir, log=args.out_dir / f"compile_{index}.log")
        objects.append(compiled_obj)
    elf = args.out_dir / "mx_runtime_fp6_source.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(path) for path in objects),
          "-lm", "-lgcc", "-o", str(elf)],
         cwd=args.out_dir, log=args.out_dir / "link.log")
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = args.out_dir / "libgemmini.so"
    _run(["g++", "-L", str(args.riscv_root / "lib"),
          f"-Wl,-rpath,{args.riscv_root / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(args.riscv_root / "include"), "-fPIC", "-O3",
          *(str(path) for path in extension_sources)],
         cwd=args.out_dir, log=args.out_dir / "extension_build.log")
    result = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini",
                             str(elf)], cwd=args.out_dir, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            check=False)
    spike_log = args.out_dir / "spike.log"
    spike_log.write_text(result.stdout)
    passed = (result.returncode == 0 and
              "runtime source FP6: 0/32768 BF16 mismatches" in result.stdout)
    index = {
        "schema": "mx_gemmini.runtime_fp6_source_object_spike.v1",
        "status": "two_runtime_fp6_source_payloads_matched_on_pinned_spike" if passed else
                  "runtime_fp6_source_payload_failed_on_pinned_spike",
        "scope": "one generated Radiance FP6 source-bound Rocket object with two runtime "
                 "pointer sets, packed-code and A/B/C LUT permutation, and a serial two-wave "
                 "schedule for the pinned Spike scale-selector limitation; no Muon/RTL/FPGA claim",
        "object_sha256": _sha(obj),
        "object_manifest_sha256": _sha(args.object_dir / "object_manifest.json"),
        "qualifier_sha256": _sha(Path(__file__)),
        "elf_sha256": _sha(elf), "spike_log_sha256": _sha(spike_log),
        "extension_sha256": _sha(so), "spike_exit_code": result.returncode,
        "profile_sha256": receipt["profile_sha256"],
        "source_bundle_manifest_sha256": _sha(SOURCE / "bundle/manifest.json"),
        "rtl_revision": _revision(args.rtl_root),
        "riscv_gcc_sha256": _sha(cc), "spike_sha256": _sha(spike),
        "compared_bf16_outputs": 32768 if passed else 0,
        "permutation": "swap_M_64_rows_and_N_64_columns_with_pair_LUT_banks_v1",
        "payloads": [{"name": label,
                      **{key + "_sha256": hashlib.sha256(resources[key]).hexdigest()
                         for key in source_names}}
                     for label, resources in (("first", first), ("second", second))],
    }
    if args.baseline_index and json.loads(args.baseline_index.read_text()) != index:
        raise ValueError("runtime FP6 source object evidence differs from baseline")
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"runtime FP6 source object failed on Spike; see {spike_log}")
    print("one generated Radiance FP6 object matched 32,768 BF16 runtime outputs")


if __name__ == "__main__":
    main()
