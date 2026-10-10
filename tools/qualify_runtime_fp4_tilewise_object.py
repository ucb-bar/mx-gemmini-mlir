"""Run one four-tile FP4 MX+VPU object against two runtime payloads on Spike.

The second payload swaps 128-row M halves and 128-column N halves in the
source-generated FP4 fixture, including packed codes and E8M0 scales. Its
golden is the corresponding permutation of the independent source matrix
golden, followed by exact BF16 x2. The same mx_issue.o is called twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.quant_reference import exact_bf16_x2
from mx_gemmini_support.source_payload import load_bundle
from tools.compile_mx import _require_gitlink


ROOT = Path(__file__).resolve().parents[1]
SOURCE_EVIDENCE = ROOT / "docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593"
RTL_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
ABI_NAMES = ("activation", "activation_scales", "output_bf16",
             "scratch_output_scales", "weight", "weight_scales")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path,
                                   text=True).strip()


def _run(command: list[str], *, cwd: Path, log: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, check=False)
    log.write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")


def _swap_halves(data: bytes, rows: int, width: int) -> bytes:
    if len(data) != rows * width or width % 2:
        raise ValueError("FP4 runtime matrix row width differs from its source layout")
    half = width // 2
    return b"".join(data[row * width + half:(row + 1) * width] +
                    data[row * width:row * width + half] for row in range(rows))


def derive_second_payload(resources: dict[str, bytes]) -> dict[str, bytes]:
    """Swap complete M/N output halves without changing per-element MX math."""
    a = resources["activation"]
    a_scales = resources["activation_scales"]
    weight = resources["weight"]
    weight_scales = resources["weight_scales"]
    golden = resources["golden_bf16"]
    if (len(a) != 128 * 256 or len(a_scales) != 8 * 256 or
            len(weight) != 256 * 128 or len(weight_scales) != 8 * 256 or
            len(golden) != 256 * 256 * 2):
        raise ValueError("runtime permutation requires the generated 256x256 FP4 fixture")
    a_mid = len(a) // 2
    golden_mid = len(golden) // 2
    return {
        "activation": a[a_mid:] + a[:a_mid],
        "activation_scales": _swap_halves(a_scales, 8, 256),
        "weight": _swap_halves(weight, 256, 128),
        "weight_scales": _swap_halves(weight_scales, 8, 256),
        "golden_bf16": _swap_halves(golden[golden_mid:] + golden[:golden_mid],
                                     256, 256 * 2),
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
            receipt.get("shape_mnk") != [256, 256, 256] or
            receipt.get("transport") != "rocket_rocc" or
            receipt.get("object_sha256") != _sha(obj) or
            receipt.get("allocated_data_section_bytes") != 0 or
            receipt.get("embedded_operand_bytes") != 0 or
            receipt.get("embedded_golden_bytes") != 0 or
            tuple(entry["name"] for entry in receipt.get("buffer_abi", [])) != ABI_NAMES or
            [entry["minimum_bytes"] for entry in receipt["buffer_abi"]] !=
            [32768, 2048, 131072, 2048, 32768, 2048] or
            receipt["buffer_abi"][2]["layout"] != "output_tile_major_bf16"):
        raise ValueError("selected object lacks the four-tile runtime FP4 pointer ABI")
    manifest, first = load_bundle(SOURCE_EVIDENCE / "bundle")
    if (manifest.get("precision") != "FP4" or
            manifest.get("shape_mnk") != [256, 256, 256] or
            manifest.get("tile_mnk") != [128, 128, 128] or
            receipt["profile_sha256"] != manifest["profile_sha256"] or
            receipt["payload_manifest_sha256"] !=
            json.loads((SOURCE_EVIDENCE / "build/artifact_manifest.json").read_text())[
                "payload_manifest_sha256"]):
        raise ValueError("source fixture differs from the compiled FP4 object")
    second = derive_second_payload(first)
    if any(first[name] == second[name] for name in (
            "activation", "activation_scales", "weight", "weight_scales", "golden_bf16")):
        raise ValueError("runtime rebinding needs distinct packed operands, scales, and golden")
    expected = {"first": exact_bf16_x2(first["golden_bf16"]),
                "second": exact_bf16_x2(second["golden_bf16"])}

    software = args.rtl_root / "software/gemmini-rocc-tests"
    extension = args.rtl_root / "software/libgemmini"
    _require_gitlink(args.rtl_root, "software/gemmini-rocc-tests")
    _require_gitlink(args.rtl_root, "software/libgemmini")
    bench = software / "riscv-tests/benchmarks/common"
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if not cc.is_file() or not spike.is_file() or not (bench / "test.ld").is_file():
        parser.error("selected RISC-V toolchain or Gemmini benchmark runtime is incomplete")
    args.out_dir.mkdir(parents=True)
    shutil.copyfile(args.object_dir / "mx_issue.h", args.out_dir / "mx_issue.h")
    data = args.out_dir / "data"
    data.mkdir()
    source_names = ("activation", "activation_scales", "weight",
                    "weight_scales", "expected_bf16")
    assembly = [".section .rodata", ""]
    for label, resources in (("first", first), ("second", second)):
        for name in source_names:
            payload = expected[label] if name == "expected_bf16" else resources[name]
            filename = f"{label}_{name}.bin"
            (data / filename).write_bytes(payload)
            assembly.extend((".p2align 6", f".globl {label}_{name}",
                             f"{label}_{name}:", f'.incbin "data/{filename}"', ""))
    data_source = args.out_dir / "mx_runtime_data.S"
    data_source.write_text("\n".join(assembly))
    externs = "".join(f"extern const uint8_t {label}_{name}[];\n"
                      for label in ("first", "second") for name in source_names)
    driver = args.out_dir / "mx_runtime_driver.c"
    driver.write_text(f'''#include <stdint.h>
#include <stdio.h>
#include "mx_issue.h"
{externs}
static uint8_t first_output[131072] __attribute__((aligned(64)));
static uint8_t second_output[131072] __attribute__((aligned(64)));
static uint8_t first_scratch[2048] __attribute__((aligned(64)));
static uint8_t second_scratch[2048] __attribute__((aligned(64)));

static int compare(const uint8_t *actual_bytes, const uint8_t *expected_bytes,
                   const char *label) {{
  const uint16_t *actual = (const uint16_t *)actual_bytes;
  const uint16_t *expected = (const uint16_t *)expected_bytes;
  int errors = 0;
  for (int i = 0; i < 65536; ++i) {{
    const int row = i / 256;
    const int col = i % 256;
    const int tile = (row / 128) * 2 + col / 128;
    const int local = (row % 128) * 128 + col % 128;
    const int physical = tile * 16384 + local;
    if (actual[physical] != expected[i]) {{
      if (errors < 8)
        printf("%s mismatch %d: got=0x%04x expected=0x%04x\\n",
               label, i, actual[physical], expected[i]);
      ++errors;
    }}
  }}
  return errors;
}}

int main(void) {{
  mx_issue(first_activation, first_activation_scales, first_output,
           first_scratch, first_weight, first_weight_scales);
  int first_errors = compare(first_output, first_expected_bf16, "first");
  mx_issue(second_activation, second_activation_scales, second_output,
           second_scratch, second_weight, second_weight_scales);
  int second_errors = compare(second_output, second_expected_bf16, "second");
  first_errors += compare(first_output, first_expected_bf16, "first_after_second");
  printf("runtime FP4 tilewise: %d/131072 BF16 mismatches\\n",
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
    sources = [driver, data_source, *sorted(bench.glob("*.c")),
               *sorted(bench.glob("*.S"))]
    objects = [obj]
    for index, source in enumerate(sources):
        compiled_obj = args.out_dir / f"runtime_{index}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(compiled_obj)],
             cwd=args.out_dir, log=args.out_dir / f"compile_{index}.log")
        objects.append(compiled_obj)
    elf = args.out_dir / "mx_runtime_fp4_tilewise.elf"
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
              "runtime FP4 tilewise: 0/131072 BF16 mismatches" in result.stdout)
    index = {
        "schema": "mx_gemmini.runtime_fp4_tilewise_object_spike.v1",
        "status": "two_runtime_fp4_tilewise_payloads_matched_on_pinned_spike" if passed else
                  "runtime_fp4_tilewise_payload_failed_on_pinned_spike",
        "scope": "one compiled four-tile MX+VPU object with two runtime pointer sets; "
                 "second payload is a source-derived row/column permutation; no Muon/FPGA claim",
        "object_sha256": _sha(obj),
        "object_manifest_sha256": _sha(args.object_dir / "object_manifest.json"),
        "qualifier_sha256": _sha(Path(__file__)),
        "elf_sha256": _sha(elf), "spike_log_sha256": _sha(spike_log),
        "extension_sha256": _sha(so), "spike_exit_code": result.returncode,
        "profile_sha256": receipt["profile_sha256"],
        "source_bundle_manifest_sha256": _sha(SOURCE_EVIDENCE / "bundle/manifest.json"),
        "rtl_revision": _revision(args.rtl_root),
        "riscv_gcc_sha256": _sha(cc), "spike_sha256": _sha(spike),
        "compared_bf16_outputs": 131072 if passed else 0,
        "permutation": "swap_M_128_rows_and_N_128_columns_v1",
        "payloads": [{"name": label,
                      **{key + "_sha256": hashlib.sha256(resources[key]).hexdigest()
                         for key in ("activation", "activation_scales", "weight",
                                     "weight_scales", "golden_bf16")},
                      "expected_x2_bf16_sha256": hashlib.sha256(expected[label]).hexdigest()}
                     for label, resources in (("first", first), ("second", second))],
    }
    if args.baseline_index and json.loads(args.baseline_index.read_text()) != index:
        raise ValueError("runtime FP4 tilewise object evidence differs from baseline")
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"runtime FP4 tilewise object failed on Spike; see {spike_log}")
    print("one four-tile FP4 MX+VPU object matched 131,072 BF16 runtime outputs")


if __name__ == "__main__":
    main()
