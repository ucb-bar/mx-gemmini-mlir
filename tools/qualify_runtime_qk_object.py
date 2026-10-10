"""Link one compiler-emitted MX issuer object to two source QK payloads on Spike.

The same mx_issue.o receives different runtime activation, weight, scale, and
output pointers on consecutive calls. This proves RoCC object rebinding; the
Muon-to-MX shared-memory handoff remains a separate target-execution gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.source_payload import load_bundle
from tools.compile_mx import _require_gitlink


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def _run(command: list[str], *, cwd: Path, log: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, check=False)
    log.write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("object-dir", "first-bundle", "second-bundle",
                 "rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--baseline-index", type=Path)
    args = parser.parse_args()
    for name in ("object_dir", "first_bundle", "second_bundle", "rtl_root",
                 "riscv_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    obj_receipt = json.loads((args.object_dir / "object_manifest.json").read_text())
    obj = args.object_dir / "mx_issue.o"
    if (obj_receipt.get("status") != "rv64_rocc_issuer_object_built" or
            obj_receipt.get("shape_mnk") != [64, 64, 64] or
            obj_receipt.get("object_sha256") != _sha(obj) or
            obj_receipt.get("allocated_data_section_bytes") != 0 or
            [entry["name"] for entry in obj_receipt["buffer_abi"]] != [
                "activation", "activation_scales", "output_bf16",
                "scratch_output_scales", "weight", "weight_scales"]):
        raise ValueError("selected MX object lacks the source QK runtime pointer ABI")
    cases = []
    for label, directory, stage in (("first", args.first_bundle, "gqa_qk_head0_block0"),
                                    ("second", args.second_bundle, "gqa_qk_head7_block1")):
        manifest, resources = load_bundle(directory)
        if (manifest.get("source_derivation", {}).get("stage") != stage or
                manifest.get("shape_mnk") != [64, 64, 64] or
                manifest.get("precision") != "FP8" or
                manifest.get("profile_sha256") != obj_receipt["profile_sha256"] or
                manifest.get("source_header_sha256") !=
                json.loads((args.first_bundle / "manifest.json").read_text())[
                    "source_header_sha256"] or
                len(resources["golden_bf16"]) != 8192):
            raise ValueError(f"{label} QK payload differs from the pinned source stage")
        cases.append((label, manifest, resources, directory))
    if (_sha(args.first_bundle / "activation.bin") ==
            _sha(args.second_bundle / "activation.bin") or
            _sha(args.first_bundle / "weight.bin") ==
            _sha(args.second_bundle / "weight.bin") or
            _sha(args.first_bundle / "golden_bf16.bin") ==
            _sha(args.second_bundle / "golden_bf16.bin")):
        raise ValueError("runtime pointer test needs distinct Q/K payloads and outputs")

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
    source_names = ("activation", "activation_scales", "weight", "weight_scales",
                    "golden_bf16")
    assembly = [".section .rodata", ""]
    for label, _, resources, _ in cases:
        for name in source_names:
            filename = f"{label}_{name}.bin"
            (data / filename).write_bytes(resources[name])
            assembly.extend((".p2align 6", f".globl {label}_{name}",
                             f"{label}_{name}:", f'.incbin "data/{filename}"', ""))
    data_source = args.out_dir / "mx_runtime_data.S"
    data_source.write_text("\n".join(assembly))
    externs = "".join(f"extern const uint8_t {label}_{name}[];\n"
                      for label, _, _, _ in cases for name in source_names)
    driver = args.out_dir / "mx_runtime_driver.c"
    driver.write_text(f'''#include <stdint.h>
#include <stdio.h>
#include "mx_issue.h"
{externs}
static uint8_t first_output[8192] __attribute__((aligned(64)));
static uint8_t second_output[8192] __attribute__((aligned(64)));
static uint8_t first_scratch[2048] __attribute__((aligned(64)));
static uint8_t second_scratch[2048] __attribute__((aligned(64)));

static int compare(const uint8_t *actual_bytes, const uint8_t *expected_bytes,
                   const char *label) {{
  const uint16_t *actual = (const uint16_t *)actual_bytes;
  const uint16_t *expected = (const uint16_t *)expected_bytes;
  int errors = 0;
  for (int i = 0; i < 4096; ++i) {{
    if (actual[i] != expected[i]) {{
      if (errors < 8)
        printf("%s mismatch %d: got=0x%04x expected=0x%04x\\n",
               label, i, actual[i], expected[i]);
      ++errors;
    }}
  }}
  return errors;
}}

int main(void) {{
  mx_issue(first_activation, first_activation_scales, first_output,
           first_scratch, first_weight, first_weight_scales);
  int first_errors = compare(first_output, first_golden_bf16, "first");
  mx_issue(second_activation, second_activation_scales, second_output,
           second_scratch, second_weight, second_weight_scales);
  int second_errors = compare(second_output, second_golden_bf16, "second");
  first_errors += compare(first_output, first_golden_bf16, "first_after_second");
  printf("runtime MX QK: %d/8192 BF16 mismatches\\n",
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
    elf = args.out_dir / "mx_runtime_qk.elf"
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
    passed = result.returncode == 0 and "runtime MX QK: 0/8192 BF16 mismatches" in result.stdout
    receipt = {
        "schema": "mx_gemmini.runtime_qk_object_spike.v1",
        "status": ("two_runtime_qk_payloads_matched_on_pinned_spike" if passed else
                   "runtime_qk_payload_failed_on_pinned_spike"),
        "scope": "one RV64 RoCC issuer object rebound to two distinct QK payloads; no Muon execution",
        "object_sha256": _sha(obj),
        "object_manifest_sha256": _sha(args.object_dir / "object_manifest.json"),
        "qualifier_sha256": _sha(Path(__file__)),
        "elf_sha256": _sha(elf), "spike_log_sha256": _sha(spike_log),
        "extension_sha256": _sha(so),
        "spike_exit_code": result.returncode,
        "source_header_sha256": cases[0][1]["source_header_sha256"],
        "profile_sha256": obj_receipt["profile_sha256"],
        "rtl_revision": _revision(args.rtl_root),
        "riscv_gcc_sha256": _sha(cc), "spike_sha256": _sha(spike),
        "compared_bf16_outputs": 8192 if passed else 0,
        "payloads": [{"stage": manifest["source_derivation"]["stage"],
                      "manifest_sha256": _sha(directory / "manifest.json"),
                      "activation_sha256": _sha(directory / "activation.bin"),
                      "weight_sha256": _sha(directory / "weight.bin"),
                      "golden_bf16_sha256": _sha(directory / "golden_bf16.bin")}
                     for _, manifest, _, directory in cases],
    }
    if args.baseline_index and json.loads(args.baseline_index.read_text()) != receipt:
        raise ValueError("runtime QK object evidence differs from baseline")
    (args.out_dir / "index.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"runtime QK object failed on Spike; see {spike_log}")
    print("one MX RoCC object matched both runtime QK payloads: 8,192 BF16 outputs")


if __name__ == "__main__":
    main()
