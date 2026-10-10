"""Link a source-bound runtime MX object and check every host output on Spike."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _require_gitlink, _run


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("object-dir", "mlir", "bundle", "profile", "rtl-root",
                 "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("object_dir", "mlir", "bundle", "profile", "rtl_root",
                 "riscv_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    manifest, resources = load_bundle(args.bundle)
    receipt_path = args.object_dir / "object_manifest.json"
    receipt = json.loads(receipt_path.read_text())
    obj = args.object_dir / "mx_issue.o"
    names = tuple(row["name"] for row in receipt["buffer_abi"])
    output_format = receipt.get("host_output_format")
    m, n, _ = manifest["shape_mnk"]
    code_name = ("source_fp6_packed" if output_format == "radiance_header_fp6"
                 else "golden_fp8")
    code_bytes = m * n // (2 if output_format == "radiance_header_fp6" else 1)
    if (receipt.get("schema") != "mx_gemmini.linkable_object.v1" or
            receipt.get("status") != "rv64_rocc_composed_object_built" or
            receipt.get("transport") != "rocket_rocc" or
            output_format not in {"radiance_header_fp8", "radiance_header_fp6"} or
            receipt.get("profile_sha256") != profile_sha256(profile) or
            receipt.get("profile_sha256") != manifest["profile_sha256"] or
            receipt.get("bound_mlir_sha256") != _sha(args.mlir) or
            receipt.get("source_bundle_manifest_sha256") != _sha(args.bundle / "manifest.json") or
            receipt.get("object_sha256") != _sha(obj) or
            receipt.get("allocated_data_section_bytes") != 0 or
            receipt.get("embedded_operand_bytes") != 0 or
            receipt.get("embedded_golden_bytes") != 0 or
            {"output_bf16", "output_quantized", "scratch_output_scales"} - set(names) or
            any(name not in resources for name in names if name not in {
                "output_bf16", "output_quantized", "scratch_output_scales"}) or
            len(resources[code_name]) != code_bytes or
            len(resources["golden_output_scales"]) != m * n // 32):
        raise ValueError("host object, typed source, bundle, or runtime ABI differs")
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
    shutil.copy2(args.object_dir / "mx_issue.h", args.out_dir / "mx_issue.h")
    data = args.out_dir / "data"
    data.mkdir()
    read_names = tuple(name for name in names if name in resources)
    stored = tuple(sorted(set(read_names) | {code_name, "golden_output_scales"}))
    assembly = [".section .rodata", ".balign 64"]
    for name in stored:
        (data / f"{name}.bin").write_bytes(resources[name])
        assembly.extend((f".globl {name}", f"{name}:",
                         f'.incbin "data/{name}.bin"', ".balign 64"))
    assembly.append('.section .note.GNU-stack,"",@progbits')
    (args.out_dir / "mx_data.S").write_text("\n".join(assembly) + "\n")
    externs = "".join(f"extern const uint8_t {name}[];\n" for name in stored)
    outputs = {
        "output_bf16": m * n * 2,
        "output_quantized": code_bytes,
        "scratch_output_scales": m * n // 32,
    }
    declarations = "".join(
        f"static uint8_t {name}[{length}] __attribute__((aligned(64)));\n"
        for name, length in outputs.items())
    arguments = ", ".join(names)
    driver = f'''#include <stdint.h>
#include <stdio.h>
#include "mx_issue.h"
{externs}{declarations}
int main(void) {{
  mx_issue({arguments});
  int code_errors = 0, scale_errors = 0;
  for (uint32_t i = 0; i < {code_bytes}; ++i)
    if (output_quantized[i] != {code_name}[i]) ++code_errors;
  for (uint32_t i = 0; i < {m * n // 32}; ++i)
    if (scratch_output_scales[i] != golden_output_scales[i]) ++scale_errors;
  printf("runtime host MX: %d/{code_bytes} code mismatches, "
         "%d/{m * n // 32} scale mismatches\\n", code_errors, scale_errors);
  return code_errors != 0 || scale_errors != 0;
}}
'''
    (args.out_dir / "mx_driver.c").write_text(driver)
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             f"-ffile-prefix-map={args.out_dir}=.",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [args.out_dir / "mx_driver.c", args.out_dir / "mx_data.S",
               *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S"))]
    objects = [obj]
    for index, source in enumerate(sources):
        built = args.out_dir / f"mx_{index}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(built)],
             cwd=args.out_dir, log=args.out_dir / f"compile_{index}.log")
        objects.append(built)
    elf = args.out_dir / "mx_host_object.elf"
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
    selected_extension = ("gemmini" if profile["geometry"]["mesh_columns"] == 16
                          else f"gemmini_dim{profile['geometry']['mesh_columns']}")
    run = subprocess.run([str(spike), f"--extlib={so}",
                          f"--extension={selected_extension}", str(elf)],
                         cwd=args.out_dir, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=False)
    log = args.out_dir / "spike.log"
    log.write_text(run.stdout)
    expected = (f"runtime host MX: 0/{code_bytes} code mismatches, "
                f"0/{m * n // 32} scale mismatches")
    passed = run.returncode == 0 and expected in run.stdout
    result = {
        "schema": "mx_gemmini.runtime_host_object_spike.v1",
        "status": "source_quantized_output_matched_on_pinned_spike" if passed else
                  "source_quantized_output_failed_on_pinned_spike",
        "host_output_format": output_format,
        "shape_mnk": manifest["shape_mnk"],
        "compared_codes": code_bytes if passed else 0,
        "compared_scales": m * n // 32 if passed else 0,
        "object_sha256": _sha(obj),
        "object_manifest_sha256": _sha(receipt_path),
        "bound_mlir_sha256": _sha(args.mlir),
        "bundle_manifest_sha256": _sha(args.bundle / "manifest.json"),
        "driver_sha256": _sha(args.out_dir / "mx_driver.c"),
        "elf_sha256": _sha(elf),
        "spike_log_sha256": _sha(log),
        "spike_exit_code": run.returncode,
        "profile_sha256": profile_sha256(profile),
        "riscv_gcc_sha256": _sha(cc),
        "spike_sha256": _sha(spike),
        "extension_sha256": _sha(so),
    }
    (args.out_dir / "index.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"host MX object failed source output comparison; see {log}")
    print(expected)


if __name__ == "__main__":
    main()
