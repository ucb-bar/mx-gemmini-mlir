"""Build a source-payload-bound Muon MX kernel from a generated issuer object.

The linked ELF contains checked operand bytes and a BF16 verifier. It is not
numerically qualified until executed on a compatible mixed-engine target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from mx_gemmini_support.quant_reference import exact_bf16_x2
from mx_gemmini_support.source_payload import load_bundle, manifest_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], cwd: Path, log: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Muon payload build failed ({result.returncode}); see {log}")


def _checked_payload(object_dir: Path, bundle: Path) -> tuple[dict, dict[str, bytes],
                                                                list[dict], bytes]:
    receipt = json.loads((object_dir / "object_manifest.json").read_text())
    source_manifest, resources = load_bundle(bundle)
    physical = json.loads((object_dir / "physical_program.json").read_text())
    if (receipt.get("schema") != "mx_gemmini.muon_mmio_object.v1" or
            receipt.get("transport") != "muon_mmio" or
            receipt.get("object_sha256") != _sha(object_dir / "mx_issue.o") or
            receipt.get("issuer_h_sha256") != _sha(object_dir / "mx_issue.h") or
            receipt.get("physical_program_sha256") !=
            _sha(object_dir / "physical_program.json") or
            receipt.get("source_bundle_manifest_sha256") != _sha(bundle / "manifest.json") or
            receipt.get("payload_manifest_sha256") != manifest_sha256(source_manifest) or
            receipt.get("profile_sha256") != source_manifest.get("profile_sha256") or
            receipt.get("shape_mnk") != source_manifest.get("shape_mnk") or
            physical.get("shape_mnk") != receipt.get("shape_mnk")):
        raise ValueError("Muon object and source bundle differ")
    buffers = receipt["buffer_abi"]
    names = [entry["name"] for entry in buffers]
    if (not names or len(names) != len(set(names)) or
            any(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) is None
                for name in names) or
            [entry["position"] for entry in buffers] != list(range(len(buffers))) or
            {entry["role"] for entry in buffers} - {"read", "write", "scratch"} or
            not any(entry["name"] == "output_bf16" and entry["role"] == "write"
                    for entry in buffers)):
        raise ValueError("Muon object has an unsupported buffer ABI")
    for entry in buffers:
        if (entry["minimum_bytes"] <= 0 or entry["alignment_bytes"] != 64 or
                (entry["role"] == "read" and
                 len(resources.get(entry["name"], b"")) != entry["minimum_bytes"])):
            raise ValueError(f"Muon buffer {entry['name']} differs from its checked payload")
    if (any(entry["name"] != "output_bf16" for entry in buffers
            if entry["role"] == "write") or
            "golden_bf16" not in resources or
            len(resources["golden_bf16"]) != next(
                entry["minimum_bytes"] for entry in buffers if entry["name"] == "output_bf16")):
        raise ValueError("Muon payload linker currently needs BF16 output and golden")
    if physical.get("golden_derivation") == "bf16_exact_multiply_by_two":
        expected = exact_bf16_x2(resources["golden_bf16"])
        if hashlib.sha256(expected).hexdigest() != physical.get("derived_expected_bf16_sha256"):
            raise ValueError("derived VPU golden differs from physical program")
    elif physical.get("source_golden_preserving") is True:
        expected = resources["golden_bf16"]
    else:
        raise ValueError("Muon payload linker has no checked BF16 output reference")
    return receipt, resources, buffers, expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("object-dir", "bundle", "radiance-root", "muon-clangxx", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("object_dir", "bundle", "radiance_root", "muon_clangxx", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    receipt, resources, buffers, expected = _checked_payload(args.object_dir,
                                                              args.bundle)
    lib = args.radiance_root / "lib"
    runtime = lib / "libmuonrt.a"
    linker = lib / "linker/mu_link.ld"
    tohost = lib / "tohost.S"
    if not all(path.is_file() for path in (runtime, linker, tohost, args.muon_clangxx)):
        parser.error("selected Muon runtime, linker script, or compiler is missing")
    args.out_dir.mkdir(parents=True)
    read_buffers = [entry for entry in buffers if entry["role"] == "read"]
    for entry in read_buffers:
        (args.out_dir / f"{entry['name']}.bin").write_bytes(resources[entry["name"]])
    (args.out_dir / "mx_expected_bf16.bin").write_bytes(expected)
    assembly = [".section .rodata", ".balign 64"]
    for name in sorted([entry["name"] for entry in read_buffers] + ["mx_expected_bf16"]):
        assembly.extend((f".globl {name}", f".type {name}, @object", f"{name}:",
                         f'.incbin "{name}.bin"', f".size {name}, .-{name}",
                         ".balign 64"))
    assembly.append('.section .note.GNU-stack,"",@progbits')
    data_source = args.out_dir / "mx_payload.S"
    data_source.write_text("\n".join(assembly) + "\n")
    declaration = "".join(f"extern const uint8_t {entry['name']}[];\n"
                          for entry in read_buffers)
    declaration += "extern const uint8_t mx_expected_bf16[];\n"
    runtime_buffers = [entry for entry in buffers if entry["role"] != "read"]
    declaration += "".join(
        f"uint8_t {entry['name']}[{entry['minimum_bytes']}] "
        "__attribute__((aligned(64)));\n" for entry in runtime_buffers)
    arguments = ", ".join(entry["name"] for entry in buffers)
    output_bytes = next(entry["minimum_bytes"] for entry in buffers
                        if entry["name"] == "output_bf16")
    kernel = args.out_dir / "mx_kernel.cpp"
    kernel.write_text(
        '#include <stdint.h>\n#include <mu_intrinsics.h>\n'
        '#include <mu_schedule.h>\n#include "mx_issue.h"\n\n'
        'extern "C" {\n' + declaration + '}\n'
        'volatile uint32_t mx_mismatch_count = UINT32_MAX;\n'
        'volatile uint32_t mx_completed = 0;\n'
        'static void worker(void *, uint32_t lane, uint32_t, uint32_t block) {\n'
        '  if (lane != 0 || block != 0) return;\n'
        f'  mx_issue({arguments}, UINT32_C(0x{receipt["radiance_gateway_control_base"]:08x}));\n'
        '  mu_fence();\n'
        '  const uint16_t *got = reinterpret_cast<const uint16_t *>(output_bf16);\n'
        '  const uint16_t *expected = reinterpret_cast<const uint16_t *>(mx_expected_bf16);\n'
        '  uint32_t mismatches = 0;\n'
        f'  for (uint32_t i = 0; i < {output_bytes // 2}; ++i)\n'
        '    mismatches += got[i] != expected[i];\n'
        '  mx_mismatch_count = mismatches;\n'
        '  mu_fence();\n'
        '  mx_completed = 1;\n'
        '  mu_fence();\n'
        '}\n'
        'extern "C" int main() {\n'
        '  mu_schedule(worker, nullptr, 1);\n'
        '  return 0;\n}\n')
    flags = ["--target=riscv32-unknown-elf", "-Xclang", "-target-feature",
             "-Xclang", "+vortex", "-march=rv32im_zfinx_zhinx", "-mabi=ilp32",
             "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-exceptions",
             "-fno-rtti", "-fdata-sections", "-ffunction-sections",
             "-stdlib=libc++", "-isystem", str(args.muon_clangxx.parent.parent / "include/c++/v1"),
             f"-ffile-prefix-map={args.out_dir}=."]
    caller = args.out_dir / "mx_kernel.o"
    payload = args.out_dir / "mx_payload.o"
    _run([str(args.muon_clangxx), *flags, f"-I{lib / 'include'}",
          f"-I{args.object_dir}", "-c", str(kernel), "-o", str(caller)],
         args.out_dir, args.out_dir / "compile_kernel.log")
    _run([str(args.muon_clangxx), *flags, "-c", str(data_source), "-o", str(payload)],
         args.out_dir, args.out_dir / "compile_payload.log")
    elf = args.out_dir / "mx_kernel.elf"
    _run([str(args.muon_clangxx), *flags, "-nodefaultlibs", "-nostartfiles",
          f"-Wl,-Bstatic,-T,{linker},-z,norelro", "-fuse-ld=lld",
          str(caller), str(payload), str(args.object_dir / "mx_issue.o"),
          str(runtime), str(tohost), "-o", str(elf)],
         args.out_dir, args.out_dir / "link.log")
    bin_dir = args.muon_clangxx.parent
    readelf, nm = bin_dir / "llvm-readelf", bin_dir / "llvm-nm"
    if not readelf.is_file() or not nm.is_file():
        parser.error("selected Muon toolchain lacks llvm-readelf or llvm-nm")
    elf_header = subprocess.check_output([str(readelf), "-h", str(elf)], text=True)
    symbols = subprocess.check_output([str(nm), "-g", str(elf)], text=True)
    sized_symbols = subprocess.check_output([str(nm), "-S", str(elf)], text=True)
    undefined = subprocess.check_output([str(nm), "-u", str(elf)], text=True)
    required = {"_start", "main", "mx_issue", "mx_expected_bf16",
                "output_bf16", "mx_mismatch_count", "mx_completed"}
    required.update(entry["name"] for entry in read_buffers)
    if ("Class:                             ELF32" not in elf_header or
            "Machine:                           RISC-V" not in elf_header or
            any(re.search(rf"\b[BDRT] {name}$", symbols, re.MULTILINE) is None
                for name in required) or undefined.strip()):
        raise ValueError("Muon payload ELF is not complete or lacks a required symbol")
    expected_sizes = {entry["name"]: entry["minimum_bytes"] for entry in buffers}
    expected_sizes["mx_expected_bf16"] = len(expected)
    for name, size in expected_sizes.items():
        match = re.search(rf"^\S+\s+([0-9a-f]+)\s+[BDR]\s+{name}$",
                          sized_symbols, re.MULTILINE)
        if match is None or int(match.group(1), 16) != size:
            raise ValueError(f"Muon ELF {name} symbol does not cover its checked payload")
    file_hashes = {path.name: _sha(path) for path in sorted(args.out_dir.iterdir())
                   if path.is_file() and path.name not in {"payload_manifest.json",
                                                          "compile_kernel.log",
                                                          "compile_payload.log",
                                                          "link.log"}}
    report = {
        "schema": "mx_gemmini.muon_payload_elf.v1",
        "status": "source_payload_bound_muon_elf_unexecuted",
        "qualification": "linked_payload_and_verifier_only",
        "object_manifest_sha256": _sha(args.object_dir / "object_manifest.json"),
        "bundle_manifest_sha256": _sha(args.bundle / "manifest.json"),
        "physical_program_sha256": _sha(args.object_dir / "physical_program.json"),
        "payload_linker_sha256": _sha(Path(__file__)),
        "muon_clangxx_sha256": _sha(args.muon_clangxx),
        "muon_runtime_sha256": _sha(runtime),
        "muon_linker_script_sha256": _sha(linker),
        "muon_tohost_sha256": _sha(tohost),
        "shape_mnk": receipt["shape_mnk"],
        "bf16_elements_to_compare": output_bytes // 2,
        "expected_bf16_sha256": hashlib.sha256(expected).hexdigest(),
        "gateway_control_base": receipt["radiance_gateway_control_base"],
        "bound_buffers": [entry["name"] for entry in buffers],
        "defined_symbols_checked": sorted(required),
        "undefined_symbols": [],
        "files_sha256": file_hashes,
    }
    (args.out_dir / "payload_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"source-bound Muon RV32 MX ELF (unexecuted): {elf}")


if __name__ == "__main__":
    main()
