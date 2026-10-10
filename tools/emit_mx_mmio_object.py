"""Emit a source-bound Muon RV32 MMIO issuer object for an MX program.

This checks the Radiance gateway register protocol and builds a data-free
object. A Radiance SoC profile, linked Muon kernel, and execution receipt are
still required before this object can be qualified for a device image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from mx_gemmini_support.command_ir import Command, Fence, WaitIdle, emit_c
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.transport_lowering import issuer_commands
from tools.emit_mx_object import _referenced_buffers


_GATEWAY_DEFINES = {
    "GEMMINI_CTRL": "0x00084000",
    "GEMMINI_RS1_OFFSET": "0x10",
    "GEMMINI_RS2_OFFSET": "0x18",
    "GEMMINI_INST_OFFSET": "0x0",
    "GEMMINI_BUSY_OFFSET": "0x20",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def _verify_gateway_header(path: Path) -> None:
    source = path.read_text()
    for name, offset in _GATEWAY_DEFINES.items():
        definition = rf"^#define\s+{name}\s+{offset}\s*$"
        if re.search(definition, source, re.MULTILINE) is None:
            raise ValueError(f"Radiance MX gateway {name} register differs")
    for fragment in ("store64_shared(GEMMINI_CTRL, GEMMINI_RS1_OFFSET",
                     "store64_shared(GEMMINI_CTRL, GEMMINI_RS2_OFFSET",
                     "store_shared  (GEMMINI_CTRL, GEMMINI_INST_OFFSET",
                     "load32_shared(GEMMINI_BUSY_ADDR)"):
        if fragment not in source:
            raise ValueError("Radiance MX gateway shared-memory transaction differs")
    for fragment in ("(0x7B)", "(3 << 12)", "((funct) << 25)"):
        if fragment not in source:
            raise ValueError("Radiance MX gateway instruction encoding differs")


def _verify_address_header(path: Path) -> None:
    source = path.read_text()
    if (re.search(r"^#define\s+RAD_HOST_GPU_DRAM_BASE\s+0x100000000ul\s*$",
                  source, re.MULTILINE) is None or
            "static_cast<uint64_t>(addr) | RAD_HOST_GPU_DRAM_BASE" not in source):
        raise ValueError("Radiance Muon-to-MX global address conversion differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlir", "bundle", "profile", "rtl-root", "radiance-root",
                 "muon-clang", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("mlir", "bundle", "profile", "rtl_root", "radiance_root",
                 "muon_clang", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    if not args.muon_clang.is_file():
        parser.error("selected Muon compiler does not exist")
    gateway = args.radiance_root / "lib/include/mxgemmini_mmio.h"
    _verify_gateway_header(gateway)
    address_header = args.radiance_root / "lib/include/radiance.h"
    _verify_address_header(address_header)
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    manifest, resources = load_bundle(args.bundle)
    program = lower_bound_source(args.mlir.read_text(), profile, manifest,
                                 resources, mode="rtl_alternating")
    buffers = _referenced_buffers(program, manifest)
    commands = issuer_commands(program, "muon_mmio")
    names = tuple(entry["name"] for entry in buffers)
    issuer = emit_c(list(commands), transport="muon_mmio", buffers=names)
    args.out_dir.mkdir(parents=True)
    source = args.out_dir / "mx_issue.c"
    source.write_text(issuer)
    header = args.out_dir / "mx_issue.h"
    header.write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n\n"
        "#include <stdint.h>\n"
        "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n"
        "/* Buffer order and sizes are in object_manifest.json. */\n"
        f"void mx_issue({', '.join(f'const void *{name}' for name in names)}, "
        "uintptr_t mx_control_base);\n\n"
        "#ifdef __cplusplus\n}\n#endif\n\n#endif\n")
    physical = args.out_dir / "physical_program.json"
    physical.write_text(json.dumps(program.receipt(), indent=2, sort_keys=True) + "\n")
    obj = args.out_dir / "mx_issue.o"
    command = [str(args.muon_clang), "--target=riscv32-unknown-elf",
               "-Xclang", "-target-feature", "-Xclang", "+vortex",
               "-march=rv32im_zfinx_zhinx", "-mabi=ilp32", "-mcmodel=medany",
               "-std=gnu11", "-O2", "-ffreestanding", "-fno-builtin",
               "-fno-common", "-fdata-sections", "-ffunction-sections",
               f"-ffile-prefix-map={args.out_dir}=.",
               "-c", str(source), "-o", str(obj)]
    compiled = subprocess.run(command, cwd=args.out_dir, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              check=False)
    (args.out_dir / "compile.log").write_text(compiled.stdout)
    if compiled.returncode:
        raise RuntimeError("Muon MMIO issuer build failed; see compile.log")
    bin_dir = args.muon_clang.parent
    nm = bin_dir / "llvm-nm"
    readelf = bin_dir / "llvm-readelf"
    objdump = bin_dir / "llvm-objdump"
    if not all(tool.is_file() for tool in (nm, readelf, objdump)):
        parser.error("selected Muon toolchain lacks llvm-nm, llvm-readelf, or llvm-objdump")
    defined = subprocess.check_output([str(nm), "-g", "--defined-only", str(obj)],
                                      text=True).splitlines()
    undefined = subprocess.check_output([str(nm), "-u", str(obj)],
                                        text=True).splitlines()
    if len(defined) != 1 or defined[0].split()[-2:] != ["T", "mx_issue"] or undefined:
        raise ValueError("Muon issuer unexpectedly defines data or imports symbols")
    elf = subprocess.check_output([str(readelf), "-h", str(obj)], text=True)
    if "Class:                             ELF32" not in elf or "Machine:                           RISC-V" not in elf:
        raise ValueError("Muon issuer object is not RISC-V ELF32")
    sections = subprocess.check_output([str(readelf), "-SW", str(obj)], text=True)
    allocated_data = 0
    for line in sections.splitlines():
        found = re.match(r"^\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+[0-9a-f]+\s+"
                         r"[0-9a-f]+\s+([0-9a-f]+)\s+", line)
        if found and found.group(1).startswith((".data", ".bss", ".rodata", ".sdata", ".sbss")):
            allocated_data += int(found.group(2), 16)
    if allocated_data:
        raise ValueError("Muon issuer embeds operand or golden data")
    disassembly = args.out_dir / "disassembly.txt"
    disassembly.write_text(subprocess.check_output(
        [str(objdump), "-d", str(obj)], text=True).replace(str(obj), obj.name, 1))
    assembly = disassembly.read_text()
    shared_stores = len(re.findall(r"\bsw\.shared\b", assembly))
    shared_loads = len(re.findall(r"\blw\.shared\b", assembly))
    physical_commands = sum(isinstance(item, Command) for item in commands)
    busy_waits = sum(isinstance(item, WaitIdle) for item in commands)
    if shared_stores != 5 * physical_commands or shared_loads < busy_waits:
        raise ValueError("Muon gateway was not lowered to the expected shared-memory instructions")
    receipt = {
        "schema": "mx_gemmini.muon_mmio_object.v1",
        "status": "muon_rv32_mmio_issuer_built_unqualified",
        "qualification": "structural_object_only",
        "transport": "muon_mmio", "physical_mode": "rtl_alternating",
        "shape_mnk": list(program.shape), "buffer_abi": buffers,
        "completion_fences": sum(isinstance(step.command, Fence)
                                 for step in program.steps),
        "gateway_busy_waits": sum(isinstance(item, WaitIdle) for item in commands),
        "shared_gateway_stores": shared_stores,
        "shared_gateway_loads": shared_loads,
        "command_count": len(program.steps),
        "allocated_data_section_bytes": allocated_data,
        "embedded_operand_bytes": 0, "embedded_golden_bytes": 0,
        "profile_sha256": program.profile_sha256,
        "payload_manifest_sha256": program.payload_manifest_sha256,
        "bound_mlir_sha256": _sha(args.mlir),
        "source_bundle_manifest_sha256": _sha(args.bundle / "manifest.json"),
        "physical_program_sha256": _sha(physical),
        "disassembly_sha256": _sha(disassembly),
        "issuer_c_sha256": _sha(source), "issuer_h_sha256": _sha(header),
        "object_emitter_sha256": _sha(Path(__file__)),
        "object_sha256": _sha(obj), "muon_clang_sha256": _sha(args.muon_clang),
        "radiance_gateway_header_sha256": _sha(gateway),
        "radiance_gateway_control_base": 0x00084000,
        "radiance_address_header_sha256": _sha(address_header),
        "radiance_host_gpu_dram_base": 0x100000000,
        "radiance_revision": _revision(args.radiance_root),
        "rtl_revision": _revision(args.rtl_root),
        "defined_symbol": "mx_issue", "undefined_symbols": [],
    }
    (args.out_dir / "object_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"data-free Muon RV32 MX MMIO issuer: {obj}")


if __name__ == "__main__":
    main()
