"""Emit a linkable RV64 RoCC MX issuer with explicit runtime buffer pointers.

The typed, payload-bound MLIR and checked bundle specialize the physical
schedule. The object contains commands only: the caller supplies every input,
scale, scratch, and output pointer at invocation time. The bundle's operand
bytes and golden are never linked into this object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def _referenced_buffers(program, manifest: dict) -> list[dict]:
    uses: dict[str, list] = {}
    for step in program.steps:
        command = step.command
        if not isinstance(command, Command):
            continue
        for operand in (command.rs1, command.rs2):
            if operand.buffer is not None:
                uses.setdefault(operand.buffer, []).append(operand)
    m, n, _ = program.shape
    entries = []
    for name in sorted(uses):
        if name in manifest["resources"]:
            descriptor = manifest["resources"][name]
            length = descriptor["bytes"]
            role = "read"
            layout = descriptor["layout"]
        elif name == "output_bf16":
            length, role, layout = m * n * 2, "write", "row_major_bf16"
        elif name == "output_quantized":
            length = m * n // (2 if program.output_format in {"fp4_e2m1", "fp6_e3m2"} else 1)
            role = "write"
            layout = "tiled_quantized" if program.tiled_quant_readout else "row_major_codes"
        elif name == "scratch_output_scales":
            length, role, layout = 2048, "scratch", "e8m0_scale_storage"
        else:
            raise ValueError(f"physical command references unknown runtime buffer {name}")
        masks = {operand.address_mask for operand in uses[name]
                 if operand.address_mask is not None}
        entries.append({
            "name": name, "position": len(entries), "role": role,
            "minimum_bytes": length, "alignment_bytes": 64, "layout": layout,
            "address_masks": sorted(masks),
            "maximum_byte_offset": max(operand.byte_offset for operand in uses[name]),
        })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlir", "bundle", "profile", "rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("mlir", "bundle", "profile", "rtl_root", "riscv_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    manifest, resources = load_bundle(args.bundle)
    program = lower_bound_source(args.mlir.read_text(), profile, manifest, resources)
    if program.mode != "spike_serial" or profile.get("transport") != "rocket_rocc":
        raise ValueError("linkable MX object currently requires Rocket RoCC serial mode")
    buffers = _referenced_buffers(program, manifest)
    if not buffers or not any(entry["role"] == "write" for entry in buffers):
        raise ValueError("linkable MX issuer has no runtime output buffer")
    names = tuple(entry["name"] for entry in buffers)
    c_source = emit_c([step.command for step in program.steps],
                      transport="rocket_rocc", buffers=names)
    args.out_dir.mkdir(parents=True)
    issuer = args.out_dir / "mx_issue.c"
    issuer.write_text(c_source)
    header = args.out_dir / "mx_issue.h"
    header.write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n\n"
        "/* Buffer order, sizes, and layouts are in object_manifest.json. */\n"
        f"void mx_issue({', '.join(f'const void *{name}' for name in names)});\n\n"
        "#endif\n")
    physical = args.out_dir / "physical_program.json"
    physical.write_text(json.dumps(program.receipt(), indent=2, sort_keys=True) + "\n")
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    nm = args.riscv_root / "bin/riscv64-unknown-elf-nm"
    readelf = args.riscv_root / "bin/riscv64-unknown-elf-readelf"
    if not cc.is_file() or not nm.is_file() or not readelf.is_file():
        parser.error("selected RISC-V toolchain lacks GCC, nm, or readelf")
    obj = args.out_dir / "mx_issue.o"
    command = [str(cc), "-std=gnu99", "-O2", "-ffreestanding", "-fno-common",
               "-mcmodel=medany", "-march=rv64gc", "-Wa,-march=rv64gc",
               "-c", str(issuer), "-o", str(obj)]
    compiled = subprocess.run(command, cwd=args.out_dir, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              check=False)
    (args.out_dir / "compile.log").write_text(compiled.stdout)
    if compiled.returncode:
        raise RuntimeError("MX issuer object build failed; see compile.log")
    defined = subprocess.check_output([str(nm), "-g", "--defined-only", str(obj)],
                                      text=True).splitlines()
    undefined = subprocess.check_output([str(nm), "-u", str(obj)],
                                        text=True).splitlines()
    if len(defined) != 1 or not defined[0].split()[-2:] == ["T", "mx_issue"] or undefined:
        raise ValueError("MX issuer object unexpectedly defines data or imports symbols")
    sections = subprocess.check_output([str(readelf), "-SW", str(obj)], text=True)
    allocated_data_bytes = 0
    for line in sections.splitlines():
        found = re.match(r"^\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+[0-9a-f]+\s+"
                         r"[0-9a-f]+\s+([0-9a-f]+)\s+", line)
        if found and found.group(1).startswith((".data", ".bss", ".rodata", ".sdata", ".sbss")):
            allocated_data_bytes += int(found.group(2), 16)
    if allocated_data_bytes:
        raise ValueError("MX issuer object embeds data despite runtime pointer ABI")
    receipt = {
        "schema": "mx_gemmini.linkable_object.v1",
        "status": "rv64_rocc_issuer_object_built",
        "transport": "rocket_rocc", "mode": program.mode,
        "shape_mnk": list(program.shape),
        "buffer_abi": buffers,
        "embedded_operand_bytes": 0,
        "embedded_golden_bytes": 0,
        "allocated_data_section_bytes": allocated_data_bytes,
        "profile_sha256": program.profile_sha256,
        "payload_manifest_sha256": program.payload_manifest_sha256,
        "bound_mlir_sha256": _sha(args.mlir),
        "source_bundle_manifest_sha256": _sha(args.bundle / "manifest.json"),
        "physical_program_sha256": _sha(physical),
        "issuer_c_sha256": _sha(issuer), "issuer_h_sha256": _sha(header),
        "object_emitter_sha256": _sha(Path(__file__)),
        "object_sha256": _sha(obj), "riscv_gcc_sha256": _sha(cc),
        "rtl_revision": _git_revision(args.rtl_root),
        "defined_symbol": "mx_issue", "undefined_symbols": [],
    }
    (args.out_dir / "object_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"linkable MX RoCC issuer: {obj}")


if __name__ == "__main__":
    main()
