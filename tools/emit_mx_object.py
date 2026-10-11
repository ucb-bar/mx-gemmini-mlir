"""Emit a linkable RV64 RoCC MX issuer with explicit runtime buffer pointers.

The typed, payload-bound MLIR and checked bundle specialize the physical
schedule. The caller supplies every input, scale, scratch, and output pointer
at invocation time. Radiance host requantization follows MX command issue in
the same entry point. The bundle's operand bytes and golden are never linked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.host_requant_object import emit_composed_c
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
    host_requant = program.output_format in {"radiance_header_fp8",
                                             "radiance_header_fp6"}
    if host_requant:
        uses.setdefault("output_quantized", [])
        uses.setdefault("scratch_output_scales", [])
        if program.output_format == "radiance_header_fp6":
            uses.setdefault("output_lut", [])
    entries = []
    for name in sorted(uses):
        if name in manifest["resources"]:
            descriptor = manifest["resources"][name]
            length = descriptor["bytes"]
            role = "read"
            layout = descriptor["layout"]
        elif name == "output_bf16":
            output_tiles = program.plan.get("output_tiles", [])
            length, role = m * n * 2, "write"
            layout = ("output_tile_major_bf16" if len(output_tiles) > 1 and
                      program.plan.get("bf16_output_layout") != "row_major_bf16"
                      else "row_major_bf16")
        elif name == "output_quantized":
            packed_lut = (program.plan.get("quant_output_layout") ==
                          "packed_even_odd_m_lut_indices")
            length = m * n // (2 if packed_lut or program.output_format in {
                "fp4_e2m1", "fp6_e3m2", "radiance_header_fp6"} else 1)
            role = "write"
            layout = ("packed_even_odd_m_lut_indices" if packed_lut else
                      "pair_major_nibble_codes" if program.output_format ==
                      "radiance_header_fp6" else "tiled_quantized" if
                      program.tiled_quant_readout else "row_major_codes")
        elif name == "scratch_output_scales":
            packed_lut = (program.plan.get("quant_output_layout") ==
                          "packed_even_odd_m_lut_indices")
            length = m * n // 32 if host_requant or packed_lut else 2048
            role, layout = ("write" if host_requant or packed_lut else "scratch"), "e8m0_scale_storage"
        else:
            raise ValueError(f"physical command references unknown runtime buffer {name}")
        masks = {operand.address_mask for operand in uses[name]
                 if operand.address_mask is not None}
        entries.append({
            "name": name, "position": len(entries), "role": role,
            "minimum_bytes": length, "alignment_bytes": 64, "layout": layout,
            "address_masks": sorted(masks),
            "maximum_byte_offset": max((operand.byte_offset for operand in uses[name]),
                                       default=0),
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
    if program.mode not in {"spike_serial", "rtl_accumulator"} or \
            profile.get("transport") != "rocket_rocc":
        raise ValueError("linkable MX object requires a supported Rocket RoCC mode")
    buffers = _referenced_buffers(program, manifest)
    if not buffers or not any(entry["role"] == "write" for entry in buffers):
        raise ValueError("linkable MX issuer has no runtime output buffer")
    names = tuple(entry["name"] for entry in buffers)
    host_requant = program.output_format in {"radiance_header_fp8",
                                             "radiance_header_fp6"}
    commands_c = emit_c(
        [step.command for step in program.steps], transport="rocket_rocc",
        buffers=names, symbol="mx_issue_commands" if host_requant else "mx_issue",
        internal=host_requant)
    c_source = (emit_composed_c(commands_c, program.output_format, names,
                                *program.shape[:2]) if host_requant else commands_c)
    args.out_dir.mkdir(parents=True)
    issuer = args.out_dir / "mx_issue.c"
    issuer.write_text(c_source)
    header = args.out_dir / "mx_issue.h"
    signature = ", ".join(
        f"{'void' if host_requant and name in {'output_bf16', 'output_quantized', 'scratch_output_scales'} else 'const void'} *{name}"
        for name in names)
    header.write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n\n"
        "/* Buffer order, sizes, and layouts are in object_manifest.json. */\n"
        f"void mx_issue({signature});\n\n"
        "#endif\n")
    physical = args.out_dir / "physical_program.json"
    physical.write_text(json.dumps(program.receipt(), indent=2, sort_keys=True) + "\n")
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    nm = args.riscv_root / "bin/riscv64-unknown-elf-nm"
    readelf = args.riscv_root / "bin/riscv64-unknown-elf-readelf"
    if not cc.is_file() or not nm.is_file() or not readelf.is_file():
        parser.error("selected RISC-V toolchain lacks GCC, nm, or readelf")
    obj = args.out_dir / "mx_issue.o"
    # The command stream is already ordered. O1 keeps large issuers' constants
    # in instructions; O0 emits a .rodata pool, while O2 spends minutes here.
    issuer_opt_level = "-O1" if len(program.steps) > 10000 else "-O2"
    command = [str(cc), "-std=gnu99", issuer_opt_level, "-ffreestanding", "-fno-common",
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
    if issuer_opt_level != "-O2":
        receipt["issuer_opt_level"] = issuer_opt_level
    if host_requant:
        receipt["host_output_format"] = program.output_format
        receipt["status"] = "rv64_rocc_composed_object_built"
    if program.mode == "rtl_accumulator":
        receipt["execution_scope"] = "hardware_accumulator_commands_only"
        receipt["hardware_numerical_qualification"] = "unqualified"
        receipt["stock_spike_supported"] = False
    (args.out_dir / "object_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"linkable MX RoCC issuer: {obj}")


if __name__ == "__main__":
    main()
