"""Emit a data-free RV64 object for a verified straight-line VPU sequence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir
from mx_gemmini_support.vpu_sequence_program import (TIMELINE_ABI_SCHEMA,
                                                      lower_vpu_sequence)
from tools.compile_mx import _git_revision, _source_closure
from tools.emit_resident_pair_object import _compile_object, _file_sha, _load_mlir


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlir", "profile", "rtl-root", "riscv-root", "abi-json", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("mlir", "profile", "rtl_root", "riscv_root", "abi_json", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("VPU sequence object requires Rocket RoCC transport")
    mlir_bytes = _load_mlir(args.mlir)
    report = verify_ir(mlir_bytes.decode(), profile)
    if ((report["contracts"], report["resident_contracts"],
         report["spad_requants"]) != (0, 0, 0) or
            not 2 <= report["vpu_commands"] <= 32):
        raise ValueError("VPU sequence object needs only 2..32 VPU operations")
    spec = json.loads(args.abi_json.read_text())
    plan = lower_vpu_sequence(mlir_bytes.decode(), profile, spec)
    buffers = (*plan.inputs, *plan.outputs)
    args.out_dir.mkdir(parents=True)
    issuer = args.out_dir / "mx_issue.c"
    issuer.write_text(emit_c(list(plan.commands), transport="rocket_rocc",
                             buffers=tuple(item.name for item in buffers)))
    header = args.out_dir / "mx_issue.h"
    header.write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n\n"
        "/* Buffer order and sizes are in object_manifest.json. */\n"
        "void mx_issue(" +
        ", ".join(f"const void *{item.name}" for item in buffers) +
        ");\n\n#endif\n")
    physical = args.out_dir / "physical_program.json"
    physical_data = {
        "schema": "mx_gemmini.vpu_sequence_physical.v1",
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": hashlib.sha256(mlir_bytes).hexdigest(),
        "operations": plan.operations,
        "inputs": [asdict(item) for item in plan.inputs],
        "outputs": [asdict(item) for item in plan.outputs],
        "commands": [asdict(command) if isinstance(command, Command) else
                     {"kind": "fence"} for command in plan.commands],
    }
    if plan.abi_schema == TIMELINE_ABI_SCHEMA:
        physical_data.update(
            buffer_map_schema=plan.abi_schema,
            captures=[asdict(item) for item in plan.captures],
            reloads=[asdict(item) for item in plan.reloads])
    physical.write_text(json.dumps(physical_data, indent=2, sort_keys=True) + "\n")
    obj, data_bytes = _compile_object(args.out_dir, args.riscv_root)
    abi = [{"name": item.name, "position": index,
            "role": "read" if index < len(plan.inputs) else "write",
            "minimum_bytes": item.rows * 16, "alignment_bytes": 64,
            "layout": "row_major_bf16_lanes8"}
           for index, item in enumerate(buffers)]
    manifest = {
        "schema": "mx_gemmini.vpu_sequence_linkable_object.v1",
        "status": "rv64_rocc_vpu_sequence_object_built",
        "transport": "rocket_rocc", "operation_count": len(plan.operations),
        "buffer_abi": abi, "command_count": len(plan.commands),
        "embedded_operand_bytes": 0, "embedded_golden_bytes": 0,
        "allocated_data_section_bytes": data_bytes,
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": hashlib.sha256(mlir_bytes).hexdigest(),
        "mlir_container_sha256": _file_sha(args.mlir),
        "abi_json_sha256": _file_sha(args.abi_json),
        "physical_program_sha256": _file_sha(physical),
        "issuer_c_sha256": _file_sha(issuer),
        "issuer_h_sha256": _file_sha(header),
        "object_sha256": _file_sha(obj),
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "rtl_revision": _git_revision(args.rtl_root),
        "riscv_gcc_sha256": _file_sha(
            args.riscv_root / "bin/riscv64-unknown-elf-gcc"),
        "defined_symbol": "mx_issue", "undefined_symbols": [],
    }
    if plan.abi_schema == TIMELINE_ABI_SCHEMA:
        manifest["buffer_map_schema"] = plan.abi_schema
    (args.out_dir / "object_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"linkable VPU sequence: {obj}")


if __name__ == "__main__":
    main()
