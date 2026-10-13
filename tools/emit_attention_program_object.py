"""Emit one data-free RV64 object from the typed dense attention MX graph."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.nicolas_attention_io import ATTENTION_BUFFERS
from mx_gemmini_support.nicolas_attention_program import (
    lower_dense_attention_typed_program)
from mx_gemmini_support.nicolas_attention_shape import DenseAttentionShape
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _source_closure
from tools.emit_resident_pair_object import _compile_object, _file_sha, _load_mlir


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlir", "precursor-mlir", "binding-json", "profile",
                 "rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--symbol", default="mx_issue")
    args = parser.parse_args()
    for name in ("mlir", "precursor_mlir", "binding_json", "profile",
                 "rtl_root", "riscv_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("typed attention object requires Rocket RoCC transport")
    mlir_bytes = _load_mlir(args.mlir)
    precursor_bytes = _load_mlir(args.precursor_mlir)
    binding = json.loads(args.binding_json.read_text())
    buffer_bytes = DenseAttentionShape.from_plan(binding["plan"]).buffer_bytes()
    commands = lower_dense_attention_typed_program(
        mlir_bytes.decode(), precursor_bytes.decode(), profile, binding)
    args.out_dir.mkdir(parents=True)
    issuer = args.out_dir / "mx_issue.c"
    issuer.write_text(emit_c(list(commands), transport="rocket_rocc",
                             buffers=ATTENTION_BUFFERS, symbol=args.symbol))
    header = args.out_dir / "mx_issue.h"
    header.write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n\n"
        "/* Buffer order and sizes are in object_manifest.json. */\n"
        f"void {args.symbol}(" + ", ".join(f"const void *{name}"
                                     for name in ATTENTION_BUFFERS) +
        ");\n\n#endif\n")
    physical = args.out_dir / "physical_program.json"
    physical.write_text(json.dumps({
        "schema": "mx_gemmini.attention_program_physical.v1",
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": hashlib.sha256(mlir_bytes).hexdigest(),
        "precursor_mlir_sha256": hashlib.sha256(precursor_bytes).hexdigest(),
        "binding_json_sha256": _file_sha(args.binding_json),
        "buffers": ATTENTION_BUFFERS,
        "commands": [asdict(item) if isinstance(item, Command) else
                     {"kind": "fence"} for item in commands],
    }, indent=2, sort_keys=True) + "\n")
    obj, data_bytes = _compile_object(args.out_dir, args.riscv_root,
                                      symbol=args.symbol)
    outputs = {"scale_sink", "S_hw", "p_scales", "O_hw", "P_hw", "p_codes_hw"}
    manifest = {
        "schema": "mx_gemmini.attention_program_linkable_object.v1",
        "status": "rv64_rocc_typed_attention_object_built",
        "transport": "rocket_rocc",
        "buffer_abi": [
            {"name": name, "position": index,
             "role": "write" if name in outputs else "read",
             "minimum_bytes": buffer_bytes[name], "alignment_bytes": 64,
             "layout": "flat_bytes_source_attention"}
            for index, name in enumerate(ATTENTION_BUFFERS)],
        "command_count": sum(isinstance(item, Command) for item in commands),
        "fence_count": len(commands) - sum(isinstance(item, Command)
                                           for item in commands),
        "embedded_operand_bytes": 0, "embedded_golden_bytes": 0,
        "allocated_data_section_bytes": data_bytes,
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": hashlib.sha256(mlir_bytes).hexdigest(),
        "precursor_mlir_sha256": hashlib.sha256(precursor_bytes).hexdigest(),
        "mlir_container_sha256": _file_sha(args.mlir),
        "binding_json_sha256": _file_sha(args.binding_json),
        "physical_program_sha256": _file_sha(physical),
        "issuer_c_sha256": _file_sha(issuer),
        "issuer_h_sha256": _file_sha(header),
        "object_sha256": _file_sha(obj),
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "rtl_revision": _git_revision(args.rtl_root),
        "riscv_gcc_sha256": _file_sha(args.riscv_root / "bin/riscv64-unknown-elf-gcc"),
        "defined_symbol": args.symbol, "undefined_symbols": [],
    }
    (args.out_dir / "object_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"linkable typed dense attention object: {obj}")


if __name__ == "__main__":
    main()
