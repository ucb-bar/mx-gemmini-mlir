"""Compile the typed Nicolas BF16 VPU softmax to a data-free RV64 object."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.source_softmax import lower_softmax_program_from_ir
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_mx import _git_revision, _source_closure
from tools.emit_resident_pair_object import _compile_object, _file_sha, _load_mlir


ROOT = Path(__file__).resolve().parents[1]
_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


def _buffer_names(path: Path) -> tuple[str, str]:
    spec = json.loads(path.read_text())
    if (not isinstance(spec, dict) or set(spec) != {"schema", "inputs", "outputs"} or
            spec["schema"] != "mx_gemmini.vpu_softmax_buffer_map.v1" or
            not isinstance(spec["inputs"], dict) or set(spec["inputs"]) != {"score"} or
            not isinstance(spec["outputs"], dict) or set(spec["outputs"]) != {"output"}):
        raise ValueError("VPU softmax ABI JSON has an unsupported schema or slots")
    score, output = spec["inputs"]["score"], spec["outputs"]["output"]
    if (not all(isinstance(name, str) and _SYMBOL.fullmatch(name)
                for name in (score, output)) or score == output):
        raise ValueError("VPU softmax ABI needs two distinct C identifier symbols")
    return score, output


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
        raise ValueError("VPU softmax object requires Rocket RoCC transport")
    score, output = _buffer_names(args.abi_json)
    mlir_bytes = _load_mlir(args.mlir)
    mlir = mlir_bytes.decode()
    report = verify_ir(mlir, profile)
    if ((report["contracts"], report["resident_contracts"], report["vpu_commands"],
         report["spad_requants"]) != (0, 0, 6, 0)):
        raise ValueError("VPU softmax object needs exactly six typed VPU operations")
    commands = lower_softmax_program_from_ir(
        mlir, profile, score_buffer=score, output_buffer=output)
    if [command.funct for command in commands] != [7, 0, 0, *([2] * 4),
                                                    *([33] * 6), *([3] * 4)]:
        raise ValueError("VPU softmax physical schedule differs")
    args.out_dir.mkdir(parents=True)
    issuer = args.out_dir / "mx_issue.c"
    issuer.write_text(emit_c(commands, transport="rocket_rocc",
                             buffers=(score, output)))
    header = args.out_dir / "mx_issue.h"
    header.write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n\n"
        "/* Buffer order and sizes are in object_manifest.json. */\n"
        f"void mx_issue(const void *{score}, const void *{output});\n\n"
        "#endif\n")
    physical = args.out_dir / "physical_program.json"
    physical.write_text(json.dumps({
        "schema": "mx_gemmini.vpu_softmax_physical.v1",
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": hashlib.sha256(mlir_bytes).hexdigest(),
        "logical_shape": [16, 32],
        "commands": [{"kind": "command", **asdict(command)} for command in commands],
    }, indent=2, sort_keys=True) + "\n")
    obj, data_bytes = _compile_object(args.out_dir, args.riscv_root)
    manifest = {
        "schema": "mx_gemmini.vpu_softmax_linkable_object.v1",
        "status": "rv64_rocc_vpu_softmax_object_built",
        "transport": "rocket_rocc",
        "buffer_abi": [
            {"name": score, "slot": "score", "position": 0, "role": "read",
             "minimum_bytes": 1024, "alignment_bytes": 64, "layout": "row_major_bf16_16x32"},
            {"name": output, "slot": "output", "position": 1, "role": "write",
             "minimum_bytes": 1024, "alignment_bytes": 64, "layout": "row_major_bf16_16x32"},
        ],
        "command_count": len(commands),
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
        "object_emitter_sha256": _file_sha(Path(__file__)),
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "rtl_revision": _git_revision(args.rtl_root),
        "riscv_gcc_sha256": _file_sha(args.riscv_root / "bin/riscv64-unknown-elf-gcc"),
        "defined_symbol": "mx_issue", "undefined_symbols": [],
    }
    (args.out_dir / "object_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"linkable VPU softmax: {obj}")


if __name__ == "__main__":
    main()
