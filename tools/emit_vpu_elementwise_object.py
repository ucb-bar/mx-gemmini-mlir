"""Emit a data-free RV64 object for one typed MX VPU operation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from mx_gemmini_support.command_ir import emit_c
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir
from mx_gemmini_support.vpu_elementwise_program import lower_vpu_elementwise_program
from tools.compile_mx import _git_revision, _source_closure
from tools.emit_resident_pair_object import _compile_object, _file_sha, _load_mlir


ROOT = Path(__file__).resolve().parents[1]
_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


def _buffers(path: Path) -> dict[str, str]:
    spec = json.loads(path.read_text())
    if (not isinstance(spec, dict) or set(spec) != {"schema", "inputs", "outputs"} or
            spec["schema"] != "mx_gemmini.vpu_elementwise_buffer_map.v1" or
            not isinstance(spec["inputs"], dict) or
            set(spec["inputs"]) != {"src1", "src2"} or
            not isinstance(spec["outputs"], dict) or
            set(spec["outputs"]) not in ({"output"}, {"output", "output2"})):
        raise ValueError("VPU elementwise ABI JSON has unsupported slots")
    buffers = spec["inputs"] | spec["outputs"]
    if (any(not isinstance(name, str) or not _SYMBOL.fullmatch(name)
            for name in buffers.values()) or
            len(set(buffers.values())) != len(buffers)):
        raise ValueError("VPU elementwise ABI requires distinct C identifiers")
    return buffers


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
        raise ValueError("VPU elementwise object requires Rocket RoCC transport")
    mlir_bytes = _load_mlir(args.mlir)
    mlir = mlir_bytes.decode()
    report = verify_ir(mlir, profile)
    if ((report["contracts"], report["resident_contracts"],
         report["vpu_commands"], report["spad_requants"]) != (0, 0, 1, 0)):
        raise ValueError("VPU elementwise object needs one typed VPU operation")
    buffers = _buffers(args.abi_json)
    plan = lower_vpu_elementwise_program(mlir, profile, buffers)
    args.out_dir.mkdir(parents=True)
    order = ("src1", "src2", "output") + (
        ("output2",) if plan.second_dst_row is not None else ())
    names = tuple(buffers[slot] for slot in order)
    issuer = args.out_dir / "mx_issue.c"
    issuer.write_text(emit_c(list(plan.commands), transport="rocket_rocc",
                             buffers=names))
    header = args.out_dir / "mx_issue.h"
    header.write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n\n"
        "/* Buffer order and sizes are in object_manifest.json. */\n"
        "void mx_issue(" + ", ".join(f"const void *{name}" for name in names) +
        ");\n\n#endif\n")
    physical = args.out_dir / "physical_program.json"
    physical.write_text(json.dumps({
        "schema": "mx_gemmini.vpu_elementwise_physical.v1",
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": hashlib.sha256(mlir_bytes).hexdigest(),
        "plan": {key: value for key, value in asdict(plan).items()
                 if key != "commands"},
        "commands": [asdict(command) for command in plan.commands],
    }, indent=2, sort_keys=True) + "\n")
    obj, data_bytes = _compile_object(args.out_dir, args.riscv_root)
    minimum = {"src1": plan.rows * 16, "src2": plan.src2_rows * 16,
               "output": plan.output_rows * 16}
    if plan.second_dst_row is not None:
        minimum["output2"] = plan.rows // plan.reduction_length * 16
    abi = [{"name": buffers[slot], "slot": slot, "position": position,
            "role": "unused" if slot == "src2" and not plan.src2_rows else
                    "write" if slot.startswith("output") else "read",
            "minimum_bytes": minimum[slot], "alignment_bytes": 64,
            "layout": "row_major_bf16_lanes8"}
           for position, slot in enumerate(order)]
    manifest = {
        "schema": "mx_gemmini.vpu_elementwise_linkable_object.v1",
        "status": "rv64_rocc_vpu_elementwise_object_built",
        "transport": "rocket_rocc", "vpu_kind": plan.kind,
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
    (args.out_dir / "object_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"linkable VPU {plan.kind}: {obj}")


if __name__ == "__main__":
    main()
