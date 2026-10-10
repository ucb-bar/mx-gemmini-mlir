"""Emit a data-free RV64 object from a checked asymmetric MX source graph.

The source recipe, C program (or pinned generator), and header are inputs to
the existing asymmetric physical lowerer. They are not linked into this object.
Runtime pointers supply every operand, scale, LUT, scratch area, and output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.asymmetric_specialization import lower_asymmetric_physical
from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.source_payload import manifest_sha256
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _source_closure
from tools.emit_mx_object import _referenced_buffers
from tools.emit_resident_pair_object import _compile_object, _file_sha, _load_mlir


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlir", "profile", "rtl-root", "riscv-root", "out-dir",
                 "recipe-json", "source-driver", "source-header"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("mlir", "profile", "rtl_root", "riscv_root", "out_dir",
                 "recipe_json", "source_driver", "source_header"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("asymmetric object requires Rocket RoCC transport")
    mlir_bytes = _load_mlir(args.mlir)
    recipe = json.loads(args.recipe_json.read_text())
    program, resources, resource_manifest = lower_asymmetric_physical(
        mlir_bytes.decode(), profile, recipe,
        source=args.source_driver, header=args.source_header)
    if (program.mode != "spike_serial" or
            program.profile_sha256 != profile_sha256(profile) or
            program.payload_manifest_sha256 != manifest_sha256(resource_manifest)):
        raise ValueError("asymmetric physical program differs from checked profile or resources")
    buffers = _referenced_buffers(program, resource_manifest)
    if not buffers or not any(slot["role"] == "write" for slot in buffers):
        raise ValueError("asymmetric object has no runtime output")
    names = tuple(slot["name"] for slot in buffers)
    commands = [step.command for step in program.steps]
    args.out_dir.mkdir(parents=True)
    issuer = args.out_dir / "mx_issue.c"
    issuer.write_text(emit_c(commands, transport="rocket_rocc", buffers=names))
    header = args.out_dir / "mx_issue.h"
    header.write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n\n"
        "/* Buffer order, sizes, and layouts are in object_manifest.json. */\n"
        f"void mx_issue({', '.join(f'const void *{name}' for name in names)});\n\n"
        "#endif\n")
    physical = args.out_dir / "physical_program.json"
    physical.write_text(json.dumps(program.receipt(), indent=2, sort_keys=True) + "\n")
    obj, data_bytes = _compile_object(args.out_dir, args.riscv_root)
    manifest = {
        "schema": "mx_gemmini.asymmetric_source_linkable_object.v1",
        "status": "rv64_rocc_asymmetric_object_built",
        "transport": "rocket_rocc", "mode": program.mode,
        "shape_mnk": list(program.shape), "compute": recipe["compute"],
        "buffer_abi": buffers,
        "command_count": sum(isinstance(item, Command) for item in commands),
        "fence_count": len(commands) - sum(isinstance(item, Command) for item in commands),
        "embedded_operand_bytes": 0, "embedded_golden_bytes": 0,
        "allocated_data_section_bytes": data_bytes,
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": hashlib.sha256(mlir_bytes).hexdigest(),
        "mlir_container_sha256": _file_sha(args.mlir),
        "recipe_sha256": _file_sha(args.recipe_json),
        "source_driver_sha256": _file_sha(args.source_driver),
        "source_header_sha256": _file_sha(args.source_header),
        "resource_manifest_sha256": manifest_sha256(resource_manifest),
        "resource_sha256": {name: hashlib.sha256(data).hexdigest()
                            for name, data in sorted(resources.items())},
        "physical_program_sha256": _file_sha(physical),
        "issuer_c_sha256": _file_sha(issuer), "issuer_h_sha256": _file_sha(header),
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
    print(f"linkable asymmetric MX object: {obj}")


if __name__ == "__main__":
    main()
