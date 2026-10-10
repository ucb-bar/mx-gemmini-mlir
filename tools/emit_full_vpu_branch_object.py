"""Emit a checked three-site MX→two VPU/requant/MM2 branches as RV64 RoCC.

The existing full-chain lowerer owns command scheduling. This adapter gives
its typed graph the same data-free object and manifest contract as the other
public `compile_object` families. The preloaded graph is checked as the exact
source-derived precursor of the full graph, not executed as a C replacement.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from mx_gemmini_support.command_ir import Command, Fence, emit_c
from mx_gemmini_support.full_chain_pipelined import (
    FULL_INPUTS, FULL_OUTPUTS, lower_full_chain_pipelined)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _source_closure
from tools.emit_resident_pair_object import (
    _compile_object, _file_sha, _load_mlir, _load_resource)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REFERENCE = "c1_bf16"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _buffer_abi(commands: tuple[Command | Fence, ...]) -> list[dict]:
    sizes = {name: (4096 if name.endswith(("activation", "weight")) else 128)
             for name in FULL_INPUTS}
    sizes.update({name: (8192 if name == "c1_bf16_observed" else
                         128 if "scales" in name else 4096)
                  for name in FULL_OUTPUTS})
    uses: dict[str, list] = {}
    for command in commands:
        if isinstance(command, Command):
            for operand in (command.rs1, command.rs2):
                if operand.buffer is not None:
                    uses.setdefault(operand.buffer, []).append(operand)
    if set(uses) != set(FULL_INPUTS) | set(FULL_OUTPUTS):
        raise ValueError("three-site physical buffers differ from typed ABI")
    return [{
        "name": name, "slot": name, "position": index,
        "role": "read" if name in FULL_INPUTS else "write",
        "minimum_bytes": sizes[name], "alignment_bytes": 64,
        "maximum_byte_offset": max(op.byte_offset for op in uses[name]),
        "address_masks": sorted({op.address_mask for op in uses[name]
                                 if op.address_mask is not None}),
    } for index, name in enumerate(sorted(uses))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlir", "preloaded-mlir", "profile", "rtl-root",
                 "riscv-root", "resources-dir", "abi-json", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--issue-schedule", choices=(
        "program_order_with_dependency_fences", "pipelined"),
        default="program_order_with_dependency_fences")
    parser.add_argument("--mx-opt", type=Path)
    args = parser.parse_args()
    for name in ("mlir", "preloaded_mlir", "profile", "rtl_root",
                 "riscv_root", "resources_dir", "abi_json", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("three-site MX/VPU object requires Rocket RoCC")
    spec = json.loads(args.abi_json.read_text())
    if spec != {
            "schema": "mx_gemmini.full_vpu_branch_buffer_map.v1",
            "inputs": {name: name for name in FULL_INPUTS},
            "outputs": {name: name for name in FULL_OUTPUTS},
            "source_reference": SOURCE_REFERENCE}:
        raise ValueError("three-site MX/VPU ABI differs from lowerer buffer names")
    resources = {name: _load_resource(args.resources_dir, name)
                 for name in (*FULL_INPUTS, SOURCE_REFERENCE)}
    expected = {"a1_activation": 4096, "a1_scales": 128,
                "b1_weight": 4096, "b1_scales": 128,
                "b2_weight": 4096, "b2_scales": 128,
                SOURCE_REFERENCE: 8192}
    if {name: len(data) for name, data in resources.items()} != expected:
        raise ValueError("three-site MX/VPU source input sizes differ")
    mlir_bytes = _load_mlir(args.mlir)
    preloaded_bytes = _load_mlir(args.preloaded_mlir)
    chain = lower_full_chain_pipelined(
        mlir_bytes.decode(), preloaded_bytes.decode(), profile, resources,
        issue_schedule=args.issue_schedule)
    abi = _buffer_abi(chain.commands)
    if args.mx_opt is not None:
        with tempfile.TemporaryDirectory(prefix="mx-full-branch-verify-") as temp:
            path = Path(temp) / "input.mlir"
            path.write_bytes(mlir_bytes)
            subprocess.run([str(args.mx_opt.resolve()), str(path),
                            "-o", "/dev/null"], check=True)
    args.out_dir.mkdir(parents=True)
    names = tuple(entry["name"] for entry in abi)
    issuer = args.out_dir / "mx_issue.c"
    issuer.write_text(emit_c(chain.commands, transport="rocket_rocc", buffers=names))
    header = args.out_dir / "mx_issue.h"
    header.write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n\n"
        "/* Buffer order and sizes are in object_manifest.json. */\n"
        f"void mx_issue({', '.join(f'const void *{name}' for name in names)});\n\n"
        "#endif\n")
    physical = args.out_dir / "physical_program.json"
    physical.write_text(json.dumps({
        "schema": "mx_gemmini.full_vpu_branch_physical.v1",
        "issue_schedule": args.issue_schedule,
        "sites": chain.sites,
        "profile_sha256": profile_sha256(profile),
        "commands": [({"kind": "command", **asdict(item)} if isinstance(item, Command)
                      else {"kind": "fence"}) for item in chain.commands],
    }, indent=2, sort_keys=True) + "\n")
    obj, data_bytes = _compile_object(args.out_dir, args.riscv_root)
    manifest = {
        "schema": "mx_gemmini.full_vpu_branch_linkable_object.v1",
        "status": "rv64_rocc_full_vpu_branch_object_built",
        "transport": "rocket_rocc",
        "issue_schedule": args.issue_schedule,
        "sites": chain.sites,
        "buffer_abi": abi,
        "embedded_operand_bytes": 0, "embedded_golden_bytes": 0,
        "allocated_data_section_bytes": data_bytes,
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": _sha(mlir_bytes),
        "preloaded_mlir_sha256": _sha(preloaded_bytes),
        "mlir_container_sha256": _file_sha(args.mlir),
        "preloaded_container_sha256": _file_sha(args.preloaded_mlir),
        "abi_json_sha256": _file_sha(args.abi_json),
        "input_sha256": {name: _sha(resources[name]) for name in FULL_INPUTS},
        "source_reference_sha256": _sha(resources[SOURCE_REFERENCE]),
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
    print(f"linkable three-site MX/VPU branch: {obj}")


if __name__ == "__main__":
    main()
