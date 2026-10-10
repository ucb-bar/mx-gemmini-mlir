"""Compile a typed MX MM1→scalar VPU chain→resident MM2 to an RV64 object."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from mx_gemmini_support.command_ir import Command, Fence, emit_c
from mx_gemmini_support.physical_program import _config_st, _transfer
from mx_gemmini_support.resident_vpu_graph import (INPUTS, OUTPUTS,
                                                   lower_connected_fp8_vpu_pair)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _source_closure
from tools.emit_resident_pair_object import (_compile_object, _file_sha,
                                             _load_mlir, _load_resource)


ROOT = Path(__file__).resolve().parents[1]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _readout_commands(pair, outputs: dict[str, str]) -> tuple[Command | Fence, ...]:
    """Read both resident quantized tiles after MM2, as the source does."""
    commands: list[Command | Fence] = [*pair.commands, _config_st(16)]
    for slot, start, rows in (("c1_tiled", pair.c1_row,
                               pair.first_width * pair.first_width // 16),
                              ("c2_tiled", pair.c2_row,
                               pair.first_width * pair.second_width // 16)):
        for row in range(0, rows, 16):
            commands.append(_transfer(3, outputs[slot], row * 16, start + row))
    commands.append(Fence())
    return tuple(commands)


def _buffer_abi(commands: tuple[Command | Fence, ...], inputs: dict[str, str],
                outputs: dict[str, str], second_width: int = 64,
                first_width: int = 64) -> list[dict]:
    slots = {
        "a1_activation": (first_width ** 2, "read", "row_major_fp8"),
        "a1_scales": (first_width ** 2 // 32, "read", "k_group_major_a_scales"),
        "b1_weight": (first_width ** 2, "read", "row_major_fp8"),
        "b1_scales": (first_width ** 2 // 32, "read", "k_group_major_b_scales"),
        "b2_weight": (first_width * second_width, "read", "row_major_fp8"),
        "b2_scales": (first_width * second_width // 32, "read", "k_group_major_b_scales"),
        "c1_scales": (first_width ** 2 // 32, "write", "row_major_e8m0_scales"),
        "c1_bf16_observed": (first_width ** 2 * 2, "write", "output_tile_major_bf16"),
        "c1_tiled": (first_width ** 2, "write", "tile_major_fp8"),
        "c2_scales": (first_width * second_width // 32, "write", "row_major_e8m0_scales"),
        "c2_tiled": (first_width * second_width, "write", "tile_major_fp8"),
    }
    symbols = inputs | outputs
    reverse = {symbol: slot for slot, symbol in symbols.items()}
    uses: dict[str, list] = {}
    for command in commands:
        if isinstance(command, Command):
            for operand in (command.rs1, command.rs2):
                if operand.buffer is not None:
                    uses.setdefault(operand.buffer, []).append(operand)
    if set(uses) != set(symbols.values()):
        raise ValueError("connected MX VPU command buffers differ from ABI")
    entries = []
    for name in sorted(uses):
        slot = reverse[name]
        size, role, layout = slots[slot]
        operands = uses[name]
        entries.append({
            "name": name, "slot": slot, "position": len(entries),
            "role": role, "minimum_bytes": size, "alignment_bytes": 64,
            "layout": layout,
            "address_masks": sorted({operand.address_mask for operand in operands
                                     if operand.address_mask is not None}),
            "maximum_byte_offset": max(operand.byte_offset for operand in operands),
        })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlir", "profile", "rtl-root", "riscv-root", "resources-dir",
                 "abi-json", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--mx-opt", type=Path)
    args = parser.parse_args()
    for name in ("mlir", "profile", "rtl_root", "riscv_root", "resources_dir",
                 "abi_json", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("connected MX VPU object requires Rocket RoCC transport")
    spec = json.loads(args.abi_json.read_text())
    if (not isinstance(spec, dict) or set(spec) != {"schema", "inputs", "outputs"} or
            spec["schema"] != "mx_gemmini.resident_vpu_buffer_map.v1" or
            not isinstance(spec["inputs"], dict) or set(spec["inputs"]) != set(INPUTS) or
            not isinstance(spec["outputs"], dict) or set(spec["outputs"]) != set(OUTPUTS)):
        raise ValueError("connected MX VPU ABI JSON has unsupported slots")
    inputs, outputs = spec["inputs"], spec["outputs"]
    resources = {inputs[slot]: _load_resource(args.resources_dir, inputs[slot])
                 for slot in INPUTS}
    mlir_bytes = _load_mlir(args.mlir)
    pair = lower_connected_fp8_vpu_pair(
        mlir_bytes.decode(), profile, resources, buffers=inputs, outputs=outputs)
    commands = _readout_commands(pair, outputs)
    abi = _buffer_abi(commands, inputs, outputs, pair.second_width,
                      pair.first_width)
    if args.mx_opt is not None:
        with tempfile.TemporaryDirectory(prefix="mx-vpu-verifier-") as temp:
            native_input = Path(temp) / "input.mlir"
            native_input.write_bytes(mlir_bytes)
            subprocess.run([str(args.mx_opt.resolve()), str(native_input),
                            "-o", "/dev/null"], check=True)
    args.out_dir.mkdir(parents=True)
    names = tuple(entry["name"] for entry in abi)
    issuer = args.out_dir / "mx_issue.c"
    issuer.write_text(emit_c(commands, transport="rocket_rocc", buffers=names))
    header = args.out_dir / "mx_issue.h"
    header.write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n\n"
        "/* Buffer order, sizes, and layouts are in object_manifest.json. */\n"
        f"void mx_issue({', '.join(f'const void *{name}' for name in names)});\n\n"
        "#endif\n")
    physical = args.out_dir / "physical_program.json"
    physical.write_text(json.dumps({
        "schema": "mx_gemmini.resident_vpu_physical.v1",
        "shape_mnk": [pair.first_width, pair.second_width, pair.first_width],
        "first_site": pair.first_site, "second_site": pair.second_site,
        "profile_sha256": profile_sha256(profile),
        "commands": [({"kind": "command", **asdict(item)} if isinstance(item, Command)
                      else {"kind": "fence"}) for item in commands],
    }, indent=2, sort_keys=True) + "\n")
    obj, data_bytes = _compile_object(args.out_dir, args.riscv_root)
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    manifest = {
        "schema": "mx_gemmini.resident_vpu_linkable_object.v1",
        "status": "rv64_rocc_resident_vpu_object_built",
        "transport": "rocket_rocc",
        "shape_mnk": [pair.first_width, pair.second_width, pair.first_width],
        "first_site": pair.first_site, "second_site": pair.second_site,
        "buffer_abi": abi,
        "embedded_operand_bytes": 0, "embedded_golden_bytes": 0,
        "allocated_data_section_bytes": data_bytes,
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": _sha(mlir_bytes),
        "mlir_container_sha256": _file_sha(args.mlir),
        "abi_json_sha256": _file_sha(args.abi_json),
        "input_sha256": {slot: _sha(resources[inputs[slot]]) for slot in INPUTS},
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
        "riscv_gcc_sha256": _file_sha(cc),
        "defined_symbol": "mx_issue", "undefined_symbols": [],
    }
    (args.out_dir / "object_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"linkable resident MX VPU pair: {obj}")


if __name__ == "__main__":
    main()
