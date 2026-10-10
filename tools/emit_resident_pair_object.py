"""Compile a typed FP8 resident MX pair to a data-free RV64 RoCC object.

The ABI JSON maps the six logical MLIR arguments and four output/scratch
buffers to runtime C symbols. Input .bin or .bin.gz files are used only to
check the MLIR payload digest and physical sizes; they are not linked into
the object. The caller owns allocation, alignment, and source goldens.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

from mx_gemmini_support.command_ir import Command, Fence, emit_c
from mx_gemmini_support.resident_pair_graph import INPUTS, lower_connected_pair
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _source_closure


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ("c1_scales", "c1_tiled_observed", "c2_scales", "c2_tiled")
_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _load_resource(directory: Path, symbol: str) -> bytes:
    plain = directory / f"{symbol}.bin"
    packed = directory / f"{symbol}.bin.gz"
    if plain.is_file() == packed.is_file():
        raise ValueError(f"runtime input {symbol} needs exactly one .bin or .bin.gz file")
    return plain.read_bytes() if plain.is_file() else gzip.decompress(packed.read_bytes())


def _load_mlir(path: Path) -> bytes:
    data = path.read_bytes()
    if path.name.endswith(".mlir.gz"):
        return gzip.decompress(data)
    if path.suffix != ".mlir":
        raise ValueError("MX object input must be .mlir or .mlir.gz")
    return data


def _buffer_abi(pair, buffers: dict[str, str], outputs: dict[str, str], *,
                precision: str = "fp8_e4m3") -> list[dict]:
    m, n, k = pair.plan.m, pair.plan.n, pair.plan.k
    first_k = getattr(pair.plan, "first_k", k)
    first_n = k if hasattr(pair.plan, "first_k") else n
    pack = 2 if precision == "fp4_e2m1" else 1
    a_layout = "operand_a_tiled_packed_fp4" if pack == 2 else "row_major_fp8"
    b_layout = "row_major_n_packed_fp4" if pack == 2 else "row_major_fp8"
    tile_layout = "tile_major_packed_fp4" if pack == 2 else "tile_major_fp8"
    slots = {
        "a1_activation": (m * first_k // pack, "read", a_layout),
        "a1_scales": (m * first_k // 32, "read", "k_group_major_a_scales"),
        "b1_weight": (first_k * first_n // pack, "read", b_layout),
        "b1_scales": (first_k * first_n // 32, "read", "k_group_major_b_scales"),
        "b2_weight": (k * n // pack, "read", b_layout),
        "b2_scales": (k * n // 32, "read", "k_group_major_b_scales"),
        "c1_scales": (m * first_n // 32, "write", "row_major_e8m0_scales"),
        "c1_tiled_observed": (m * first_n // pack, "write", tile_layout),
        "c2_scales": (m * n // 32, "write", "row_major_e8m0_scales"),
        "c2_tiled": (m * n // pack, "write", tile_layout),
    }
    symbols = {slot: buffers[slot] for slot in INPUTS} | {
        slot: outputs[slot] for slot in OUTPUTS}
    uses: dict[str, list] = {}
    for command in pair.commands:
        if isinstance(command, Command):
            for operand in (command.rs1, command.rs2):
                if operand.buffer is not None:
                    uses.setdefault(operand.buffer, []).append(operand)
    if set(uses) != set(symbols.values()):
        raise ValueError("resident pair command buffers differ from declared ABI")
    reverse = {symbol: slot for slot, symbol in symbols.items()}
    entries = []
    for name in sorted(uses):
        slot = reverse[name]
        length, role, layout = slots[slot]
        operands = uses[name]
        entries.append({
            "name": name, "slot": slot, "position": len(entries),
            "role": role, "minimum_bytes": length,
            "alignment_bytes": 64, "layout": layout,
            "address_masks": sorted({operand.address_mask for operand in operands
                                      if operand.address_mask is not None}),
            "maximum_byte_offset": max(operand.byte_offset for operand in operands),
        })
    return entries


def _compile_object(directory: Path, riscv_root: Path) -> tuple[Path, int]:
    cc = riscv_root / "bin/riscv64-unknown-elf-gcc"
    nm = riscv_root / "bin/riscv64-unknown-elf-nm"
    readelf = riscv_root / "bin/riscv64-unknown-elf-readelf"
    if not all(path.is_file() for path in (cc, nm, readelf)):
        raise ValueError("selected RISC-V toolchain lacks GCC, nm, or readelf")
    obj = directory / "mx_issue.o"
    command = [str(cc), "-std=gnu99", "-O2", "-ffreestanding", "-fno-common",
               "-mcmodel=medany", "-march=rv64gc", "-Wa,-march=rv64gc",
               "-c", "mx_issue.c", "-o", "mx_issue.o"]
    result = subprocess.run(command, cwd=directory, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            check=False)
    (directory / "compile.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError("MX issuer object build failed; see compile.log")
    defined = subprocess.check_output([str(nm), "-g", "--defined-only", str(obj)],
                                      text=True).splitlines()
    undefined = subprocess.check_output([str(nm), "-u", str(obj)],
                                        text=True).splitlines()
    if len(defined) != 1 or defined[0].split()[-2:] != ["T", "mx_issue"] or undefined:
        raise ValueError("MX issuer object has unexpected symbols")
    sections = subprocess.check_output([str(readelf), "-SW", str(obj)], text=True)
    allocated_data_bytes = 0
    for line in sections.splitlines():
        found = re.match(r"^\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+[0-9a-f]+\s+"
                         r"[0-9a-f]+\s+([0-9a-f]+)\s+", line)
        if found and found.group(1).startswith((".data", ".bss", ".rodata",
                                                 ".sdata", ".sbss")):
            allocated_data_bytes += int(found.group(2), 16)
    if allocated_data_bytes:
        raise ValueError("MX issuer object embeds runtime or golden data")
    return obj, allocated_data_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlir", "profile", "rtl-root", "riscv-root", "resources-dir",
                 "abi-json", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--mx-opt", type=Path,
                        help="also verify the typed module with native mx-gemmini-opt")
    parser.add_argument("--precision", choices=("fp8_e4m3", "fp4_e2m1"),
                        default="fp8_e4m3")
    parser.add_argument("--baseline-manifest", type=Path,
                        help="require identical generated issuer, object, and physical program")
    args = parser.parse_args()
    for name in ("mlir", "profile", "rtl_root", "riscv_root", "resources_dir",
                 "abi_json", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("resident pair object requires Rocket RoCC transport")
    spec = json.loads(args.abi_json.read_text())
    if (not isinstance(spec, dict) or
            set(spec) != {"schema", "inputs", "outputs"} or
            spec["schema"] != "mx_gemmini.resident_pair_buffer_map.v1" or
            not isinstance(spec["inputs"], dict) or
            set(spec["inputs"]) != set(INPUTS) or
            not isinstance(spec["outputs"], dict) or
            set(spec["outputs"]) != set(OUTPUTS)):
        raise ValueError("resident pair ABI JSON has an unsupported schema or slots")
    buffers, outputs = spec["inputs"], spec["outputs"]
    if (any(not isinstance(name, str) or not _SYMBOL.fullmatch(name)
            for name in (*buffers.values(), *outputs.values())) or
            len(set((*buffers.values(), *outputs.values()))) != len(INPUTS) + len(OUTPUTS)):
        raise ValueError("resident pair ABI needs distinct C identifier symbols")
    resources = {buffers[slot]: _load_resource(args.resources_dir, buffers[slot])
                 for slot in INPUTS}
    mlir_bytes = _load_mlir(args.mlir)
    mlir_text = mlir_bytes.decode()
    pair = lower_connected_pair(
        mlir_text, profile, resources, buffers=buffers,
        c1_scales=outputs["c1_scales"],
        c1_tiled_observed=outputs["c1_tiled_observed"],
        c2_tiled=outputs["c2_tiled"], precision=args.precision)
    # The typed MM2 operation must carry the same scale destination as the ABI.
    used_outputs = {operand.buffer for command in pair.commands
                    if isinstance(command, Command)
                    for operand in (command.rs1, command.rs2)
                    if operand.buffer is not None and
                    operand.buffer not in buffers.values()}
    if used_outputs != set(outputs.values()):
        raise ValueError("resident pair output ABI differs from typed MLIR")
    scale_destinations = [command.rs1.buffer for command in pair.commands
                          if isinstance(command, Command) and command.funct == 26]
    if scale_destinations != [outputs["c1_scales"], outputs["c2_scales"]]:
        raise ValueError("resident pair scale output slots differ from typed MLIR")
    buffer_abi = _buffer_abi(pair, buffers, outputs, precision=args.precision)
    if args.mx_opt is not None:
        if args.mlir.name.endswith(".mlir.gz"):
            with tempfile.TemporaryDirectory(prefix="mx-pair-verifier-") as temp:
                native_input = Path(temp) / "input.mlir"
                native_input.write_bytes(mlir_bytes)
                subprocess.run([str(args.mx_opt.resolve()), str(native_input),
                                "-o", "/dev/null"], check=True)
        else:
            subprocess.run([str(args.mx_opt.resolve()), str(args.mlir),
                            "-o", "/dev/null"], check=True)
    args.out_dir.mkdir(parents=True)
    issuer = args.out_dir / "mx_issue.c"
    names = tuple(entry["name"] for entry in buffer_abi)
    issuer.write_text(emit_c(pair.commands, transport="rocket_rocc", buffers=names))
    header = args.out_dir / "mx_issue.h"
    header.write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n\n"
        "/* Buffer order, sizes, and layouts are in object_manifest.json. */\n"
        f"void mx_issue({', '.join(f'const void *{name}' for name in names)});\n\n"
        "#endif\n")
    physical = args.out_dir / "physical_program.json"
    physical_record = {
        "schema": "mx_gemmini.resident_pair_physical.v1",
        "shape_mnk": [pair.plan.m, pair.plan.n, pair.plan.k],
        "first_site": pair.first_site, "second_site": pair.second_site,
        "profile_sha256": profile_sha256(profile),
        "plan": asdict(pair.plan),
        "commands": [({"kind": "command", **asdict(item)} if isinstance(item, Command)
                      else {"kind": "fence"}) for item in pair.commands],
    }
    if hasattr(pair.plan, "first_k"):
        physical_record["first_shape_mnk"] = [pair.plan.m, pair.plan.k,
                                              pair.plan.first_k]
    if args.precision != "fp8_e4m3":
        physical_record["precision"] = args.precision
    physical.write_text(json.dumps(physical_record, indent=2, sort_keys=True) + "\n")
    obj, data_bytes = _compile_object(args.out_dir, args.riscv_root)
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    manifest = {
        "schema": "mx_gemmini.resident_pair_linkable_object.v1",
        "status": "rv64_rocc_resident_pair_object_built",
        "transport": "rocket_rocc",
        "shape_mnk": [pair.plan.m, pair.plan.n, pair.plan.k],
        "first_site": pair.first_site, "second_site": pair.second_site,
        "buffer_abi": buffer_abi,
        "embedded_operand_bytes": 0, "embedded_golden_bytes": 0,
        "allocated_data_section_bytes": data_bytes,
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": _sha(mlir_bytes),
        "mlir_container_sha256": _file_sha(args.mlir),
        "abi_json_sha256": _file_sha(args.abi_json),
        "input_sha256": {slot: _sha(resources[buffers[slot]]) for slot in INPUTS},
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
    if hasattr(pair.plan, "first_k"):
        manifest["first_shape_mnk"] = [pair.plan.m, pair.plan.k,
                                        pair.plan.first_k]
    if args.precision != "fp8_e4m3":
        manifest["precision"] = args.precision
    (args.out_dir / "object_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.baseline_manifest is not None:
        baseline = json.loads(args.baseline_manifest.read_text())
        stable = ("schema", "status", "transport", "shape_mnk", "first_site",
                  "second_site", "buffer_abi", "profile_sha256",
                  "bound_mlir_sha256", "mlir_container_sha256",
                  "abi_json_sha256", "input_sha256",
                  "physical_program_sha256", "issuer_c_sha256",
                  "issuer_h_sha256", "object_sha256", "rtl_revision",
                  "riscv_gcc_sha256", "allocated_data_section_bytes",
                  "embedded_operand_bytes", "embedded_golden_bytes",
                  "defined_symbol", "undefined_symbols",
                  "object_emitter_sha256", "compiler_source_closure_sha256")
        if any(manifest[key] != baseline.get(key) for key in stable):
            raise ValueError("resident pair object differs from baseline")
    print(f"linkable resident MX pair: {obj}")


if __name__ == "__main__":
    main()
