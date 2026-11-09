"""Emit a data-free RV64 object for one typed MX VPU operation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vpu_emission import emit_vpu_elementwise, write_vpu_sources
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
    mlir_bytes = _load_mlir(args.mlir)
    buffers = _buffers(args.abi_json)
    emission = emit_vpu_elementwise(mlir_bytes.decode(), profile, buffers)
    plan, abi = emission.plan, emission.buffer_abi
    args.out_dir.mkdir(parents=True)
    issuer, header, physical = write_vpu_sources(args.out_dir, emission)
    obj, data_bytes = _compile_object(args.out_dir, args.riscv_root)
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
