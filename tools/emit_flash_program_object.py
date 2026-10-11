"""Emit a linkable RV64 object from one source-bound typed flash program."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.nicolas_attention_shape import DenseAttentionShape
from mx_gemmini_support.nicolas_flash_mid import (STREAM_SHAPE, flash_buffer_abi,
                                                  lower_flash_mid,
                                                  render_flash_mid)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _source_closure
from tools.emit_resident_pair_object import _compile_object, _file_sha, _load_mlir


ROOT = Path(__file__).resolve().parents[1]
PINNED_CORE_SHA256 = "004459aa0e95def61427d5a6e9d8f71be59fcbc281a93a35745c1f33591bf7c0"
PINNED_MID_WRAPPER_SHA256 = "5f46198f35539b261ccbc2c689ce56beb823ba7e6ec74d8ff60936418619089f"
PINNED_MID_FP4_WRAPPER_SHA256 = "650ceb3c3b77cfd5b91521b879921debe9e35832fe1d1c5435ad09ee1fb6d6c0"
PINNED_MID_EXPSUM_WRAPPER_SHA256 = "549b0440b05b149c84059985f3b20f55338da7af879d8272557f662cb263824d"
PINNED_MID_FP4_EXPSUM_WRAPPER_SHA256 = "fca0f83b3a81ecb5210bc69a08f2329bcb70447441b2e02b88e5388e7aa9218c"
PINNED_LLAMA_WRAPPER_SHA256 = "1245d6c5d994687eda049ff4ddfa7577c0cf0e246b26767c4bb6ba6e641224f2"
PINNED_LLAMA_CAUSAL_WRAPPER_SHA256 = "687f2d4be5cf00dabbe4daccc0b07f71a2cc24e6521e17cbbff9db48fac3319f"


def _source_text(path: Path) -> str:
    data = path.read_bytes()
    return (gzip.decompress(data) if path.name.endswith(".gz") else data).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlir", "precursor-mlir", "binding-json", "source-core",
                 "source-wrapper", "source-header", "profile", "rtl-root",
                 "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("mlir", "precursor_mlir", "binding_json", "source_core",
                 "source_wrapper", "source_header", "profile", "rtl_root",
                 "riscv_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("flash object needs Rocket RoCC transport")
    mlir_bytes = _load_mlir(args.mlir)
    precursor_bytes = _load_mlir(args.precursor_mlir)
    binding = json.loads(args.binding_json.read_text())
    dims = binding.get("policy", {}).get("shape_sq_sk_d_bk")
    if (not isinstance(dims, list) or len(dims) != 4 or
            any(type(value) is not int for value in dims)):
        raise ValueError("flash object binding has no selected source shape")
    shape, block_k = DenseAttentionShape(*dims[:3]), dims[3]
    input_format = binding["policy"].get("input_format", "fp8_e4m3")
    softmax_mode = binding["policy"].get("softmax_mode", "split")
    causal = binding["policy"].get("causal", False)
    if type(causal) is not bool:
        raise ValueError("flash causal policy must be boolean")
    buffers, buffer_bytes = flash_buffer_abi(shape, block_k, input_format,
                                             causal=causal)
    wrapper_sha256 = (PINNED_LLAMA_CAUSAL_WRAPPER_SHA256 if causal else
                      PINNED_LLAMA_WRAPPER_SHA256
                      if shape == STREAM_SHAPE and block_k == 256
                      else PINNED_MID_FP4_EXPSUM_WRAPPER_SHA256
                      if softmax_mode == "expsum" and input_format == "fp4_e2m1"
                      else PINNED_MID_EXPSUM_WRAPPER_SHA256 if softmax_mode == "expsum"
                      else PINNED_MID_FP4_WRAPPER_SHA256 if input_format == "fp4_e2m1"
                      else PINNED_MID_WRAPPER_SHA256 if block_k == 32
                      else PINNED_CORE_SHA256)
    if (_file_sha(args.source_core) != PINNED_CORE_SHA256 or
            _file_sha(args.source_wrapper) != wrapper_sha256):
        raise ValueError("flash object needs pinned Nicolas source core and wrapper")
    frontend = precursor_bytes.decode()
    core = _source_text(args.source_core)
    wrapper = _source_text(args.source_wrapper)
    header = _source_text(args.source_header)
    expected, selected = render_flash_mid(
        frontend, core, wrapper, header, profile, shape=shape, block_k=block_k,
        input_format=input_format, softmax_mode=softmax_mode, causal=causal)
    if binding != selected or mlir_bytes.decode() != expected:
        raise ValueError("typed flash source, frontend, graph, or binding differs")
    commands = lower_flash_mid(
        expected, frontend, core, wrapper, header, profile,
        shape=shape, block_k=block_k, input_format=input_format,
        softmax_mode=softmax_mode, causal=causal)
    slug = ("flash_llama_causal_stream" if causal else
            "flash_llama_stream" if shape == STREAM_SHAPE and block_k == 256 else
            "flash_mid_fp4_expsum" if softmax_mode == "expsum" and input_format == "fp4_e2m1"
            else "flash_mid_expsum" if softmax_mode == "expsum" else
            "flash_mid_fp4" if input_format == "fp4_e2m1" else
            "flash_mid" if block_k == 32 else "flash_fa")
    args.out_dir.mkdir(parents=True)
    issuer = args.out_dir / "mx_issue.c"
    issuer.write_text(emit_c(list(commands), transport="rocket_rocc", buffers=buffers))
    header_path = args.out_dir / "mx_issue.h"
    header_path.write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n"
        "void mx_issue(" + ", ".join(f"const void *{name}" for name in buffers) +
        ");\n#endif\n")
    physical = args.out_dir / "physical_program.json"
    physical.write_text(json.dumps({
        "schema": f"mx_gemmini.{slug}_serial_physical.v1",
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": hashlib.sha256(mlir_bytes).hexdigest(),
        "commands": [asdict(command) if isinstance(command, Command)
                     else {"kind": "fence"} for command in commands],
    }, indent=2, sort_keys=True) + "\n")
    obj, data_bytes = _compile_object(args.out_dir, args.riscv_root)
    command_count = sum(isinstance(command, Command) for command in commands)
    manifest = {
        "schema": "mx_gemmini.flash_program_linkable_object.v1",
        "status": "rv64_rocc_typed_flash_program_object_built",
        "transport": "rocket_rocc", "schedule": "fenced_serial",
        "shape_sq_sk_d_bk": dims,
        "input_format": input_format,
        "softmax_mode": softmax_mode,
        "causal": causal,
        "buffer_abi": [{"name": name, "position": index,
                        "role": "write" if name in {"scale_sink", "p_scales_0",
                                                    "p_scales_1", "O_HW"} else "read",
                        "minimum_bytes": length, "alignment_bytes": 64,
                        "layout": f"flat_bytes_source_{slug}"}
                       for index, (name, length) in enumerate(zip(buffers, buffer_bytes))],
        "command_count": command_count,
        "fence_count": len(commands) - command_count,
        "embedded_operand_bytes": 0, "embedded_golden_bytes": 0,
        "allocated_data_section_bytes": data_bytes,
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": hashlib.sha256(mlir_bytes).hexdigest(),
        "precursor_mlir_sha256": hashlib.sha256(precursor_bytes).hexdigest(),
        "mlir_container_sha256": _file_sha(args.mlir),
        "binding_json_sha256": _file_sha(args.binding_json),
        "source_core_sha256": _file_sha(args.source_core),
        "source_wrapper_sha256": _file_sha(args.source_wrapper),
        "source_header_sha256": hashlib.sha256(header.encode()).hexdigest(),
        "physical_program_sha256": _file_sha(physical),
        "issuer_c_sha256": _file_sha(issuer),
        "issuer_h_sha256": _file_sha(header_path),
        "object_sha256": _file_sha(obj),
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
    print(f"linkable {slug} object: {obj}")


if __name__ == "__main__":
    main()
