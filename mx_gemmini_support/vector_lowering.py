"""Lower physical MX VPU and SPAD_REQUANT MLIR ops to a Rocket command issuer."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .command_ir import Command, emit_c, spad_requant_command, vpu_command
from .target_profile import load_profile
from .verify_profile_ir import _bool_attr, _int_attr, _operation_name, _text_attr, verify_ir


def lower_vector_commands(mlir_text: str, profile: dict) -> tuple[Command, ...]:
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    report = verify_ir(mlir_text, profile)
    if report["contracts"] or report["encodes"] or report["requantizes"]:
        raise ValueError("MX matrix handoff needs a separate executable matrix lowering")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    commands = []
    for op in module.walk():
        name = _operation_name(op)
        if name == "mx_gemmini.vpu_execute":
            commands.append(vpu_command(
                profile, kind=_text_attr(op, "kind"),
                src1_row=_int_attr(op, "src1_row"), src2_row=_int_attr(op, "src2_row"),
                dst_row=_int_attr(op, "dst_row"), rows=_int_attr(op, "rows"),
                reduction_length=_int_attr(op, "reduction_length"),
                broadcast=_bool_attr(op, "broadcast"),
                immediate_bf16=_int_attr(op, "immediate_bf16"),
                second_dst_row=_int_attr(op, "second_dst_row")))
        elif name == "mx_gemmini.spad_requant":
            commands.append(spad_requant_command(
                profile, source_row=_int_attr(op, "source_row"),
                destination_row=_int_attr(op, "destination_row"),
                m=_int_attr(op, "m"), n=_int_attr(op, "n"),
                output_format=_text_attr(op, "output_format"),
                tiled=_bool_attr(op, "tiled"), resident=_bool_attr(op, "resident"),
                scale_dram_address=_int_attr(op, "scale_dram_address"),
                scale_buffer=_text_attr(op, "scale_buffer")))
        elif name.startswith("mx_gemmini."):
            raise ValueError(f"MX operation {name} has no physical vector lowering")
    return tuple(commands)


def lower_vector_c(mlir_text: str, profile: dict) -> str:
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("selected MX profile does not describe a Rocket RoCC endpoint")
    commands = lower_vector_commands(mlir_text, profile)
    buffers = tuple(sorted({operand.buffer for command in commands
                            for operand in (command.rs1, command.rs2)
                            if operand.buffer is not None}))
    return emit_c(list(commands), transport="rocket_rocc", buffers=buffers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--mx-opt", type=Path, help="compiled native MLIR verifier")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if args.mx_opt:
        subprocess.run([str(args.mx_opt), str(args.mlir), "-o", "/dev/null"], check=True)
    output = lower_vector_c(args.mlir.read_text(), profile)
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output)
    print(f"lowered {args.mlir} -> {args.out} for {profile['name']}")


if __name__ == "__main__":
    main()
