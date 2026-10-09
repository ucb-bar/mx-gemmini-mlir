"""Lower Nicolas's first FP8 matrix site to a BF16 scratchpad tile.

This is a source-bound DIM16 schedule. The frontend MLIR must contain both
model2MLIR contraction sites, and the numerical payload is audited against
Nicolas's checked-in chain header by source_vector_chain.py.
"""

from __future__ import annotations

from .command_ir import Command, Fence, Operand
from .verify_profile_ir import _operation_name, _text_attr, verify_ir


def lower_first_matrix_commands(frontend_mlir: str, profile: dict,
                                resources: dict[str, bytes], *,
                                output_row: int = 0x1000) -> tuple[Command | Fence, ...]:
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    report = verify_ir(frontend_mlir, profile)
    if (report["contracts"], report["resident_contracts"],
            report["vpu_commands"], report["spad_requants"]) != (2, 0, 0, 0):
        raise ValueError("first MX matrix requires the two-site model2MLIR handoff")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, frontend_mlir).parse_module()
    contracts = [op for op in module.walk() if _operation_name(op) == "mx_gemmini.contract"]
    if ([_text_attr(op, "site_id") for op in contracts] !=
            ["functional:matmul", "functional:matmul_1"] or
            any((_text_attr(op, "activation_format"),
                 _text_attr(op, "weight_format"),
                 _text_attr(op, "activation_projection"),
                 _text_attr(op, "weight_projection")) !=
                ("fp8_e4m3", "fp8_e4m3", "direct", "direct") for op in contracts)):
        raise ValueError("first MX matrix site or selected precision differs")
    if {name: len(resources.get(name, b"")) for name in
            ("a1_activation", "b1_weight", "a1_scales", "b1_scales", "c1_bf16")
            } != {"a1_activation": 4096, "b1_weight": 4096,
                  "a1_scales": 128, "b1_scales": 128, "c1_bf16": 8192}:
        raise ValueError("first MX matrix source payload differs from 64x64x64")
    rows = profile["resources"]["scratchpad_bytes"] // 16
    b_base = rows - 256
    if (profile["geometry"]["mesh_columns"] != 16 or
            type(output_row) is not int or output_row != 0x1000 or
            output_row + 512 > b_base or
            "bf16" not in profile["candidate_output_modes"]):
        raise ValueError("first MX matrix scratchpad or BF16 output differs")

    def cmd(funct: int, rs1: int | Operand, rs2: int | Operand) -> Command:
        return Command(funct, rs1 if isinstance(rs1, Operand) else Operand(immediate=rs1),
                       rs2 if isinstance(rs2, Operand) else Operand(immediate=rs2))

    commands: list[Command | Fence] = [
        cmd(7, 0, 0),
        # E4M3 inputs, BF16 output, WS. Source uses the same mode with
        # E4M3 output; the BF16 specialization preserves the VPU input tile.
        cmd(0, (1 << 16) | (3 << 14) | (1 << 2), 1 << 48),
        cmd(27, Operand(buffer="a1_scales"), 128),
        cmd(27, Operand(buffer="b1_scales"), (1 << 32) | 128),
        Fence(),
        cmd(0, (16 << 16) | (1 << 8) | 1, 64),
    ]
    for i in range(4):
        for k in range(4):
            commands.append(cmd(2, Operand(buffer="a1_activation",
                                           byte_offset=i * 16 * 64 + k * 16),
                                (16 << 48) | (16 << 32) | (i * 4 + k) * 16))
    for j in range(4):
        for k in range(4):
            commands.append(cmd(2, Operand(buffer="b1_weight",
                                           byte_offset=j * 16 * 64 + k * 16),
                                (16 << 48) | (16 << 32) | (b_base + (j * 4 + k) * 16)))
    commands += [
        Fence(),
        cmd(0, 2, 2),
        cmd(26, Operand(buffer="c1_scales", address_mask=(1 << 33) - 1,
                        or_bits=(4 << 51) | (4 << 42) | (4 << 33)), 1),
        cmd(9, 0, (4 << 32) | (4 << 16) | 4),
        cmd(24, 0, rows),
        cmd(8, 0, (output_row << 32) | 0x200 | 0x38),
        Fence(),
        # Readback is diagnostic only. The VPU still consumes the live tile.
        cmd(0, 2, 16),
    ]
    for row in range(0, 512, 16):
        commands.append(cmd(3, Operand(buffer="c1_bf16_observed", byte_offset=row * 16),
                            (16 << 48) | (16 << 32) | (output_row + row)))
    commands.append(Fence())
    return tuple(commands)
