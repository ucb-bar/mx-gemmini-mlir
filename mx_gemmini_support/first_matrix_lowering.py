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
    return emit_verified_first_matrix_commands(profile, resources,
                                               output_row=output_row)


def emit_verified_first_matrix_commands(profile: dict,
                                        resources: dict[str, bytes], *,
                                        output_row: int = 0x1000,
                                        buffers: dict[str, str] | None = None,
                                        shape: tuple[int, int, int] = (64, 64, 64)
                                        ) -> tuple[Command | Fence, ...]:
    """Emit a complete FP8 MM1 BF16 tile after its typed site was checked.

    The connected VPU source stores B1 in output-column tile order. Keep that
    physical layout explicit: a different B1 layout needs its own planner.
    """
    names = {name: name for name in
             ("a1_activation", "b1_weight", "a1_scales", "b1_scales",
              "c1_scales", "c1_bf16_observed")}
    if buffers is not None:
        if set(buffers) != set(names):
            raise ValueError("first MX matrix buffer map differs")
        names = buffers
    if (len(shape) != 3 or any(type(value) is not int or value <= 0
                               for value in shape)):
        raise ValueError("first MX matrix needs positive static dimensions")
    m, n, k = shape
    if (m % 16 or n % 32 or k % 32 or n != k or
            any(value // 16 > 0xffff for value in shape)):
        raise ValueError("first MX matrix needs a complete column-tile B1 layout")
    expected = {"a1_activation": m * k, "b1_weight": k * n,
                "a1_scales": m * k // 32, "b1_scales": k * n // 32}
    if any(len(resources.get(names[slot], b"")) != length
           for slot, length in expected.items()) or (
               buffers is None and len(resources.get("c1_bf16", b"")) !=
               m * n * 2):
        raise ValueError("first MX matrix source payload differs from its shape")
    rows = profile["resources"]["scratchpad_bytes"] // 16
    a_rows, b_rows, bf16_rows = m * k // 16, k * n // 16, m * n * 2 // 16
    b_base = rows - b_rows
    scale_capacity = profile["resources"]["scale_mem_config"]["size_bytes"] // 4
    if (profile["geometry"]["mesh_rows"] != 16 or
            profile["geometry"]["mesh_columns"] != 16 or
            type(output_row) is not int or output_row < a_rows or
            output_row % 16 or output_row + bf16_rows > b_base or
            rows > 1 << 14 or
            max(m * k, k * n, m * n) // 32 > scale_capacity or
            m * n * 2 > profile["resources"]["accumulator_bytes"] or
            "bf16" not in profile["candidate_output_modes"]):
        raise ValueError("first MX matrix scratchpad or BF16 output differs")
    i_tiles, j_tiles, k_tiles = m // 16, n // 16, k // 16

    def cmd(funct: int, rs1: int | Operand, rs2: int | Operand) -> Command:
        return Command(funct, rs1 if isinstance(rs1, Operand) else Operand(immediate=rs1),
                       rs2 if isinstance(rs2, Operand) else Operand(immediate=rs2))

    commands: list[Command | Fence] = [
        cmd(7, 0, 0),
        # E4M3 inputs, BF16 output, WS. Source uses the same mode with
        # E4M3 output; the BF16 specialization preserves the VPU input tile.
        cmd(0, (1 << 16) | (3 << 14) | (1 << 2), 1 << 48),
        cmd(27, Operand(buffer=names["a1_scales"]), m * k // 32),
        cmd(27, Operand(buffer=names["b1_scales"]), (1 << 32) | (k * n // 32)),
        Fence(),
        cmd(0, (16 << 16) | (1 << 8) | 1, k),
    ]
    for i in range(i_tiles):
        for kk in range(k_tiles):
            commands.append(cmd(2, Operand(buffer=names["a1_activation"],
                                           byte_offset=i * 16 * k + kk * 16),
                                (16 << 48) | (16 << 32) | (i * k_tiles + kk) * 16))
    for j in range(j_tiles):
        for kk in range(k_tiles):
            commands.append(cmd(2, Operand(buffer=names["b1_weight"],
                                           byte_offset=j * 16 * n + kk * 16),
                                (16 << 48) | (16 << 32) |
                                (b_base + (j * k_tiles + kk) * 16)))
    commands += [
        Fence(),
        cmd(0, 2, 2),
        cmd(26, Operand(buffer=names["c1_scales"], address_mask=(1 << 33) - 1,
                        or_bits=(k_tiles << 51) | (j_tiles << 42) |
                                (i_tiles << 33)), 1),
        cmd(9, 0, (k_tiles << 32) | (j_tiles << 16) | i_tiles),
        cmd(24, 0, rows),
        cmd(8, 0, (output_row << 32) | 0x200 | 0x38),
        Fence(),
        # Readback is diagnostic only. The VPU still consumes the live tile.
        cmd(0, 2, 16),
    ]
    for row in range(0, bf16_rows, 16):
        commands.append(cmd(3, Operand(buffer=names["c1_bf16_observed"], byte_offset=row * 16),
                            (16 << 48) | (16 << 32) | (output_row + row)))
    commands.append(Fence())
    return tuple(commands)
