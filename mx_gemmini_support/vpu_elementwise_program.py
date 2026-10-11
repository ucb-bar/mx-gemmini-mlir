"""Lower one verified typed MX VPU operation to runtime-buffer RoCC commands."""

from __future__ import annotations

from dataclasses import dataclass

from .command_ir import Command, VPU_OPCODES
from .physical_program import _cmd, _config_ld, _config_st, _transfer_rect
from .vector_lowering import lower_vector_commands


_BINARY = {"add", "sub", "mul", "max", "expsub", "expsum"}
_REDUCING = {"rmax", "ramax", "rsum"}


@dataclass(frozen=True)
class VpuElementwisePlan:
    kind: str
    rows: int
    reduction_length: int
    broadcast: bool
    src1_row: int
    src2_row: int
    dst_row: int
    second_dst_row: int | None
    src2_rows: int
    output_rows: int
    commands: tuple[Command, ...]


def _transfer_rows(funct: int, name: str, base: int, count: int) -> list[Command]:
    return [_transfer_rect(funct, name, offset * 16, base + offset,
                           rows=min(16, count - offset))
            for offset in range(0, count, 16)]


def lower_vpu_elementwise_program(mlir_text: str, profile: dict,
                                  buffers: dict[str, str]) -> VpuElementwisePlan:
    vector = lower_vector_commands(mlir_text, profile)
    if len(vector) != 1 or vector[0].funct != 33:
        raise ValueError("single VPU object needs exactly one typed vpu_execute")
    command = vector[0]
    if command.rs1.immediate is None or command.rs2.immediate is None:
        raise ValueError("VPU operation has no encoded physical address")
    rs1, rs2 = command.rs1.immediate, command.rs2.immediate
    by_opcode = {value: key for key, value in VPU_OPCODES.items()}
    kind = by_opcode.get(rs2 & 0xf)
    if kind is None:
        raise ValueError("VPU operation has no known physical opcode")
    src1_row = rs1 & 0x3fff
    src2_row = (rs1 >> 14) & 0x3fff
    dst_row = (rs1 >> 28) & 0x3fff
    rows = (rs1 >> 42) & 0xffff
    reduction_length = (rs2 >> 5) & 0x3ff
    broadcast = bool((rs2 >> 4) & 1)
    src2_rows = ((rows + reduction_length - 1) // reduction_length
                 if broadcast else rows) if kind in _BINARY else 0
    output_rows = rows // reduction_length if kind in _REDUCING else rows
    second_dst_row = (rs2 >> 16) & 0xffff if kind == "expsum" else None
    expected_slots = {"src1", "src2", "output"} | (
        {"output2"} if second_dst_row is not None else set())
    if set(buffers) != expected_slots or len(set(buffers.values())) != len(buffers):
        raise ValueError("VPU object buffer map differs from operation outputs")
    commands = [_cmd(7, 0, 0), _config_ld(16), _config_st(16),
                *_transfer_rows(2, buffers["src1"], src1_row, rows)]
    if src2_rows:
        commands.extend(_transfer_rows(2, buffers["src2"], src2_row, src2_rows))
    commands.extend((command, *_transfer_rows(
        3, buffers["output"], dst_row, output_rows)))
    if second_dst_row is not None:
        commands.extend(_transfer_rows(
            3, buffers["output2"], second_dst_row, rows // reduction_length))
    return VpuElementwisePlan(
        kind=kind, rows=rows, reduction_length=reduction_length,
        broadcast=broadcast, src1_row=src1_row, src2_row=src2_row,
        dst_row=dst_row, second_dst_row=second_dst_row,
        src2_rows=src2_rows, output_rows=output_rows,
        commands=tuple(commands))
