"""Lower a flat typed VPU command sequence with explicit runtime buffers."""

from __future__ import annotations

from dataclasses import dataclass
import re

from .command_ir import Command, Fence, VPU_OPCODES
from .physical_program import _cmd, _config_ld, _config_st, _transfer_rect
from .vector_lowering import lower_vector_commands


SCHEMA = "mx_gemmini.vpu_sequence.v1"
ABI_SCHEMA = "mx_gemmini.vpu_sequence_buffer_map.v1"
TIMELINE_ABI_SCHEMA = "mx_gemmini.vpu_sequence_buffer_map.v2"
_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_BINARY = {"add", "sub", "mul", "max", "expsub", "expsum"}
_REDUCING = {"rmax", "ramax", "rsum"}


@dataclass(frozen=True)
class Buffer:
    name: str
    row: int
    rows: int


@dataclass(frozen=True)
class Capture:
    name: str
    row: int
    rows: int
    after_operation: int


@dataclass(frozen=True)
class Reload:
    input: str
    row: int
    after_operation: int


@dataclass(frozen=True)
class VpuSequencePlan:
    operations: tuple[dict, ...]
    inputs: tuple[Buffer, ...]
    outputs: tuple[Buffer, ...]
    commands: tuple[Command | Fence, ...]
    captures: tuple[Capture, ...] = ()
    reloads: tuple[Reload, ...] = ()
    abi_schema: str = ABI_SCHEMA


def _flat_vpu_function(mlir_text: str) -> None:
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    from .verify_profile_ir import _operation_name, _text_attr

    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    if _text_attr(module, "mx.vpu_sequence_schema") != SCHEMA:
        raise ValueError("VPU sequence MLIR lacks its explicit schema")
    functions = [op for op in module.walk() if op.name == "func.func"]
    if len(functions) != 1 or len(functions[0].regions[0].blocks) != 1:
        raise ValueError("VPU sequence needs one flat function")
    top_level = list(module.regions[0].blocks[0].ops)
    block = functions[0].regions[0].blocks[0]
    if top_level != functions or block.args:
        raise ValueError("VPU sequence needs one closed, flat function")
    ops = list(block.ops)
    if not ops or _operation_name(ops[-1]) != "func.return":
        raise ValueError("VPU sequence function needs a return")
    for op in ops[:-1]:
        if _operation_name(op) != "mx_gemmini.vpu_execute" or op.operands or op.results:
            raise ValueError("VPU sequence accepts only ordered physical VPU operations")


def parse_abi(spec: object, profile: dict) -> tuple[
        tuple[Buffer, ...], tuple[Buffer, ...], tuple[Capture, ...],
        tuple[Reload, ...], str]:
    if (not isinstance(spec, dict) or spec.get("schema") not in
            (ABI_SCHEMA, TIMELINE_ABI_SCHEMA)):
        raise ValueError("VPU sequence buffer map has an unsupported schema")
    timeline = spec["schema"] == TIMELINE_ABI_SCHEMA
    required = ({"schema", "inputs", "outputs", "reloads"} if timeline else
                {"schema", "inputs", "outputs"})
    if set(spec) != required:
        raise ValueError("VPU sequence buffer map has unsupported fields")
    limit = profile.get("resources", {}).get("scratchpad_bytes", 0) // 16

    def buffers(key: str) -> tuple[Buffer, ...]:
        rows = spec[key]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"VPU sequence {key} must be a nonempty list")
        result = []
        for item in rows:
            fields = {"name", "row", "rows", "after_operation"} if timeline and key == "outputs" else {"name", "row", "rows"}
            if (not isinstance(item, dict) or set(item) != fields or
                    not isinstance(item["name"], str) or
                    _NAME.fullmatch(item["name"]) is None or
                    type(item["row"]) is not int or type(item["rows"]) is not int or
                    item["row"] < 0 or item["rows"] <= 0 or
                    item["row"] + item["rows"] > limit or
                    (timeline and key == "outputs" and
                     (type(item["after_operation"]) is not int or
                      item["after_operation"] < 0))):
                raise ValueError(f"VPU sequence {key} has an invalid buffer span")
            result.append(Buffer(item["name"], item["row"], item["rows"]))
        return tuple(result)

    inputs, outputs = buffers("inputs"), buffers("outputs")
    names = [item.name for item in (*inputs, *outputs)]
    if len(set(names)) != len(names):
        raise ValueError("VPU sequence buffer names must be distinct")
    for group in ((inputs,) if timeline else (inputs, outputs)):
        occupied: set[int] = set()
        for item in group:
            span = set(range(item.row, item.row + item.rows))
            if occupied & span:
                raise ValueError("VPU sequence buffer spans overlap")
            occupied.update(span)
    if not timeline:
        return inputs, outputs, (), (), ABI_SCHEMA
    captures = tuple(Capture(**item) for item in spec["outputs"])
    if not isinstance(spec["reloads"], list):
        raise ValueError("VPU sequence reloads must be a list")
    by_name = {item.name: item for item in inputs}
    reloads = []
    for item in spec["reloads"]:
        if (not isinstance(item, dict) or set(item) != {"input", "row", "after_operation"} or
                not isinstance(item["input"], str) or item["input"] not in by_name or
                type(item["row"]) is not int or
                type(item["after_operation"]) is not int or item["row"] < 0 or
                item["row"] + by_name[item["input"]].rows > limit or
                item["after_operation"] < 0):
            raise ValueError("VPU sequence has an invalid reload")
        reloads.append(Reload(**item))
    return inputs, outputs, captures, tuple(reloads), TIMELINE_ABI_SCHEMA


def _transfer_rows(funct: int, buffer: Buffer) -> list[Command]:
    return [_transfer_rect(funct, buffer.name, offset * 16, buffer.row + offset,
                           rows=min(16, buffer.rows - offset))
            for offset in range(0, buffer.rows, 16)]


def lower_vpu_sequence(mlir_text: str, profile: dict, spec: object) -> VpuSequencePlan:
    _flat_vpu_function(mlir_text)
    inputs, outputs, captures, reloads, abi_schema = parse_abi(spec, profile)
    vectors = lower_vector_commands(mlir_text, profile)
    if not 2 <= len(vectors) <= 32 or any(command.funct != 33 for command in vectors):
        raise ValueError("VPU sequence needs 2..32 typed VPU operations")
    if any(event.after_operation >= len(vectors) for event in (*captures, *reloads)):
        raise ValueError("VPU sequence capture or reload is outside the operation list")
    initialized = {row for item in inputs
                   for row in range(item.row, item.row + item.rows)}
    operations = []
    commands = [_cmd(7, 0, 0), _config_ld(16), _config_st(16)]
    for item in inputs:
        commands.extend(_transfer_rows(2, item))
    by_opcode = {value: key for key, value in VPU_OPCODES.items()}
    for index, command in enumerate(vectors):
        if command.rs1.immediate is None or command.rs2.immediate is None:
            raise ValueError("VPU sequence operation needs physical addresses")
        rs1, rs2 = command.rs1.immediate, command.rs2.immediate
        kind = by_opcode.get(rs2 & 0xf)
        if kind is None:
            raise ValueError("VPU sequence has an unknown opcode")
        src1, src2 = rs1 & 0x3fff, (rs1 >> 14) & 0x3fff
        dst, rows = (rs1 >> 28) & 0x3fff, (rs1 >> 42) & 0xffff
        reduction_length, broadcast = (rs2 >> 5) & 0x3ff, bool((rs2 >> 4) & 1)
        src2_rows = ((rows + reduction_length - 1) // reduction_length
                     if broadcast else rows) if kind in _BINARY else 0
        dst_rows = rows // reduction_length if kind in _REDUCING else rows
        for label, start, count in (("source 1", src1, rows),
                                    ("source 2", src2, src2_rows)):
            if count and not set(range(start, start + count)) <= initialized:
                raise ValueError(f"VPU sequence operation {index} reads uninitialized {label}")
        second_dst = (rs2 >> 16) & 0xffff if kind == "expsum" else None
        initialized.update(range(dst, dst + dst_rows))
        if second_dst is not None:
            initialized.update(range(second_dst, second_dst + rows // reduction_length))
        operations.append({"kind": kind, "src1_row": src1, "src2_row": src2,
                           "dst_row": dst, "rows": rows,
                           "reduction_length": reduction_length,
                           "broadcast": broadcast, "src2_rows": src2_rows,
                           "dst_rows": dst_rows, "second_dst_row": second_dst})
        commands.append(command)
        for reload in reloads:
            if reload.after_operation == index:
                source = next(item for item in inputs if item.name == reload.input)
                target = Buffer(source.name, reload.row, source.rows)
                commands.extend(_transfer_rows(2, target))
                initialized.update(range(target.row, target.row + target.rows))
        captured = False
        for capture in captures:
            if capture.after_operation == index:
                if not set(range(capture.row, capture.row + capture.rows)) <= initialized:
                    raise ValueError(f"VPU sequence output {capture.name} is uninitialized")
                commands.extend(_transfer_rows(3, Buffer(capture.name, capture.row,
                                                       capture.rows)))
                captured = True
        if captured:
            commands.append(Fence())
    if abi_schema == ABI_SCHEMA:
        for item in outputs:
            if not set(range(item.row, item.row + item.rows)) <= initialized:
                raise ValueError(f"VPU sequence output {item.name} is uninitialized")
            commands.extend(_transfer_rows(3, item))
    return VpuSequencePlan(tuple(operations), inputs, outputs, tuple(commands),
                           captures, reloads, abi_schema)
