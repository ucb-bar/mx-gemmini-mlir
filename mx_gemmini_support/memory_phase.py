"""Profile-checked physical DMA phases for standalone MX memory programs.

These commands describe transfers without inventing a contraction. A frontend
may use them to lower explicit transfer operations or benchmark phases.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from .command_ir import Command, Operand
from .layout import scale_load_rs2
from .physical_program import _cmd, _config_ld, _config_st
from .target_profile import profile_sha256


@dataclass(frozen=True)
class MemoryPhase:
    commands: tuple[Command, ...]
    transferred_bytes: int
    row_requests: int
    minimum_buffer_bytes: int


def _capacity(profile: Mapping) -> tuple[int, int]:
    if (profile.get("schema") != "mx_gemmini.target_profile.v2" or
            profile.get("transport") != "rocket_rocc" or
            profile.get("geometry", {}).get("mesh_columns") != 16):
        raise ValueError("standalone MX DMA needs a selected DIM16 RoCC profile")
    resources = profile.get("resources", {})
    scratchpad = resources.get("scratchpad_bytes")
    maximum = resources.get("dma_max_bytes")
    if (type(scratchpad) is not int or scratchpad <= 0 or scratchpad % 16 or
            type(maximum) is not int or maximum < 16 or maximum % 16):
        raise ValueError("selected MX profile has no checked DMA capacity")
    return scratchpad // 16, maximum


def lower_matrix_mvin(profile: Mapping, *, buffer: str, matrix_rows: int,
                      matrix_cols: int, burst_cols: int,
                      spad_row: int) -> MemoryPhase:
    """Load a row-major byte matrix in DIM16-high, 16..DMA-max-wide tiles."""
    spad_limit, maximum = _capacity(profile)
    if (any(type(n) is not int for n in
            (matrix_rows, matrix_cols, burst_cols, spad_row)) or
            matrix_rows <= 0 or matrix_rows % 16 or
            matrix_cols <= 0 or matrix_cols % 16 or
            burst_cols < 16 or burst_cols % 16 or
            burst_cols > maximum or matrix_cols % burst_cols or
            spad_row < 0 or spad_row % 16 or
            spad_row + matrix_rows * matrix_cols // 16 > spad_limit):
        raise ValueError("MX matrix DMA shape or scratchpad placement is unsupported")
    # The source instruction accepts 16-bit cols/rows but a single DMA row
    # cannot exceed the profile's actual maximum transfer width.
    if burst_cols >= 1 << 16 or matrix_cols >= 1 << 32:
        raise ValueError("MX matrix DMA width exceeds the command field")
    tiles_per_row = matrix_cols // 16
    commands = [_config_ld(matrix_cols)]
    for i in range(matrix_rows // 16):
        for k in range(0, tiles_per_row, burst_cols // 16):
            offset = i * 16 * matrix_cols + k * 16
            row = spad_row + (i * tiles_per_row + k) * 16
            commands.append(_cmd(
                2, Operand(buffer=buffer, byte_offset=offset),
                (16 << 48) | (burst_cols << 32) | row))
    requests = matrix_rows * matrix_cols // burst_cols
    return MemoryPhase(tuple(commands), matrix_rows * matrix_cols,
                       requests, matrix_rows * matrix_cols)


def lower_linear_spad_mvout(profile: Mapping, *, buffer: str, total_bytes: int,
                            tile_cols: int, spad_row: int) -> MemoryPhase:
    """Store consecutive DIM16 scratchpad tiles into a flat byte buffer."""
    spad_limit, maximum = _capacity(profile)
    if any(type(n) is not int for n in (total_bytes, tile_cols, spad_row)):
        raise ValueError("MX linear scratchpad store needs integer geometry")
    tile_bytes = 16 * tile_cols
    if (
            tile_cols < 16 or tile_cols % 16 or tile_cols > maximum or
            total_bytes <= 0 or total_bytes % tile_bytes or
            spad_row < 0 or spad_row % 16 or
            spad_row + total_bytes // 16 > spad_limit):
        raise ValueError("MX linear scratchpad store exceeds the selected profile")
    commands = [_config_st(tile_cols)]
    for offset in range(0, total_bytes, tile_bytes):
        row = spad_row + offset // 16
        commands.append(_cmd(
            3, Operand(buffer=buffer, byte_offset=offset),
            (16 << 48) | (tile_cols << 32) | row))
    return MemoryPhase(tuple(commands), total_bytes,
                       total_bytes // tile_cols, total_bytes)


def lower_scale_load(profile: Mapping, *, buffer: str,
                     payload_bytes: int, operand: str) -> MemoryPhase:
    """Issue the source's one-dimensional MX scale-loader command."""
    _capacity(profile)
    config = profile.get("resources", {}).get("scale_mem_config")
    if (type(payload_bytes) is not int or
            not isinstance(config, dict) or
            type(config.get("size_bytes")) is not int or
            payload_bytes > config["size_bytes"] // 2):
        raise ValueError("selected MX scale memory cannot hold this operand")
    command = _cmd(27, Operand(buffer=buffer),
                   scale_load_rs2(payload_bytes, operand=operand))
    return MemoryPhase((command,), payload_bytes,
                       payload_bytes // 8, payload_bytes)


def render_nicolas_memory_mlir(profile: Mapping, source_sha256: str) -> str:
    """Describe the source's reusable transfer phases as typed MX operations."""
    if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise ValueError("MX memory source digest must be SHA-256")
    _capacity(profile)
    digest = profile_sha256(profile)
    shared = f'profile_sha256 = "{digest}", source_sha256 = "{source_sha256}"'

    def function(name: str, bytes_: int, operation: str, attributes: str) -> str:
        return (f"  func.func @mx_issue_{name}(%arg0: memref<{bytes_}xi8>) {{\n"
                f'    "mx_gemmini.{operation}"(%arg0) '
                f'{{site_id = "{name}", {attributes}, {shared}}} : '
                f'(memref<{bytes_}xi8>) -> ()\n'
                "    return\n  }\n")

    return (f'module attributes {{mx.profile_sha256 = "{digest}", '
            f'mx.source_sha256 = "{source_sha256}"}} {{\n'
            f'  func.func @mx_issue_setup() {{\n'
            f'    "mx_gemmini.memory_setup"() '
            f'{{site_id = "setup", {shared}}} : () -> ()\n'
            '    return\n  }\n'
            + function("a64", 16384, "dma_matrix",
                       "matrix_rows = 128 : i32, matrix_cols = 128 : i32, "
                       "burst_cols = 64 : i32, spad_row = 0 : i32")
            + function("b16", 16384, "dma_matrix",
                       "matrix_rows = 128 : i32, matrix_cols = 128 : i32, "
                       "burst_cols = 16 : i32, spad_row = 1024 : i32")
            + function("scale", 512, "load_scales",
                       'payload_bytes = 512 : i32, scale_target = "activation"')
            + function("mvout", 16384, "spad_mvout_linear",
                       "total_bytes = 16384 : i32, tile_cols = 16 : i32, "
                       "spad_row = 0 : i32")
            + "}\n")


def lower_nicolas_memory_mlir(mlir_text: str, profile: Mapping,
                              source_sha256: str) -> dict[str, tuple[Command, ...]]:
    """Lower the typed source-bound phases to five linkable MX entry points."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser

    from .verify_profile_ir import _int_attr, _operation_name, _text_attr, verify_ir

    checked = verify_ir(mlir_text, dict(profile))
    if checked["memory_phases"] != 5 or checked["contracts"]:
        raise ValueError("MX memory benchmark needs exactly five transfer phases")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    if _text_attr(module, "mx.source_sha256") != source_sha256:
        raise ValueError("MX memory program source digest differs")
    functions = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(functions) != 5:
        raise ValueError("MX memory program needs five phase functions")
    expected = {"setup": "memory_setup", "a64": "dma_matrix",
                "b16": "dma_matrix", "scale": "load_scales",
                "mvout": "spad_mvout_linear"}
    commands: dict[str, tuple[Command, ...]] = {}
    for function in functions:
        symbol = function.sym_name.data
        name = symbol.removeprefix("mx_issue_") if symbol else None
        if name not in expected or name in commands or symbol != f"mx_issue_{name}":
            raise ValueError("MX memory phase function name differs")
        ops = list(function.body.block.ops)
        if (len(ops) != 2 or _operation_name(ops[0]) !=
                f"mx_gemmini.{expected[name]}" or
                not isinstance(ops[1], ReturnOp) or ops[1].operands or
                _text_attr(ops[0], "site_id") != name):
            raise ValueError("MX memory phase operation or return differs")
        op = ops[0]
        args = list(function.body.block.args)
        if name == "setup":
            if args or op.operands:
                raise ValueError("MX memory setup has no runtime buffer")
            commands[name] = (_cmd(7, 0, 0),
                              _cmd(0, (1 << 16) | (3 << 14) | (1 << 2), 1 << 48))
            continue
        if len(args) != 1 or list(op.operands) != args:
            raise ValueError("MX memory transfer must consume its function buffer")
        expected_type = "memref<512xi8>" if name == "scale" else "memref<16384xi8>"
        if str(args[0].type) != expected_type:
            raise ValueError("MX memory transfer ABI buffer type differs")
        buffer = "dst" if name == "mvout" else "src"
        if name in {"a64", "b16"}:
            phase = lower_matrix_mvin(
                profile, buffer=buffer,
                matrix_rows=_int_attr(op, "matrix_rows"),
                matrix_cols=_int_attr(op, "matrix_cols"),
                burst_cols=_int_attr(op, "burst_cols"),
                spad_row=_int_attr(op, "spad_row"))
        elif name == "scale":
            phase = lower_scale_load(
                profile, buffer=buffer,
                payload_bytes=_int_attr(op, "payload_bytes"),
                operand=_text_attr(op, "scale_target"))
        else:
            phase = lower_linear_spad_mvout(
                profile, buffer=buffer,
                total_bytes=_int_attr(op, "total_bytes"),
                tile_cols=_int_attr(op, "tile_cols"),
                spad_row=_int_attr(op, "spad_row"))
        commands[name] = phase.commands
    if set(commands) != set(expected):
        raise ValueError("MX memory program is missing a transfer phase")
    return commands
