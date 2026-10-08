"""Physical MX command operands and the two selected issuer transports.

The same checked command stream can be issued by a Rocket RoCC instruction or
by the Muon-side Radiance MMIO gateway. This module does not choose MX
formats, layouts, or schedules: those belong to the selected contraction
lowering. Generated C still needs the matching target compiler and runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_OPCODE = 0x7B
_FUNCT3 = 3


@dataclass(frozen=True)
class Operand:
    """One 64-bit command field, literal or runtime buffer address plus offset."""

    immediate: int | None = None
    buffer: str | None = None
    byte_offset: int = 0

    def __post_init__(self) -> None:
        if (self.immediate is None) == (self.buffer is None):
            raise ValueError("MX operand needs exactly one immediate or buffer")
        if self.immediate is not None:
            if type(self.immediate) is not int or not 0 <= self.immediate < 1 << 64 or self.byte_offset:
                raise ValueError("MX immediate must fit 64 bits and have no offset")
        elif (not isinstance(self.buffer, str) or not _NAME.fullmatch(self.buffer) or
              type(self.byte_offset) is not int or self.byte_offset < 0):
            raise ValueError("MX buffer address needs a safe name and nonnegative byte offset")

    def c_expr(self) -> str:
        if self.immediate is not None:
            return f"UINT64_C(0x{self.immediate:016x})"
        return f"((uint64_t)(uintptr_t){self.buffer} + UINT64_C({self.byte_offset}))"


@dataclass(frozen=True)
class Command:
    funct: int
    rs1: Operand
    rs2: Operand

    def __post_init__(self) -> None:
        if type(self.funct) is not int or not 0 <= self.funct < 128:
            raise ValueError("MX funct must fit the selected seven-bit field")

    @property
    def instruction_word(self) -> int:
        # Source: radiance-kernels/lib/include/gemmini_mmio.h.
        return _OPCODE | (_FUNCT3 << 12) | (1 << 15) | (2 << 20) | (self.funct << 25)


@dataclass(frozen=True)
class WaitIdle:
    """Muon gateway busy-register completion check from gemmini_mmio.h."""


def emit_c(commands: list[Command | WaitIdle], *, transport: str, buffers: tuple[str, ...]) -> str:
    """Emit a freestanding command issuer with explicit runtime pointer inputs.

    `mx_control_base` is the selected Radiance tile's Gemmini control-register
    address; the caller obtains it from its verified SoC profile/runtime.
    """
    if transport not in ("rocket_rocc", "muon_mmio"):
        raise ValueError("MX transport must be rocket_rocc or muon_mmio")
    if transport == "rocket_rocc" and any(isinstance(item, WaitIdle) for item in commands):
        raise ValueError("Rocket RoCC has no selected busy-register completion endpoint")
    if (any(not isinstance(name, str) or not _NAME.fullmatch(name) for name in buffers) or
            len(set(buffers)) != len(buffers) or
            (transport == "muon_mmio" and "mx_control_base" in buffers)):
        raise ValueError("MX issuer needs unique safe buffer names")
    referenced = {operand.buffer for command in commands if isinstance(command, Command)
                  for operand in (command.rs1, command.rs2) if operand.buffer is not None}
    if referenced - set(buffers):
        raise ValueError(f"MX command references unbound buffers {sorted(referenced - set(buffers))}")
    parameters = [f"const void *{name}" for name in buffers]
    if transport == "muon_mmio":
        parameters.append("uintptr_t mx_control_base")
    lines = ["#include <stdint.h>", ""]
    if transport == "rocket_rocc":
        lines.extend(("_Static_assert(sizeof(uintptr_t) == 8, \"MX Rocket commands require RV64\");", ""))
    lines.append(f"void mx_issue({', '.join(parameters) or 'void'}) {{")
    for command in commands:
        if isinstance(command, WaitIdle):
            lines.append("  while (*(volatile uint32_t *)(mx_control_base + 0x20)) {}")
            continue
        if not isinstance(command, Command):
            raise ValueError("unknown MX command stream item")
        if transport == "rocket_rocc":
            lines.append(
                f'  __asm__ volatile (".insn r 0x7b, 3, {command.funct}, x0, %0, %1" '
                f': : "r"({command.rs1.c_expr()}), "r"({command.rs2.c_expr()}) : "memory");'
            )
        else:
            lines.extend((
                f"  *(volatile uint64_t *)(mx_control_base + 0x10) = {command.rs1.c_expr()};",
                f"  *(volatile uint64_t *)(mx_control_base + 0x18) = {command.rs2.c_expr()};",
                f"  *(volatile uint32_t *)(mx_control_base + 0x00) = UINT32_C(0x{command.instruction_word:08x});",
            ))
    lines.extend(("}", ""))
    return "\n".join(lines)
