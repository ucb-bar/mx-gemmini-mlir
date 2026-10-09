"""Physical MX command operands and the two selected issuer transports.

The same checked command stream can be issued by a Rocket RoCC instruction or
by the Muon-side Radiance MMIO gateway. This module does not choose MX
formats, layouts, or schedules: those belong to the selected contraction
lowering. Generated C still needs the matching target compiler and runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping


_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")
_OPCODE = 0x7B
_FUNCT3 = 3


@dataclass(frozen=True)
class Operand:
    """One 64-bit command field, literal or runtime buffer address plus offset."""

    immediate: int | None = None
    buffer: str | None = None
    byte_offset: int = 0
    address_mask: int | None = None
    or_bits: int = 0

    def __post_init__(self) -> None:
        if (self.immediate is None) == (self.buffer is None):
            raise ValueError("MX operand needs exactly one immediate or buffer")
        if self.immediate is not None:
            if (type(self.immediate) is not int or not 0 <= self.immediate < 1 << 64 or
                    self.byte_offset or self.address_mask is not None or self.or_bits):
                raise ValueError("MX immediate must fit 64 bits and have no offset")
        elif (not isinstance(self.buffer, str) or not _NAME.fullmatch(self.buffer) or
              type(self.byte_offset) is not int or self.byte_offset < 0):
            raise ValueError("MX buffer address needs a safe name and nonnegative byte offset")
        if self.buffer is not None:
            if (type(self.or_bits) is not int or not 0 <= self.or_bits < 1 << 64 or
                    (self.address_mask is not None and
                     (type(self.address_mask) is not int or
                      self.address_mask not in ((1 << 33) - 1, (1 << 40) - 1) or
                      self.or_bits & self.address_mask)) or
                    (self.address_mask is None and self.or_bits)):
                raise ValueError("MX pointer field mask or command bits are invalid")

    def c_expr(self) -> str:
        if self.immediate is not None:
            return f"UINT64_C(0x{self.immediate:016x})"
        address = f"((uint64_t)(uintptr_t){self.buffer} + UINT64_C({self.byte_offset}))"
        if self.address_mask is None:
            return address
        return f"(({address} & UINT64_C(0x{self.address_mask:x})) | UINT64_C(0x{self.or_bits:x}))"


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


@dataclass(frozen=True)
class Fence:
    """Order earlier DMA and accelerator work before dependent commands."""


VPU_OPCODES = {
    "add": 0, "sub": 1, "mul": 2, "adds": 3, "muls": 4,
    "exp": 5, "rcp": 6, "rsqrt": 7, "rmax": 8, "rsum": 9,
    "ramax": 10, "max": 11, "expsub": 12, "expsum": 13,
}


def _mx_rows(profile: Mapping) -> int:
    if profile.get("schema") != "mx_gemmini.target_profile.v2":
        raise ValueError("MX vector commands need a source-bound target profile v2")
    resources = profile.get("resources", {})
    # The selected VPU and SPAD_REQUANT read 16-byte BF16 scratchpad rows.
    if profile.get("geometry", {}).get("mesh_columns") != 16:
        raise ValueError("MX vector row layout has only been derived for DIM16")
    return resources.get("scratchpad_bytes", 0) // 16


def _row_span(label: str, start: int, count: int, limit: int) -> None:
    if type(start) is not int or type(count) is not int or start < 0 or count <= 0 or start + count > limit:
        raise ValueError(f"MX {label} row span exceeds selected scratchpad")


def vpu_command(profile: Mapping, *, kind: str, src1_row: int, src2_row: int,
                dst_row: int, rows: int, reduction_length: int = 1,
                broadcast: bool = False, immediate_bf16: int = 0,
                second_dst_row: int | None = None) -> Command:
    """Encode RTL funct 33 after checking op gates and scratchpad row footprints."""
    limit = _mx_rows(profile)
    cfg = profile["resources"].get("vpu_config")
    if not profile["resources"].get("vpu") or not isinstance(cfg, dict):
        raise ValueError("selected MX profile has no VPU")
    if kind not in VPU_OPCODES:
        raise ValueError(f"unknown MX VPU operation {kind}")
    if kind == "expsub" and not cfg["exp_sub"] or kind == "expsum" and not cfg["exp_sum"]:
        raise ValueError(f"MX VPU operation {kind} is not built in selected profile")
    if type(rows) is not int or not 1 <= rows <= 0xffff:
        raise ValueError("MX VPU rows must fit 16 bits and be nonzero")
    if type(reduction_length) is not int or not 1 <= reduction_length <= 1023:
        raise ValueError("MX VPU reduction length must be 1..1023")
    if type(broadcast) is not bool:
        raise ValueError("MX VPU broadcast must be Boolean")
    reducing = kind in {"rmax", "rsum", "ramax"}
    if (reducing or kind == "expsum") and rows % reduction_length:
        raise ValueError("MX VPU logical rows must divide the input row count")
    _row_span("VPU source 1", src1_row, rows, limit)
    src2_count = (rows + reduction_length - 1) // reduction_length if broadcast else rows
    uses_src2 = kind in {"add", "sub", "mul", "max", "expsub", "expsum"}
    if uses_src2:
        _row_span("VPU source 2", src2_row, src2_count, limit)
    elif src2_row != 0 or broadcast:
        raise ValueError(f"MX VPU {kind} does not use source 2")
    _row_span("VPU destination", dst_row, rows // reduction_length if reducing else rows, limit)
    if kind == "expsum":
        if second_dst_row is None:
            raise ValueError("MX VPU EXPSUM needs a second destination")
        if immediate_bf16 != 0:
            raise ValueError("MX VPU EXPSUM uses the immediate field for its second destination")
        _row_span("VPU second destination", second_dst_row, rows // reduction_length, limit)
        immediate = second_dst_row
    else:
        if second_dst_row is not None:
            raise ValueError(f"MX VPU {kind} has no second destination")
        if kind not in {"adds", "muls"} and immediate_bf16 != 0:
            raise ValueError(f"MX VPU {kind} has no BF16 immediate")
        immediate = immediate_bf16
    if type(immediate) is not int or not 0 <= immediate <= 0xffff:
        raise ValueError("MX VPU immediate must fit 16 bits")
    if any(address > 0x3fff for address in (src1_row, src2_row, dst_row)):
        raise ValueError("MX VPU address must fit 14 bits")
    rs1 = src1_row | (src2_row << 14) | (dst_row << 28) | (rows << 42)
    rs2 = VPU_OPCODES[kind] | (int(broadcast) << 4) | (reduction_length << 5) | (immediate << 16)
    return Command(33, Operand(immediate=rs1), Operand(immediate=rs2))


def spad_requant_command(profile: Mapping, *, source_row: int, destination_row: int,
                         m: int, n: int, output_format: str, tiled: bool,
                         resident: bool, scale_dram_address: int) -> Command:
    """Encode RTL funct 34 for the selected DIM16 VPU/SPAD_REQUANT build."""
    limit = _mx_rows(profile)
    resources = profile["resources"]
    if not resources.get("spad_requant") or not resources.get("requantizer"):
        raise ValueError("selected MX profile has no SPAD_REQUANT")
    if output_format not in {"fp8_e4m3", "fp4_e2m1"}:
        raise ValueError("SPAD_REQUANT output must be E4M3 or FP4")
    if type(m) is not int or type(n) is not int or not 0 < m < 1 << 16 or not 0 < n < 1 << 16:
        raise ValueError("MX SPAD_REQUANT dimensions must fit 16 bits")
    blocks = m * n // 32
    if m % 8 or n % 32 or blocks % 32 or blocks > 2048:
        raise ValueError("MX SPAD_REQUANT requires M multiple of 8, N multiple of 32, and 32..2048 blocks")
    if type(tiled) is not bool or type(resident) is not bool:
        raise ValueError("MX SPAD_REQUANT layout and residency must be Boolean")
    if type(scale_dram_address) is not int or not 0 <= scale_dram_address < 1 << 33:
        raise ValueError("MX SPAD_REQUANT scale DRAM address must fit 33 bits")
    _row_span("SPAD_REQUANT source", source_row, 4 * blocks, limit)
    _row_span("SPAD_REQUANT destination", destination_row,
              blocks if output_format == "fp4_e2m1" else 2 * blocks, limit)
    if source_row > 0x3fff or destination_row > 0x3fff:
        raise ValueError("MX SPAD_REQUANT address must fit 14 bits")
    rs1 = (source_row | (destination_row << 14) | (int(tiled) << 28) |
           (int(resident) << 29) | (scale_dram_address << 30))
    rs2 = m | (n << 16) | (int(output_format == "fp4_e2m1") << 32)
    return Command(34, Operand(immediate=rs1), Operand(immediate=rs2))


def emit_c(commands: list[Command | WaitIdle | Fence], *, transport: str,
           buffers: tuple[str, ...]) -> str:
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
    checked = set()
    for command in commands:
        if not isinstance(command, Command):
            continue
        for operand in (command.rs1, command.rs2):
            if operand.address_mask is None:
                continue
            key = (operand.buffer, operand.byte_offset, operand.address_mask)
            if key in checked:
                continue
            checked.add(key)
            address = f"((uint64_t)(uintptr_t){operand.buffer} + UINT64_C({operand.byte_offset}))"
            lines.append(f"  if ({address} & ~UINT64_C(0x{operand.address_mask:x})) __builtin_trap();")
    for command in commands:
        if isinstance(command, Fence):
            lines.append('  __asm__ volatile ("fence" ::: "memory");' if transport == "rocket_rocc"
                         else "  __sync_synchronize();")
            continue
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
