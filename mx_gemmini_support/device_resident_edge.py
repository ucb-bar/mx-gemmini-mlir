"""Source-bound BF16 matrix-to-VPU scratchpad edges.

This contract covers flat BF16 LOOP_WS outputs.  It intentionally refuses
requantized/tiled output and layouts whose physical row effects are unknown.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from math import prod
from typing import Any

from .command_ir import Command, Fence, emit_c, vpu_command
from .physical_memory_plan import source_bound_capacities
from .physical_program import _config_st, _transfer_rect
from .target_profile import model_vpu_lanes, profile_sha256
from .vpu_unary_kernel import sigmoid_steps


class DeviceEdgeError(ValueError):
    """The selected matrix result cannot feed this VPU region in place."""


@dataclass(frozen=True)
class DeviceEdgePlan:
    schema: str
    producer: str
    consumers: tuple[str, ...]
    profile_sha256: str
    matrix_source_sha256: str
    producer_source_kind: str
    logical_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    view_shape: tuple[int, ...]
    source_row: int
    physical_rows: int
    full_padded_rows: int
    matrix_weight_first_row: int
    row_stride_rows: int
    row_bytes: int
    lanes: int
    alias_kind: str
    element_dtype: str
    output_encoding: str

    def address(self, row: int, column: int) -> tuple[int, int]:
        """Return physical scratchpad row and BF16 lane of one logical value."""
        m, n = self.logical_shape
        if not 0 <= row < m or not 0 <= column < n:
            raise DeviceEdgeError("logical matrix index exceeds captured extent")
        return self.source_row + row * self.row_stride_rows + column // self.lanes, column % self.lanes

    def receipt(self) -> dict[str, Any]:
        value = asdict(self)
        value["logical_mapping"] = "source_row + row*(padded_n/lanes) + col//lanes; lane=col%lanes"
        value["host_edge_bytes"] = 0
        return value


@dataclass(frozen=True)
class ResidentKernel:
    source: str
    commands: tuple[Command | Fence, ...]
    source_rows: tuple[int, ...]
    temporary_row: int
    result_row: int
    multiply_row: int | None
    output_bytes: int
    vpu_command_count: int
    dma_readout_commands: int
    source_sha256: str


@dataclass(frozen=True)
class ResidentTensor:
    """One BF16 row-major value whose producer and scratchpad row are known."""

    profile_sha256: str
    producer: str
    producer_source_sha256: str
    source_kind: str
    logical_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    source_row: int
    physical_rows: int
    row_stride_rows: int
    row_bytes: int
    lanes: int
    dtype: str = "bf16"

    def receipt(self) -> dict[str, Any]:
        return asdict(self)


def matrix_resident_tensor(edge: DeviceEdgePlan) -> ResidentTensor:
    """Expose a certified matrix C as a reusable VPU operand."""
    return ResidentTensor(edge.profile_sha256, edge.producer,
                          edge.matrix_source_sha256, "flat_bf16_matrix_C",
                          edge.logical_shape, edge.padded_shape,
                          edge.source_row, edge.physical_rows,
                          edge.row_stride_rows, edge.row_bytes, edge.lanes)


def vpu_result_tensor(edge: DeviceEdgePlan, kernel: ResidentKernel, *,
                      result: str, producer: str) -> ResidentTensor:
    """Name the sigmoid or multiply result of a checked resident issuer."""
    if result not in {"sigmoid", "multiply"} or not producer:
        raise DeviceEdgeError("resident VPU result must name a known output")
    row = kernel.result_row if result == "sigmoid" else kernel.multiply_row
    if row is None or kernel.output_bytes != edge.physical_rows * edge.row_bytes:
        raise DeviceEdgeError("resident VPU output does not match edge row layout")
    return ResidentTensor(edge.profile_sha256, producer, kernel.source_sha256,
                          "resident_vpu_" + result, edge.logical_shape,
                          edge.padded_shape, row, edge.physical_rows,
                          edge.row_stride_rows, edge.row_bytes, edge.lanes)


def _sha(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def plan_matrix_vpu_edge(profile: dict, *, matrix_plan: Any,
                         matrix_source: str, logical_shape: tuple[int, int],
                         padded_shape: tuple[int, int], view_shape: tuple[int, ...],
                         producer: str, consumers: tuple[str, ...],
                         producer_source_kind: str = "generated_driver") -> DeviceEdgePlan:
    """Prove a generated LOOP_WS BF16 C tile is a row-major VPU alias.

    ``matrix_plan`` is the actual ``mxgemm_emit.MxGemmPlan`` selected by the
    source-bound recipe, rather than an inferred shape.  The generated matrix
    issuer and its flat readout are checked against that plan before any ELF.
    """
    dim = profile.get("geometry", {}).get("mesh_rows")
    lanes = model_vpu_lanes(profile)
    capacities = source_bound_capacities(profile)
    if (dim not in (4, 8, 16, 32) or lanes is None or lanes * 2 != dim or
            profile["geometry"].get("mesh_columns") != dim):
        raise DeviceEdgeError("matrix/VPU edge needs a selected coupled square mesh")
    if (not isinstance(producer, str) or not producer or
            not consumers or any(not isinstance(name, str) or not name for name in consumers) or
            len(set(consumers)) != len(consumers)):
        raise DeviceEdgeError("matrix/VPU edge needs named producer and consumers")
    if (len(logical_shape) != 2 or len(padded_shape) != 2 or
            any(type(x) is not int or x <= 0 for x in (*logical_shape, *padded_shape, *view_shape)) or
            prod(view_shape) != prod(logical_shape)):
        raise DeviceEdgeError("matrix/VPU view must preserve every logical element")
    lm, ln = logical_shape
    pm, pn = padded_shape
    if (lm > pm or ln > pn or pm % dim or pn % dim or pn % lanes):
        raise DeviceEdgeError("matrix/VPU padding is not mesh-row aligned")
    p = matrix_plan
    if ((p.m, p.n) != padded_shape or p.dim != dim or p.out_fmt != "bf16" or
            p.chained_out or p.chained_in or p.use_lut or
            p.spad_dest < 0 or p.out_bytes != pm * pn * 2):
        raise DeviceEdgeError("matrix producer is not a flat BF16 LOOP_WS result")
    physical_rows = lm * pn // lanes
    full_rows = p.out_bytes // dim
    if (p.spad_dest + full_rows > capacities["scratchpad"] or
            p.spad_dest + full_rows > p.b_base or
            p.spad_dest < p.tiles_i * p.tiles_k * dim):
        raise DeviceEdgeError("matrix C rows overlap A, B, or scratchpad end")
    common = (
        f"gemmini_loop_ws_spad({p.tiles_i}, {p.tiles_j}, {p.tiles_k},",
        f"                       {p.spad_dest},",
        "                       0x38);",
    )
    generated_driver = (
        f"for (int r = 0; r < {full_rows}; r += {dim})",
        f"gemmini_extended_mvout(c_base + r * {dim}, {p.spad_dest} + r, {dim}, {dim});",
        "  gemmini_fence();\n  c1 = read_cycles();",
    )
    if any(matrix_source.count(fragment) != 1 for fragment in common):
        raise DeviceEdgeError("generated matrix commands differ from flat BF16 row contract")
    if producer_source_kind == "generated_driver":
        source_qualified = (matrix_source.startswith("/* Generated by mxgemm_emit.py") and
                            all(matrix_source.count(fragment) == 1 for fragment in generated_driver))
    elif producer_source_kind == "composable_stage":
        source_qualified = (
            matrix_source.startswith("{\n  /* source matrix " + producer +
                                     f": {p.m}x{p.n}x{p.k} */") and
            matrix_source.rstrip().endswith("gemmini_fence();\n}") and
            "gemmini_extended_mvout" not in matrix_source)
    else:
        raise DeviceEdgeError("unknown matrix producer source kind")
    if not source_qualified:
        raise DeviceEdgeError("generated matrix commands differ from flat BF16 row contract")
    return DeviceEdgePlan(
        "mx_gemmini.device_edge.v1", producer, tuple(consumers),
        profile_sha256(profile), _sha(matrix_source), producer_source_kind,
        logical_shape, padded_shape,
        view_shape, p.spad_dest, physical_rows, full_rows, p.b_base,
        pn // lanes, dim, lanes,
        "row_major_collapse_expand_no_permutation", "bf16", "flat_bf16_loop_ws")


def _windows(profile: dict, edge: DeviceEdgePlan, *, output_row: int | None = None,
             second_output: bool = False) -> tuple[int, int, int | None]:
    total = source_bound_capacities(profile)["scratchpad"]
    if edge.profile_sha256 != profile_sha256(profile):
        raise DeviceEdgeError("resident edge profile changed")
    # Reserve a complete logical producer extent, even if VPU commands split it.
    if total >= 0x2400:
        temp, result, multiply = 0x1000, 0x2000, 0x3000
    else:
        temp, result, multiply = 0x400, 0x800, 0xc00
    if output_row is not None:
        result = output_row
    spans = [("matrix_C_full", edge.source_row, edge.source_row + edge.full_padded_rows),
             ("temporary", temp, temp + min(edge.physical_rows, 1024)),
             ("result", result, result + edge.physical_rows)]
    if second_output:
        spans.append(("multiply", multiply, multiply + edge.physical_rows))
    for name, lo, hi in spans:
        if lo < 0 or hi > total:
            raise DeviceEdgeError(f"{name} rows exceed source-bound scratchpad capacity")
        if name != "matrix_C_full" and hi > edge.matrix_weight_first_row:
            raise DeviceEdgeError(f"resident {name} rows overlap retained matrix weights")
    for i, (name, lo, hi) in enumerate(spans):
        for other, lo2, hi2 in spans[i + 1:]:
            if max(lo, lo2) < min(hi, hi2):
                raise DeviceEdgeError(f"resident {name}/{other} row lifetimes overlap")
    return temp, result, multiply if second_output else None


def _readout(profile: dict, *, buffer: str, source_row: int,
             physical_rows: int, row_bytes: int) -> list[Command | Fence]:
    commands: list[Command | Fence] = [_config_st(row_bytes), Fence()]
    for start in range(0, physical_rows, 16):
        commands.append(_transfer_rect(3, buffer, start * row_bytes,
                                       source_row + start,
                                       rows=min(16, physical_rows - start),
                                       cols=row_bytes))
    commands.append(Fence())
    return commands


def emit_resident_sigmoid_multiply(profile: dict, edge: DeviceEdgePlan, *,
                                   symbol: str) -> ResidentKernel:
    """Keep C live for sigmoid and the original-C multiply branch."""
    temp, result, multiply = _windows(profile, edge, second_output=True)
    assert multiply is not None
    commands: list[Command | Fence] = []
    starts = tuple(range(0, edge.physical_rows, 1024))
    for start in starts:
        rows = min(1024, edge.physical_rows - start)
        source_row = edge.source_row + start
        commands.extend(vpu_command(profile, rows=rows, **step) for step in
                        sigmoid_steps(source_row=source_row, temporary_row=temp,
                                      result_row=result + start))
        commands.append(Fence())
        commands.append(vpu_command(profile, kind="mul", src1_row=source_row,
                                    src2_row=result + start, dst_row=multiply + start,
                                    rows=rows))
        commands.append(Fence())
    commands.extend(_readout(profile, buffer="sigmoid", source_row=result,
                             physical_rows=edge.physical_rows, row_bytes=edge.row_bytes))
    commands.extend(_readout(profile, buffer="multiply", source_row=multiply,
                             physical_rows=edge.physical_rows, row_bytes=edge.row_bytes))
    source = emit_c(commands, transport="rocket_rocc",
                    buffers=("sigmoid", "multiply"), symbol=symbol, internal=True)
    return ResidentKernel(source, tuple(commands),
                          tuple(edge.source_row + start for start in starts),
                          temp, result, multiply,
                          edge.physical_rows * edge.row_bytes,
                          sum(isinstance(c, Command) and c.funct == 33 for c in commands),
                          sum(isinstance(c, Command) and c.funct == 3 for c in commands), _sha(source))


def emit_resident_binary(profile: dict, *, lhs: ResidentTensor | DeviceEdgePlan,
                         rhs: ResidentTensor | DeviceEdgePlan, operation: str,
                         output_row: int, symbol: str) -> ResidentKernel:
    """Read two proven BF16 row layouts, write a resident result and BF16 readout."""
    if operation not in {"add", "sub", "mul"}:
        raise DeviceEdgeError("resident binary operation is unqualified")
    lhs = matrix_resident_tensor(lhs) if isinstance(lhs, DeviceEdgePlan) else lhs
    rhs = matrix_resident_tensor(rhs) if isinstance(rhs, DeviceEdgePlan) else rhs
    if (not isinstance(lhs, ResidentTensor) or not isinstance(rhs, ResidentTensor) or
            lhs.dtype != "bf16" or rhs.dtype != "bf16" or
            not lhs.producer_source_sha256 or not rhs.producer_source_sha256):
        raise DeviceEdgeError("resident binary needs two source-bound BF16 values")
    if (lhs.profile_sha256 != rhs.profile_sha256 or
            lhs.logical_shape != rhs.logical_shape or
            lhs.padded_shape != rhs.padded_shape or
            lhs.row_bytes != rhs.row_bytes or lhs.physical_rows != rhs.physical_rows):
        raise DeviceEdgeError("resident binary operands have different physical BF16 layouts")
    if lhs.profile_sha256 != profile_sha256(profile):
        raise DeviceEdgeError("resident binary profile changed")
    total = source_bound_capacities(profile)["scratchpad"]
    spans = ((lhs.source_row, lhs.source_row + lhs.physical_rows),
             (rhs.source_row, rhs.source_row + rhs.physical_rows),
             (output_row, output_row + lhs.physical_rows))
    if any(lo < 0 or hi > total for lo, hi in spans) or any(
            max(spans[i][0], spans[j][0]) < min(spans[i][1], spans[j][1])
            for i in range(3) for j in range(i + 1, 3)):
        raise DeviceEdgeError("resident binary input/output rows overlap or exceed capacity")
    commands: list[Command | Fence] = []
    starts = tuple(range(0, lhs.physical_rows, 1024))
    for start in starts:
        commands.append(vpu_command(profile, kind=operation,
                                    src1_row=lhs.source_row + start,
                                    src2_row=rhs.source_row + start,
                                    dst_row=output_row + start,
                                    rows=min(1024, lhs.physical_rows - start)))
        commands.append(Fence())
    commands.extend(_readout(profile, buffer="result", source_row=output_row,
                             physical_rows=lhs.physical_rows, row_bytes=lhs.row_bytes))
    source = emit_c(commands, transport="rocket_rocc", buffers=("result",),
                    symbol=symbol, internal=True)
    return ResidentKernel(source, tuple(commands),
                          tuple(lhs.source_row + start for start in starts),
                          0, output_row, None,
                          lhs.physical_rows * lhs.row_bytes,
                          len(starts), sum(isinstance(c, Command) and c.funct == 3
                                           for c in commands), _sha(source))
