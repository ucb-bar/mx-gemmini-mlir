"""Emit a deliberately bounded source-bound RTL bringup program.

This is a bounded square-matrix diagnostic, not the Merlin backend. Its command
order is derived from the selected Gemmini bare-metal tests. Single-window
programs at sizes 32 and 64 have RTL checks; the serial independent-batch
variant currently has a Spike diagnostic only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .contraction import MxContractionPayload
from .model2mlir import IndexedMxPayload, IndexedTiledMxPayload


def _c_bytes(name: str, values: bytes) -> str:
    rows = [
        ", ".join(f"0x{value:02x}" for value in values[i : i + 16])
        for i in range(0, len(values), 16)
    ]
    return (
        f"static const uint8_t {name}[] __attribute__((aligned(64))) = {{\n"
        + ",\n".join("  " + row for row in rows)
        + "\n};\n"
    )


def _c_expected(values: Sequence[Sequence[int]], size: int) -> str:
    if len(values) != size or any(len(row) != size for row in values):
        raise ValueError(f"diagnostic BF16 expected matrix must be {size}x{size}")
    if any(type(value) is not int or not 0 <= value <= 0xffff for row in values for value in row):
        raise ValueError("diagnostic BF16 expected matrix needs 16-bit bit patterns")
    rows = ["  {" + ", ".join(f"0x{value:04x}" for value in row) + "}" for row in values]
    return "static const uint16_t expected[M][N] = {\n" + ",\n".join(rows) + "\n};\n"


def emit_single_window_baremetal_c(
    payload: MxContractionPayload, expected_bf16: Sequence[Sequence[int]]
) -> str:
    """Render one pinned-shape test from packed A/B, E8M0, and optional LUT bytes.

    The output is for bare-metal RTL diagnostics with the selected Gemmini
    header and ``MX_ROCKET``. No arbitrary shape, tail, batch, or K-wave
    scheduling is inferred from these two square one-window programs.
    """
    if payload.fmt not in ("mxfp8", "mxfp6", "mxfp4"):
        raise ValueError("selected diagnostic supports MXFP8, MXFP6, MXFP4")
    if payload.m not in (32, 64) or (payload.m, payload.k, payload.n) != (payload.m,) * 3 or len(payload.waves) != 1:
        raise ValueError("selected diagnostic requires one square 32 or 64 scale window")
    wave = payload.waves[0]
    expected_operand_bytes = payload.m * payload.k if payload.fmt == "mxfp8" else payload.m * payload.k // 2
    if (len(wave.activation_bytes), len(wave.weight_bytes)) != (
        expected_operand_bytes, expected_operand_bytes
    ):
        raise ValueError("packed operand lengths disagree with selected shape")
    expected_scale_bytes = payload.m * payload.k // 32
    if (len(wave.activation_scale_bytes), len(wave.weight_scale_bytes)) != (expected_scale_bytes,) * 2:
        raise ValueError("packed E8M0 lengths disagree with selected shape")
    is_fp6 = payload.fmt == "mxfp6"
    if is_fp6:
        lut_lines = payload.m // 2
        if payload.lut_lines_per_operand != (lut_lines,) * 2 or payload.lut_granularity_shift != 1:
            raise ValueError("selected FP6 diagnostic requires half-size LUT lines and shift one")
        if (len(payload.activation_lut_bytes), len(payload.weight_lut_bytes)) != (12 * lut_lines,) * 2:
            raise ValueError("selected FP6 diagnostic requires 12 packed bytes per LUT line")
    elif payload.activation_lut_bytes or payload.weight_lut_bytes:
        raise ValueError("direct MX diagnostic must not carry LUT bytes")

    format_code = {"mxfp8": 0, "mxfp6": 1, "mxfp4": 2}[payload.fmt]
    tiles_i = tiles_j = payload.m // (16 if payload.fmt == "mxfp8" else 32)
    tiles_k = payload.k // 16
    c_base = tiles_i * tiles_k * 16 if payload.fmt == "mxfp8" else 128
    arrays = (
        _c_bytes("A_in", wave.activation_bytes)
        + _c_bytes("B_in", wave.weight_bytes)
        + _c_bytes("A_scales", wave.activation_scale_bytes)
        + _c_bytes("B_scales", wave.weight_scale_bytes)
        + _c_expected(expected_bf16, payload.m)
    )
    if is_fp6:
        arrays += _c_bytes("A_lut", payload.activation_lut_bytes)
        arrays += _c_bytes("B_lut", payload.weight_lut_bytes)
    lut_setup = (
        f"  gemmini_mx_load_lut_dt((uint64_t)B_lut, {payload.n // 2}, 0, 6);\n"
        f"  gemmini_mx_load_lut_dt((uint64_t)A_lut, {payload.m // 2}, 1, 6);\n"
        if is_fp6 else "  gemmini_mx_lut_disable();\n"
    )
    b_load = (
        "  gemmini_config_ld(N);\n"
        "  for (int j = 0; j < tiles_J; j++)\n"
        "    for (int k = 0; k < tiles_K; k++)\n"
        "      gemmini_extended_mvin((void*)(B_in + j*DIM*K + k*DIM),\n"
        "          b_base + (j*tiles_K+k)*DIM, DIM, DIM);\n"
        if payload.fmt == "mxfp8" else
        "  gemmini_config_ld(N/2);\n"
        "  for (int k = 0; k < tiles_K; k++)\n"
        "    for (int j = 0; j < tiles_J; j++)\n"
        "      gemmini_extended_mvin((void*)(B_in + k*DIM*(N/2) + j*DIM),\n"
        "          b_base + (k*tiles_J+j)*DIM, DIM, DIM);\n"
    )
    return (
        "#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n"
        '#include "include/gemmini_testutils.h"\n'
        + f"#define M {payload.m}\n#define K {payload.k}\n#define N {payload.n}\n#define DIM 16\n"
        + arrays
        + "static uint64_t C_hw[M][N/4] __attribute__((aligned(64)));\n"
        + "static uint32_t output_scales[512] __attribute__((aligned(64)));\n"
        + "int main(void) {\n"
        + f"  const int tiles_I = {tiles_i}, tiles_J = {tiles_j}, tiles_K = {tiles_k};\n"
        + "  const uint32_t a_base = 0;\n"
        + "  const uint32_t b_base = BANK_NUM * BANK_ROWS - tiles_K * tiles_J * DIM;\n"
        + f"  const uint32_t c_base = {c_base};\n"
        + "  memset(C_hw, 0, sizeof C_hw);\n"
        + "  gemmini_flush(0);\n"
        + "  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,\n"
        + f"      1, 1, 0, 0, false, {format_code}, {format_code}, 3, {int(is_fp6)});\n"
        + lut_setup
        + "  gemmini_mx_load_scales((uint64_t)A_scales, sizeof A_scales, 0);\n"
        + "  gemmini_mx_load_scales((uint64_t)B_scales, sizeof B_scales, 1);\n"
        + "  gemmini_fence();\n"
        + "  gemmini_config_ld(K);\n"
        + "  for (int i = 0; i < tiles_I; i++)\n"
        + "    for (int k = 0; k < tiles_K; k++)\n"
        + "      gemmini_extended_mvin((void*)(A_in + i*DIM*K + k*DIM),\n"
        + "          a_base + (i*tiles_K+k)*DIM, DIM, DIM);\n"
        + b_load
        + "  gemmini_config_st((N/4) * sizeof(uint64_t));\n"
        + "  gemmini_mxquant_config_mvout((uint64_t)output_scales,\n"
        + "      tiles_I, tiles_J, tiles_K, 0, 0, 1);\n"
        + "  gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K,\n"
        + "      0, 0, 0, a_base, BANK_NUM * BANK_ROWS, 0, c_base,\n"
        + "      false, false, false, false, false, NO_ACTIVATION,\n"
        + "      0, 0, false, 0x38);\n"
        + "  gemmini_fence();\n"
        + "  gemmini_config_st(DIM);\n"
        + "  uint8_t *c_bytes = (uint8_t*)C_hw;\n"
        + "  const int spad_rows = M*N*2/DIM;\n"
        + "  for (int row = 0; row < spad_rows; row += DIM)\n"
        + "    gemmini_extended_mvout(c_bytes + row*DIM, c_base + row, DIM, DIM);\n"
        + "  gemmini_fence();\n"
        + "  int errors = 0;\n"
        + "  for (int i = 0; i < M; i++)\n"
        + "    for (int j = 0; j < N; j++) {\n"
        + "      uint16_t got = (C_hw[i][j/4] >> (16*(j%4))) & 0xffff;\n"
        + "      if (got != expected[i][j]) {\n"
        + '        if (errors < 8) printf("MISMATCH (%d,%d): got=0x%04x exp=0x%04x\\n", i, j, got, expected[i][j]);\n'
        + "        errors++;\n"
        + "      }\n"
        + "    }\n"
        + f'  printf("generated {payload.fmt.upper()} {payload.m}x{payload.k}x{payload.n}: %d mismatches\\n", errors);\n'
        + "  return errors != 0;\n}\n"
    )


def emit_two_wave_baremetal_c(
    payload: MxContractionPayload, expected_bf16: Sequence[Sequence[int]]
) -> str:
    """Render the pinned 32x32x64 two-wave accumulation diagnostic.

    This reuses the tested one-wave command body twice. The second body reloads
    both E8M0 banks and operand tiles, then sets ``ex_accumulate`` while keeping
    the same C destination. FP6 LUT contents remain resident across both waves.
    """
    if (payload.m, payload.k, payload.n) != (32, 64, 32) or len(payload.waves) != 2:
        raise ValueError("two-wave diagnostic requires a 32x32x64 contraction")
    if [(w.wave.block_start, w.wave.block_stop) for w in payload.waves] != [(0, 1), (1, 2)]:
        raise ValueError("two-wave diagnostic requires consecutive 32-element K waves")
    first, second = payload.waves
    if (len(second.activation_bytes), len(second.weight_bytes)) != (
        len(first.activation_bytes), len(first.weight_bytes)
    ) or (len(second.activation_scale_bytes), len(second.weight_scale_bytes)) != (32, 32):
        raise ValueError("second wave payload lengths disagree with selected shape")

    first_payload = replace(payload, k=32, waves=(first,))
    source = emit_single_window_baremetal_c(first_payload, expected_bf16)
    for old, new in (
        ("A_in", "A0"), ("B_in", "B0"),
        ("A_scales", "AS0"), ("B_scales", "BS0"),
    ):
        source = source.replace(old, new)
    arrays = (
        _c_bytes("A1", second.activation_bytes)
        + _c_bytes("B1", second.weight_bytes)
        + _c_bytes("AS1", second.activation_scale_bytes)
        + _c_bytes("BS1", second.weight_scale_bytes)
    )
    marker = "static uint64_t C_hw"
    if source.count(marker) != 1:
        raise AssertionError("single-window C template changed before second-wave insertion")
    source = source.replace(marker, arrays + marker, 1)
    start_marker = "  gemmini_mx_load_scales((uint64_t)AS0"
    end_marker = "  gemmini_config_st(DIM);"
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise AssertionError("single-window C command template changed")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    second_commands = source[start:end]
    for old, new in (("AS0", "AS1"), ("BS0", "BS1"), ("A0", "A1"), ("B0", "B1")):
        second_commands = second_commands.replace(old, new)
    accumulation_flag = "false, false, false, false, false, NO_ACTIVATION"
    if second_commands.count(accumulation_flag) != 1:
        raise AssertionError("single-window accumulation flag template changed")
    second_commands = second_commands.replace(
        accumulation_flag, "false, false, false, false, true, NO_ACTIVATION"
    )
    source = (
        source[:end]
        + "  // Second K wave accumulates into the same C destination.\n"
        + second_commands
        + source[end:]
    )
    label = f"generated {payload.fmt.upper()} 32x32x32"
    if source.count(label) != 1:
        raise AssertionError("single-window diagnostic label changed")
    return source.replace(label, f"two-wave {payload.fmt.upper()} 32x32x64")


def emit_independent_batches_baremetal_c(
    cases: Sequence[tuple[IndexedMxPayload, Sequence[Sequence[int]]]],
) -> str:
    """Render independent MX contractions as successive bounded diagnostics.

    Each batch runs the source-checked one-window command sequence with its
    own buffers and output. This does not schedule a fused attention kernel.
    """
    if not cases or len(cases) > 4:
        raise ValueError("batched diagnostic needs one to four independent contractions")
    indices = [indexed.batch_index for indexed, _ in cases]
    if len(set(indices)) != len(indices) or any(not index for index in indices):
        raise ValueError("batched diagnostic needs unique, nonempty batch indices")
    shape_and_format = {
        (indexed.payload.fmt, indexed.payload.m, indexed.payload.k, indexed.payload.n)
        for indexed, _ in cases
    }
    if len(shape_and_format) != 1:
        raise ValueError("batched diagnostic needs matching formats and shapes")

    header: str | None = None
    functions: list[str] = []
    for ordinal, (indexed, expected) in enumerate(cases):
        source = emit_single_window_baremetal_c(indexed.payload, expected)
        marker = "static const uint8_t A_in[]"
        if source.count(marker) != 1 or source.count("int main(void) {") != 1:
            raise AssertionError("single-window C template changed")
        prefix, body = source.split(marker, 1)
        if header is None:
            header = prefix
        elif prefix != header:
            raise ValueError("batched diagnostic needs matching C configuration")
        body = marker + body
        for symbol in (
            "A_in", "B_in", "A_scales", "B_scales", "A_lut", "B_lut",
            "expected", "C_hw", "output_scales",
        ):
            body = body.replace(symbol, f"{symbol}_{ordinal}")
        body = body.replace("int main(void) {", f"static int run_batch_{ordinal}(void) {{", 1)
        label = f"generated {indexed.payload.fmt.upper()} {indexed.payload.m}x{indexed.payload.k}x{indexed.payload.n}"
        if body.count(label) != 1:
            raise AssertionError("single-window diagnostic label changed")
        body = body.replace(label, f"batch {indexed.batch_index} {label}", 1)
        functions.append(body)
    assert header is not None
    return (
        header
        + "\n".join(functions)
        + "int main(void) {\n"
        + "  int errors = 0;\n"
        + "".join(f"  errors += run_batch_{index}();\n" for index in range(len(cases)))
        + "  return errors != 0;\n}\n"
    )


def emit_spatial_tiles_baremetal_c(
    tiles: Sequence[IndexedTiledMxPayload],
    expected_bf16: Sequence[Sequence[int]],
) -> str:
    """Check every 32x32 tile of one bounded rank-2 contraction in one ELF.

    The same one-window command body executes each spatial tile successively.
    This checks tile offsets and expected output slices; it does not implement
    a general runtime schedule or copy the pieces into a combined output.
    """
    if not tiles or len(tiles) > 4 or any(tile.batch_index for tile in tiles):
        raise ValueError("spatial diagnostic needs one to four rank-2 tiles")
    m = max(tile.m_stop for tile in tiles)
    n = max(tile.n_stop for tile in tiles)
    expected_origins = {(row, col) for row in range(0, m, 32) for col in range(0, n, 32)}
    origins = {(tile.m_start, tile.n_start) for tile in tiles}
    if (m not in (32, 64) or n not in (32, 64)
            or len(origins) != len(tiles) or origins != expected_origins
            or any((tile.m_stop - tile.m_start, tile.n_stop - tile.n_start) != (32, 32)
                   or (tile.payload.m, tile.payload.n, tile.payload.k) != (32, 32, 32)
                   for tile in tiles)):
        raise ValueError("spatial diagnostic needs a complete 32x32 tile grid")
    if len(expected_bf16) != m or any(len(row) != n for row in expected_bf16):
        raise ValueError("spatial diagnostic BF16 expected matrix has wrong shape")
    cases = [
        (
            IndexedMxPayload((tile.m_start, tile.n_start), tile.payload),
            [row[tile.n_start:tile.n_stop] for row in expected_bf16[tile.m_start:tile.m_stop]],
        )
        for tile in sorted(tiles, key=lambda tile: (tile.m_start, tile.n_start))
    ]
    source = emit_independent_batches_baremetal_c(cases)
    label = "batch ("
    if source.count(label) != len(cases):
        raise AssertionError("serial C diagnostic label changed")
    return source.replace(label, "tile (")
