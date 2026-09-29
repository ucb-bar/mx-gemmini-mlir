"""Emit a deliberately bounded source-bound RTL bringup program.

This is a 32x32x32 diagnostic, not the Merlin backend. Its command order is
derived from the selected Gemmini bare-metal tests and has been checked only
for one full scale window per format.
"""

from __future__ import annotations

from collections.abc import Sequence

from .contraction import MxContractionPayload


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


def _c_expected(values: Sequence[Sequence[int]]) -> str:
    if len(values) != 32 or any(len(row) != 32 for row in values):
        raise ValueError("diagnostic BF16 expected matrix must be 32x32")
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
    scheduling is inferred from this one-window program.
    """
    if payload.fmt not in ("mxfp8", "mxfp6", "mxfp4"):
        raise ValueError("selected diagnostic supports MXFP8, MXFP6, MXFP4")
    if (payload.m, payload.k, payload.n) != (32, 32, 32) or len(payload.waves) != 1:
        raise ValueError("selected diagnostic requires one 32x32x32 scale window")
    wave = payload.waves[0]
    expected_operand_bytes = 1024 if payload.fmt == "mxfp8" else 512
    if (len(wave.activation_bytes), len(wave.weight_bytes)) != (
        expected_operand_bytes, expected_operand_bytes
    ):
        raise ValueError("packed operand lengths disagree with selected shape")
    if (len(wave.activation_scale_bytes), len(wave.weight_scale_bytes)) != (32, 32):
        raise ValueError("packed E8M0 lengths disagree with selected shape")
    is_fp6 = payload.fmt == "mxfp6"
    if is_fp6:
        if payload.lut_lines_per_operand != (16, 16) or payload.lut_granularity_shift != 1:
            raise ValueError("selected FP6 diagnostic requires 16 LUT lines and shift one")
        if (len(payload.activation_lut_bytes), len(payload.weight_lut_bytes)) != (192, 192):
            raise ValueError("selected FP6 diagnostic requires 192 packed LUT bytes per operand")
    elif payload.activation_lut_bytes or payload.weight_lut_bytes:
        raise ValueError("direct MX diagnostic must not carry LUT bytes")

    format_code = {"mxfp8": 0, "mxfp6": 1, "mxfp4": 2}[payload.fmt]
    tiles_i = tiles_j = 2 if payload.fmt == "mxfp8" else 1
    c_base = 64 if payload.fmt == "mxfp8" else 128
    arrays = (
        _c_bytes("A_in", wave.activation_bytes)
        + _c_bytes("B_in", wave.weight_bytes)
        + _c_bytes("A_scales", wave.activation_scale_bytes)
        + _c_bytes("B_scales", wave.weight_scale_bytes)
        + _c_expected(expected_bf16)
    )
    if is_fp6:
        arrays += _c_bytes("A_lut", payload.activation_lut_bytes)
        arrays += _c_bytes("B_lut", payload.weight_lut_bytes)
    lut_setup = (
        "  gemmini_mx_load_lut_dt((uint64_t)B_lut, 16, 0, 6);\n"
        "  gemmini_mx_load_lut_dt((uint64_t)A_lut, 16, 1, 6);\n"
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
        "#define M 32\n#define K 32\n#define N 32\n#define DIM 16\n"
        + arrays
        + "static uint64_t C_hw[M][N/4] __attribute__((aligned(64)));\n"
        + "static uint32_t output_scales[512] __attribute__((aligned(64)));\n"
        + "int main(void) {\n"
        + f"  const int tiles_I = {tiles_i}, tiles_J = {tiles_j}, tiles_K = 2;\n"
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
        + f'  printf("generated {payload.fmt.upper()} 32x32x32: %d mismatches\\n", errors);\n'
        + "  return errors != 0;\n}\n"
    )
