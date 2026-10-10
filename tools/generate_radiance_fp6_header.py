"""Generate missing Radiance FP6 headers from its checked-in 2048-K data.

This is a source-fixture generator, not a compiler lowering. It slices the
checked-in packed inputs and LUTs, then invokes Radiance's own mx_golden for
the selected K. A full-size control run must reproduce the checked-in C
projection, scales, and BF16 output before any new header is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from mx_gemmini_support.source_fp6 import _array, _bytes


M = N = 128
FULL_K = 2048
LUT_LINES = 64


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(header: str, name: str, ctype: str, dimensions: str,
          count: int, width: int) -> bytes:
    return _bytes(_array(header, name=name, ctype=ctype, dimensions=dimensions,
                         count=count, maximum=(1 << (width * 8)) - 1), width)


def _unpack_lut(packed: bytes) -> bytes:
    if len(packed) != LUT_LINES * 12:
        raise ValueError("source FP6 LUT bank has the wrong size")
    return bytes((int.from_bytes(packed[row * 12:(row + 1) * 12], "little")
                  >> (6 * code)) & 63
                 for row in range(LUT_LINES) for code in range(16))


def _run_golden(executable: Path, directory: Path, k: int,
                data: dict[str, bytes]) -> tuple[bytes, bytes, bytes]:
    for name, content in data.items():
        (directory / f"{name}.bin").write_bytes(content)
    common = [str(executable), str(M), str(N), str(k),
              *(str(directory / f"{name}.bin") for name in ("a", "b", "sa", "sb"))]
    subprocess.run([*common, str(directory / "bf16.bin"), "1",
                    str(directory / "la.bin"), str(directory / "lb.bin"), "1"],
                   check=True, env={"PATH": os.environ.get("PATH", "")})
    environment = dict(os.environ)
    environment.update(MX_OUT_FMT="1", MX_SCALES_OUT=str(directory / "scales.bin"),
                       MX_LUT_C=str(directory / "lc.bin"), MX_G="1")
    subprocess.run([*common, str(directory / "projection.bin"), "1",
                    str(directory / "la.bin"), str(directory / "lb.bin"), "1"],
                   check=True, env=environment)
    bf16 = (directory / "bf16.bin").read_bytes()
    projection = (directory / "projection.bin").read_bytes()[:M * N // 2]
    scales = (directory / "scales.bin").read_bytes()
    if (len(bf16), len(projection), len(scales)) != (M * N * 2, M * N // 2,
                                                     M * N // 32):
        raise ValueError("Radiance FP6 golden output sizes differ")
    return bf16, projection, scales


def _emit(lines: list[str], ctype: str, name: str, dimensions: str, data: bytes,
          rows: int, columns: int, width: int) -> None:
    if len(data) != rows * columns * width:
        raise ValueError(f"{name}: generated array size differs")
    digits = width * 2
    lines.append(f"static const {ctype} {name}{dimensions} = {{")
    for row in range(rows):
        values = (int.from_bytes(data[(row * columns + col) * width:
                                      (row * columns + col + 1) * width], "little")
                  for col in range(columns))
        lines.append("    { " + ", ".join(f"0x{value:0{digits}x}" for value in values) +
                     " },")
    lines.append("};")
    lines.append("")


def generate(radiance_root: Path, k: int, output: Path) -> dict:
    if k not in {128, 256, 512, 1024}:
        raise ValueError("only the source-derived FP6 128/256/512/1024 K shapes are supported")
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    header_path = (radiance_root / "kernels/gemm_mxgemmini" /
                   "mxgemm.data.fp6.m128n128k2048.h")
    golden = radiance_root / "lib/golden/mx_golden"
    if not golden.is_file():
        raise ValueError("Radiance's mx_golden must be built in lib/golden")
    full_header = header_path.read_bytes()
    source = full_header.decode("ascii")
    if any(f"#define MATMUL_{axis}   {value}" not in source
           for axis, value in (("M", M), ("N", N), ("K", FULL_K))):
        raise ValueError("selected FP6 source header has unexpected dimensions")
    full = {
        "a": _read(source, "A_in_hw", "uint8_t", "[64][2048]", M * FULL_K // 2, 1),
        "b": _read(source, "B_in", "uint8_t", "[MATMUL_K][MATMUL_N / 2]",
                   FULL_K * N // 2, 1),
        "sa": _read(source, "A_scales_row", "uint8_t", "[MATMUL_GK][MATMUL_M]",
                    FULL_K // 32 * M, 1),
        "sb": _read(source, "B_scales_col", "uint8_t", "[MATMUL_GK][MATMUL_N]",
                    FULL_K // 32 * N, 1),
    }
    lut_packed = {
        short: _read(source, f"{long}_lut", "uint32_t", "[64][3]", 64 * 3, 4)
        for short, long in (("la", "A"), ("lb", "B"), ("lc", "C"))
    }
    full.update({name: _unpack_lut(data) for name, data in lut_packed.items()})
    expected_bf16 = _read(source, "C_out_bf16", "uint16_t",
                          "[MATMUL_M][MATMUL_N]", M * N, 2)
    expected_proj = _read(source, "C_proj_hw", "uint8_t", "[64][128]",
                          M * N // 2, 1)
    expected_packed_n = _read(source, "C_out", "uint8_t",
                              "[MATMUL_M][MATMUL_N / 2]", M * N // 2, 1)
    expected_scales = _read(source, "C_scales_row", "uint8_t",
                            "[MATMUL_GN][MATMUL_M]", M * N // 32, 1)

    with TemporaryDirectory(prefix="mx-fp6-header-") as tmp:
        temp = Path(tmp)
        control_bf16, control_proj, control_scales = _run_golden(
            golden, temp, FULL_K, full)
        control_group_major = bytes(
            control_scales[row * (N // 32) + group]
            for group in range(N // 32) for row in range(M))
        if ((control_bf16, control_proj, control_group_major) !=
                (expected_bf16, expected_proj, expected_scales)):
            raise ValueError("Radiance mx_golden differs from its checked-in FP6 header")
        control_packed_n = bytearray(M * N // 2)
        for row in range(M):
            for col in range(N):
                code = (control_proj[(row // 2) * N + col] >> (4 * (row & 1))) & 15
                control_packed_n[row * (N // 2) + col // 2] |= code << (4 * (col & 1))
        if bytes(control_packed_n) != expected_packed_n:
            raise ValueError("checked-in Radiance FP6 C_out differs from its C projection")
        selected = {
            "a": b"".join(full["a"][pair * FULL_K:pair * FULL_K + k]
                          for pair in range(M // 2)),
            "b": full["b"][:k * N // 2],
            "sa": full["sa"][:k // 32 * M],
            "sb": full["sb"][:k // 32 * N],
            "la": full["la"], "lb": full["lb"], "lc": full["lc"],
        }
        bf16, projection, scales = _run_golden(golden, temp, k, selected)

    packed_n = bytearray(M * N // 2)
    for row in range(M):
        for col in range(N):
            code = (projection[(row // 2) * N + col] >> (4 * (row & 1))) & 15
            packed_n[row * (N // 2) + col // 2] |= code << (4 * (col & 1))
    group_major = bytes(scales[row * (N // 32) + group]
                        for group in range(N // 32) for row in range(M))
    lines = [
        f"// Generated from {header_path.name} with Radiance mx_golden.",
        f"// Source header SHA-256: {_sha(full_header)}",
        f"#ifndef MXGEMM_DATA_FP6_M128N128K{k}_H",
        f"#define MXGEMM_DATA_FP6_M128N128K{k}_H",
        "",
        "#include <stdint.h>",
        "",
        "#define MATMUL_M   128",
        f"#define MATMUL_K   {k}",
        "#define MATMUL_N   128",
        f"#define MATMUL_GK  {k // 32}",
        "#define MATMUL_GN  4",
        "#define A_TILE_M   32",
        "#define K_TILE     16",
        "",
    ]
    _emit(lines, "uint8_t", "A_in_hw", f"[64][{k}]", selected["a"], 64, k, 1)
    _emit(lines, "uint8_t", "B_in", "[MATMUL_K][MATMUL_N / 2]",
          selected["b"], k, N // 2, 1)
    for short, long in (("la", "A"), ("lb", "B"), ("lc", "C")):
        _emit(lines, "uint32_t", f"{long}_lut", "[64][3]",
              lut_packed[short], 64, 3, 4)
    _emit(lines, "uint8_t", "A_scales_row", "[MATMUL_GK][MATMUL_M]",
          selected["sa"], k // 32, M, 1)
    _emit(lines, "uint8_t", "B_scales_col", "[MATMUL_GK][MATMUL_N]",
          selected["sb"], k // 32, N, 1)
    _emit(lines, "uint8_t", "C_out", "[MATMUL_M][MATMUL_N / 2]",
          bytes(packed_n), M, N // 2, 1)
    _emit(lines, "uint8_t", "C_scales_row", "[MATMUL_GN][MATMUL_M]",
          group_major, N // 32, M, 1)
    _emit(lines, "uint16_t", "C_out_bf16", "[MATMUL_M][MATMUL_N]",
          bf16, M, N, 2)
    _emit(lines, "uint8_t", "C_proj_hw", "[64][128]",
          projection, M // 2, N, 1)
    lines.extend([f"#endif // MXGEMM_DATA_FP6_M128N128K{k}_H", ""])
    output.write_text("\n".join(lines), encoding="ascii")
    return {"schema": "mx_gemmini.generated_radiance_fp6_fixture.v1",
            "k": k, "source_header_sha256": _sha(full_header),
            "golden_source_sha256": _sha((radiance_root / "lib/golden/mx_golden.cpp").read_bytes()),
            "golden_binary_sha256": _sha(golden.read_bytes()),
            "generated_header_sha256": _sha(output.read_bytes()),
            "bf16_sha256": _sha(bf16),
            "packed_output_sha256": _sha(projection),
            "scale_sha256": _sha(group_major)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radiance-root", required=True, type=Path)
    parser.add_argument("--k", required=True, type=int, choices=(128, 256, 512, 1024))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.radiance_root, args.k, args.out),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
