"""Specialized private row packing agrees with the canonical scalar quantizer."""

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mx_gemmini_support.fp8_row_packing import emit_fp8_row_packer

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("rows,width", [(16, 32), (160, 160), (160, 2048)])
def test_streamed_rows_preserve_all_codes_scales_and_untouched_rows(
    tmp_path, rows, width
):
    source = tmp_path / "pack.c"
    source.write_text(
        emit_fp8_row_packer(rows=rows, width=width, symbol="private_pack")
        + "\nint pack(const float *x,unsigned first,unsigned count,uint8_t *c,uint8_t *s){return private_pack(x,first,count,c,s);}\n"
    )
    library = tmp_path / "pack.so"
    reference = tmp_path / "reference.so"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(ROOT / "mx_gemmini_support"),
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-DMX_QUANT_REFERENCE",
            str(ROOT / "mx_gemmini_support/runtime_quant.c"),
            "-o",
            str(reference),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(library))
    ref = ctypes.CDLL(str(reference))
    dll.pack.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    ref.mx_pack_activation_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    rng = np.random.default_rng(641)
    values = np.ldexp(
        rng.normal(size=(rows, width)).astype(np.float32),
        rng.integers(-120, 100, (rows, width)),
    ).astype(np.float32)
    values[0] = 0
    values[1] = np.float32(-0.0)
    # BF16 ties on both parities, minimum-scale groups, and FP8 code ties.
    ties = np.array(
        [
            0x3F808000,
            0x3F818000,
            0xBF808000,
            0xBF818000,
            0x00000001,
            0x80000001,
            0x39800000,
            0xB9800000,
        ],
        np.uint32,
    ).view(np.float32)
    values[2] = np.resize(ties, width)
    expected_codes = np.empty(rows * width, np.uint8)
    expected_scales = np.empty(rows * width // 32, np.uint8)
    assert (
        ref.mx_pack_activation_f32(
            values.ctypes.data,
            rows,
            width,
            8,
            expected_codes.ctypes.data,
            expected_scales.ctypes.data,
        )
        == 0
    )
    codes = np.full(expected_codes.size + 128, 0xA5, np.uint8)
    scales = np.full(expected_scales.size + 128, 0xA5, np.uint8)
    want_codes = np.full((rows, width), 0xA5, np.uint8)
    want_scales = np.full((width // 32, rows), 0xA5, np.uint8)
    for first in range(0, rows, 51):
        count = min(rows - first, 51)
        tile = values[first : first + count].copy()
        assert (
            dll.pack(
                tile.ctypes.data,
                first,
                count,
                codes[64:].ctypes.data,
                scales[64:].ctypes.data,
            )
            == 0
        )
        want_codes[first : first + count] = expected_codes.reshape(rows, width)[
            first : first + count
        ]
        want_scales[:, first : first + count] = expected_scales.reshape(
            width // 32, rows
        )[:, first : first + count]
        assert np.array_equal(codes[64:-64], want_codes.ravel())
        assert np.array_equal(scales[64:-64], want_scales.ravel())
        assert np.all(codes[:64] == 0xA5) and np.all(codes[-64:] == 0xA5)
        assert np.all(scales[:64] == 0xA5) and np.all(scales[-64:] == 0xA5)
    for first, count in ((rows, 1), (0, 0), (rows - 1, 2)):
        before = (codes.copy(), scales.copy())
        assert (
            dll.pack(
                values.ctypes.data,
                first,
                count,
                codes[64:].ctypes.data,
                scales[64:].ctypes.data,
            )
            == -1
        )
        assert np.array_equal(codes, before[0]) and np.array_equal(scales, before[1])


def test_invalid_static_extents_or_symbols_refuse():
    for rows, width, symbol in (
        (True, 32, "pack"),
        (16, 16, "pack"),
        (17, 32, "pack"),
        (16, 32, "bad-name"),
    ):
        with pytest.raises(ValueError):
            emit_fp8_row_packer(rows=rows, width=width, symbol=symbol)
