"""C operand packing matches MXQuant's pinned BF16/E8M0 encoder bytes."""

from __future__ import annotations

import ctypes
import hashlib
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "mx_gemmini_support/runtime_quant.c"
# MXQuant b4af5430, functional_contraction_operands followed by
# pack_direct_operands, for the deterministic f32 matrices below. Both scale
# outputs contain several exponent values and a zero-block code (104).
REFERENCE = {
    8: (
        "3b0f28ff90d02993064a000dc29066cb2a6f2296884b8757c8fc6e09954a0fc2",
        "8c3d2ba0b18bc648c805904001da7dc81cf9dd85121c0be72f8f2455c04d8311",
        "f551d42bd31965d6fa43eafcca5b35fe632861ec7df56dad41dc0b2319f8a84d",
        "f2902908105e226744d919a74044b9e141591418f49287da79431a630791ca4c",
    ),
    4: (
        "5fd6c38a8581fb44dd6286cc064a97dd93ea10eacb11309e0f3c0c6780264013",
        "8c3d2ba0b18bc648c805904001da7dc81cf9dd85121c0be72f8f2455c04d8311",
        "90888ec008fcf9da68b8d9a8b686acd1e5cdcc82e19ff6649e103171ca5f8631",
        "f2902908105e226744d919a74044b9e141591418f49287da79431a630791ca4c",
    ),
}


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    output = tmp_path_factory.mktemp("mx-runtime-quant") / "libquant.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-std=c11", "-Wall", "-Wextra",
                    "-Werror", str(SOURCE), "-lm", "-o", str(output)], check=True)
    lib = ctypes.CDLL(str(output))
    for name in ("mx_pack_activation_f32", "mx_pack_weight_f32"):
        function = getattr(lib, name)
        function.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
                             ctypes.c_uint32, ctypes.c_uint32,
                             ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
        function.restype = ctypes.c_int
    lib.mx_pack_activation_zero_rows_f32.argtypes = [ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
    lib.mx_pack_activation_zero_rows_f32.restype = ctypes.c_int
    lib.mx_widen_bf16_f32.argtypes = [ctypes.POINTER(ctypes.c_uint16),
                                     ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
    lib.mx_widen_bf16_f32.restype = ctypes.c_int
    return lib


def _activation() -> list[float]:
    return [0.0 if row % 11 == 0 and col // 32 == 1 else
            (((row * 64 + col) * 37) % 211 - 105) / 17.0 *
            2.0 ** ((row % 7) - 3 + (col // 32) - 1)
            for row in range(32) for col in range(64)]


def _weight() -> list[float]:
    return [0.0 if col % 13 == 0 and row // 32 == 0 else
            (((row * 32 + col) * 53) % 223 - 111) / 19.0 *
            2.0 ** ((col % 9) - 4 + (row // 32))
            for row in range(64) for col in range(32)]


@pytest.mark.parametrize("format_bits", [8, 4])
def test_live_operand_bytes_match_mxquant(library, format_bits: int) -> None:
    expected = REFERENCE[format_bits]
    results = []
    for kind, values, rows, cols in (
        ("activation", _activation(), 32, 64),
        ("weight", _weight(), 64, 32),
    ):
        input_array = (ctypes.c_float * len(values))(*values)
        codes = (ctypes.c_uint8 * (rows * cols * format_bits // 8))()
        scales = (ctypes.c_uint8 * (rows * cols // 32))()
        function = getattr(library, f"mx_pack_{kind}_f32")
        assert function(input_array, rows, cols, format_bits, codes, scales) == 0
        results.extend((hashlib.sha256(bytes(codes)).hexdigest(),
                        hashlib.sha256(bytes(scales)).hexdigest()))
    assert tuple(results) == expected


def test_nonfinite_and_bad_shape_refuse(library) -> None:
    values = (ctypes.c_float * (32 * 32))(*([0.0] * (32 * 32)))
    values[17] = float("inf")
    codes = (ctypes.c_uint8 * 1024)()
    scales = (ctypes.c_uint8 * 32)()
    assert library.mx_pack_activation_f32(values, 32, 32, 8, codes, scales) == -1
    assert library.mx_pack_activation_f32(values, 31, 32, 8, codes, scales) == -1


def test_bf16_output_widens_without_changing_bits(library) -> None:
    words = (ctypes.c_uint16 * 4)(0x3f80, 0xbf80, 0x0000, 0x8000)
    result = (ctypes.c_float * 4)()
    assert library.mx_widen_bf16_f32(words, 4, result) == 0
    assert [struct.pack("<f", value) for value in result] == [
        b"\x00\x00\x80\x3f", b"\x00\x00\x80\xbf",
        b"\x00\x00\x00\x00", b"\x00\x00\x00\x80",
    ]


@pytest.mark.parametrize("format_bits", [4, 8])
@pytest.mark.parametrize("live_rows,k", [(1, 32), (1, 96), (2, 64), (3, 64), (15, 32),
                                       (17, 96), (19, 64), (31, 32), (32, 32), (33, 64)])
def test_implicit_zero_rows_match_explicit_padding_in_every_byte(library, format_bits, live_rows, k):
    import numpy as np

    tile = 16 if format_bits == 8 else 32
    m = (live_rows + tile - 1) // tile * tile
    values = np.random.default_rng(7123 + live_rows).normal(size=(live_rows, k)).astype(np.float32)
    values.flat[:8] = [0.0, -0.0, 1e-38, -1e-38, 1e30, -1e30, 3.3895314e38, -3.3895314e38]
    # The implicit input is exactly the live allocation, with no ghost rows.
    explicit = np.zeros((m, k), dtype=np.float32)
    explicit[:live_rows] = values
    size, scales_size = m * k * format_bits // 8, m * k // 32
    expected_codes = (ctypes.c_uint8 * size)(*([0xA5] * size))
    expected_scales = (ctypes.c_uint8 * scales_size)(*([0xA5] * scales_size))
    code_storage = (ctypes.c_uint8 * (size + 2))(*([0xA5] * (size + 2)))
    scale_storage = (ctypes.c_uint8 * (scales_size + 2))(*([0xA5] * (scales_size + 2)))
    codes = ctypes.cast(ctypes.byref(code_storage, 1), ctypes.POINTER(ctypes.c_uint8))
    scales = ctypes.cast(ctypes.byref(scale_storage, 1), ctypes.POINTER(ctypes.c_uint8))
    pointer = ctypes.POINTER(ctypes.c_float)
    assert library.mx_pack_activation_f32(explicit.ctypes.data_as(pointer), m, k, format_bits,
                                          expected_codes, expected_scales) == 0
    assert library.mx_pack_activation_zero_rows_f32(values.ctypes.data_as(pointer), live_rows, m, k,
                                                    format_bits, codes, scales) == 0
    assert bytes(code_storage)[1:-1] == bytes(expected_codes)
    assert bytes(scale_storage)[1:-1] == bytes(expected_scales)
    assert (code_storage[0], code_storage[-1], scale_storage[0], scale_storage[-1]) == (0xA5,) * 4


@pytest.mark.parametrize("format_bits", [4, 8])
@pytest.mark.parametrize("word", [0x7F800000, 0xFF800000, 0x7FC01234, 0x7F800001, 0x7F7FFFFF])
def test_implicit_zero_rows_preserve_refusal_and_partial_writes(library, format_bits, word):
    import numpy as np

    live_rows, m, k = 3, 32, 96
    values = np.ones((live_rows, k), dtype=np.float32)
    values.view(np.uint32)[2, 67] = word  # later group, after earlier ghost rows were written
    explicit = np.zeros((m, k), dtype=np.float32)
    explicit[:live_rows] = values
    results = []
    for implicit in (False, True):
        codes = (ctypes.c_uint8 * (m * k * format_bits // 8))(*([0xA5] * (m * k * format_bits // 8)))
        scales = (ctypes.c_uint8 * (m * k // 32))(*([0xA5] * (m * k // 32)))
        pointer = ctypes.POINTER(ctypes.c_float)
        if implicit:
            status = library.mx_pack_activation_zero_rows_f32(values.ctypes.data_as(pointer), live_rows,
                                                               m, k, format_bits, codes, scales)
        else:
            status = library.mx_pack_activation_f32(explicit.ctypes.data_as(pointer), m, k,
                                                    format_bits, codes, scales)
        results.append((status, bytes(codes), bytes(scales)))
    assert results[0][0] == -1 and results[0] == results[1]


@pytest.mark.parametrize("live_rows,m,k,format_bits", [(0, 32, 32, 8), (33, 32, 32, 8),
    (1, 15, 32, 8), (1, 32, 31, 4), (1, 32, 32, 7)])
def test_invalid_implicit_zero_extents_do_not_publish_bytes(library, live_rows, m, k, format_bits):
    values = (ctypes.c_float * 32)()
    codes, scales = (ctypes.c_uint8 * 4096)(*([0xA5] * 4096)), (ctypes.c_uint8 * 128)(*([0xA5] * 128))
    assert library.mx_pack_activation_zero_rows_f32(values, live_rows, m, k, format_bits, codes, scales) == -1
    assert bytes(codes) == bytes([0xA5]) * 4096 and bytes(scales) == bytes([0xA5]) * 128


@pytest.mark.parametrize("site,shape", [("g", (32, 352, 128)), ("dn", (32, 128, 352))])
def test_live_small_llama_fp4_operands_match_archived_payload(library, site, shape) -> None:
    np = pytest.importorskip("numpy")
    root = ROOT / "docs/evidence/model2mlir_small_llama_mlp_fp4_padding_e9ded36_266c593"
    activation = np.load(root / "inputs" / f"{site}.activation.npy", allow_pickle=False)
    weight = np.load(root / "inputs" / f"{site}.weight.npy", allow_pickle=False)
    m, n, k = shape
    a = np.zeros((m, k), dtype=np.float32)
    b = np.zeros((k, n), dtype=np.float32)
    a[:activation.shape[0], :activation.shape[1]] = activation
    b[:weight.shape[0], :weight.shape[1]] = weight
    for kind, values in (("activation", a), ("weight", b)):
        rows, cols = values.shape
        codes = (ctypes.c_uint8 * (rows * cols // 2))()
        scales = (ctypes.c_uint8 * (rows * cols // 32))()
        function = getattr(library, f"mx_pack_{kind}_f32")
        assert function(values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        rows, cols, 4, codes, scales) == 0
        assert bytes(codes) == (root / site / "bundle" / f"{kind}.bin").read_bytes()
        assert bytes(scales) == (root / site / "bundle" / f"{kind}_scales.bin").read_bytes()
