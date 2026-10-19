"""Integer group preparation preserves complete buffers and refusal progress."""

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "mx_gemmini_support/runtime_quant.c"


@pytest.fixture(scope="module")
def quantizers(tmp_path_factory):
    from mx_gemmini_support.runtime_group_preparation import (
        emit_runtime_group_preparation,
    )

    directory = tmp_path_factory.mktemp("quant-group-equivalence")
    (directory / "runtime_group_prepare.h").write_text(emit_runtime_group_preparation())
    libraries = []
    for name, flags in (("original", []), ("words", ["-DMX_QUANT_GROUP_WORDS"]),
                        ("original_inline", ["-DMX_QUANT_INLINE_GROUP"]),
                        ("inlined", ["-DMX_QUANT_GROUP_WORDS", "-DMX_QUANT_INLINE_GROUP"])):
        path = directory / (name + ".so")
        subprocess.run(["cc", "-O2", "-fno-fast-math", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
                        *flags, "-I", str(directory), str(SOURCE), "-o", str(path)], check=True)
        library = ctypes.CDLL(str(path))
        for operation in ("mx_pack_activation_f32", "mx_pack_weight_f32"):
            fn = getattr(library, operation)
            fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                           ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
            fn.restype = ctypes.c_int
        libraries.append(library)
    return libraries


def execute(library, values, precision, operation):
    rows, cols = values.shape
    codes = (ctypes.c_uint8 * (rows * cols * precision // 8))(*([0xa5] * (rows * cols * precision // 8)))
    scales = (ctypes.c_uint8 * (rows * cols // 32))(*([0xa5] * (rows * cols // 32)))
    result = getattr(library, operation)(values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         rows, cols, precision, codes, scales)
    return result, bytes(codes), bytes(scales)


@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("operation", ["mx_pack_activation_f32", "mx_pack_weight_f32"])
@pytest.mark.parametrize("low_bits", [0, 0x7fff, 0x8000, 0x8001, 0xffff])
def test_all_finite_bf16_words_and_narrowing_ties_match(quantizers, precision, operation, low_bits):
    words = np.arange(65536, dtype=np.uint32)
    words = words[((words >> 7) & 255) != 255]
    bits = (words << 16) | low_bits
    rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
    bits = bits[((rounded >> 23) & 255) != 255]
    bits = np.pad(bits, (0, 65536 - bits.size)).astype("<u4")
    values = bits.view("<f4").reshape(64, 1024)
    results = [execute(library, values, precision, operation) for library in quantizers]
    assert results[0][0] == 0 and all(result == results[0] for result in results)


@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("operation", ["mx_pack_activation_f32", "mx_pack_weight_f32"])
@pytest.mark.parametrize("bits", [0x7f800000, 0xff800000, 0x7fc01234, 0x7fa01234, 0x7f7fffff, 0x7fffffff])
def test_rounded_exception_membership_and_all_partial_writes_match(quantizers, precision, operation, bits):
    values = np.linspace(-2.0, 2.0, 8192, dtype=np.float32).reshape(64, 128)
    values.view(np.uint32).reshape(-1)[6007] = bits
    results = [execute(library, values, precision, operation) for library in quantizers]
    assert all(result == results[0] for result in results)
