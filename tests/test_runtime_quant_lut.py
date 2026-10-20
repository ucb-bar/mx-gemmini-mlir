"""The fast live MX quantizer preserves the reference's exact byte layout."""

from __future__ import annotations

import ctypes
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "mx_gemmini_support/runtime_quant.c"


def test_fast_codes_match_reference_for_all_bf16_mantissas(tmp_path) -> None:
    wrapper = tmp_path / "quant_test.c"
    wrapper.write_text(
        '#include "runtime_quant.c"\n'
        'uint8_t test_fast(float value, int exponent, uint32_t format) {\n'
        '  return fast_element_code(value, exponent, format);\n}\n'
        'uint8_t test_reference(float value, int exponent, uint32_t format) {\n'
        '  return element_code(value / power_of_two(exponent), format);\n}\n')
    library = tmp_path / "quant_test.so"
    subprocess.run([
        "cc", "-O2", "-fno-fast-math", "-fPIC", "-shared", "-I", str(SOURCE.parent),
        str(wrapper), "-o", str(library)], check=True)
    dll = ctypes.CDLL(str(library))
    for name in ("test_fast", "test_reference"):
        fn = getattr(dll, name)
        fn.argtypes = [ctypes.c_float, ctypes.c_int, ctypes.c_uint32]
        fn.restype = ctypes.c_uint8
    for scale in (-23, 0, 100):
        for delta in range(17):
            for mantissa in range(128):
                bits = ((scale - delta + 127) << 23) | (mantissa << 16)
                for sign in (0, 0x80000000):
                    value = struct.unpack("<f", struct.pack("<I", bits | sign))[0]
                    for precision in (4, 8):
                        assert dll.test_fast(value, scale, precision) == dll.test_reference(
                            value, scale, precision)
    for bits in (0, 0x80000000, 0x00010000, 0x80010000):
        value = struct.unpack("<f", struct.pack("<I", bits))[0]
        for precision in (4, 8):
            assert dll.test_fast(value, -23, precision) == dll.test_reference(
                value, -23, precision)


@pytest.mark.parametrize("word_groups", [False, True])
def test_fast_group_packing_matches_reference(tmp_path, word_groups) -> None:
    from mx_gemmini_support.runtime_group_preparation import (
        emit_runtime_group_preparation,
    )

    (tmp_path / "runtime_group_prepare.h").write_text(emit_runtime_group_preparation())
    libraries = []
    for name, flags in (("fast", []), ("reference", ["-DMX_QUANT_REFERENCE"])):
        if name == "fast" and word_groups:
            flags = ["-DMX_QUANT_GROUP_WORDS", "-DMX_QUANT_INLINE_GROUP", "-I", str(tmp_path)]
        library = tmp_path / f"{name}.so"
        subprocess.run([
            "cc", "-O2", "-fno-fast-math", "-fPIC", "-shared", *flags,
            str(SOURCE), "-o", str(library)], check=True)
        libraries.append(ctypes.CDLL(str(library)))
    rng = np.random.default_rng(20261013)
    powers = rng.integers(-30, 31, size=(64, 128))
    values = np.ldexp(rng.uniform(-1.99, 1.99, size=(64, 128)), powers).astype(np.float32)
    values.flat[:4] = [0.0, -0.0, np.float32(2**-149), -np.float32(2**-149)]
    for precision in (4, 8):
        for name, shape in (("mx_pack_activation_f32", (64, 128)),
                            ("mx_pack_weight_f32", (64, 128))):
            outputs = []
            for library in libraries:
                fn = getattr(library, name)
                fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
                               ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
                fn.restype = ctypes.c_int
                codes = (ctypes.c_uint8 * (64 * 128 // (2 if precision == 4 else 1)))()
                scales = (ctypes.c_uint8 * (64 * 128 // 32))()
                assert fn(values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                          *shape, precision, codes, scales) == 0
                outputs.append((bytes(codes), bytes(scales)))
            assert outputs[0] == outputs[1]
