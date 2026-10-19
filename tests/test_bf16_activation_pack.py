"""Compact inputs retain complete MX bytes, scale selection and refusal order."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mx_gemmini_support.runtime_group_preparation import emit_runtime_group_preparation

SUPPORT = Path(__file__).resolve().parents[1] / "mx_gemmini_support"


@pytest.fixture(scope="module")
def packers(tmp_path_factory):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler required")
    root = tmp_path_factory.mktemp("bf16_pack_images")
    (root / "runtime_group_prepare.h").write_text(emit_runtime_group_preparation(bf16_input=True))
    result = {}
    for name, flags in [
        ("words", ["-DMX_QUANT_GROUP_WORDS", "-DMX_QUANT_INLINE_GROUP", "-DMX_QUANT_BF16_INPUT"]),
        ("reference", ["-DMX_QUANT_REFERENCE"]),
    ]:
        image = root / (name + ".so")
        subprocess.run([cc, "-shared", "-fPIC", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
                        *flags, "-I", str(root), str(SUPPORT / "runtime_quant.c"), "-o", str(image)],
                       check=True)
        library = ctypes.CDLL(str(image))
        library.mx_pack_activation_f32.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p]
        library.mx_pack_activation_f32.restype = ctypes.c_int
        if name == "words":
            library.mx_pack_activation_bf16.argtypes = library.mx_pack_activation_f32.argtypes
            library.mx_pack_activation_bf16.restype = ctypes.c_int
        result[name] = library
    assert not hasattr(result["reference"], "mx_pack_activation_bf16")
    return result


def pack(function, values, m, k, format_bits):
    codes = np.full(m * k * format_bits // 8 + 2, 0xA5, dtype=np.uint8)
    scales = np.full(m * k // 32 + 2, 0xA5, dtype=np.uint8)
    status = function(values.ctypes.data, m, k, format_bits,
                      codes.ctypes.data + 1, scales.ctypes.data + 1)
    assert (codes[0], codes[-1], scales[0], scales[-1]) == (0xA5,) * 4
    return status, codes.tobytes(), scales.tobytes()


def compare(packers, words, format_bits):
    assert words.dtype == np.uint16 and words.ndim == 2 and words.flags.c_contiguous
    m, k = words.shape
    widened = (words.astype(np.uint32) << 16).view(np.float32)
    compact = pack(packers["words"].mx_pack_activation_bf16, words, m, k, format_bits)
    selected = pack(packers["words"].mx_pack_activation_f32, widened, m, k, format_bits)
    reference = pack(packers["reference"].mx_pack_activation_f32, widened, m, k, format_bits)
    assert compact == selected == reference
    return compact


@pytest.mark.parametrize("format_bits", [4, 8])
def test_every_finite_bf16_word_as_its_own_group_preserves_codes_and_scales(packers, format_bits):
    words = np.arange(65536, dtype=np.uint16)
    finite = words[(words & 0x7F80) != 0x7F80]
    # Each represented value is a group maximum, exercising every exponent,
    # mantissa, sign, signed zero and subnormal without a larger value hiding it.
    groups = np.repeat(finite, 32).reshape(32, -1)
    assert compare(packers, groups, format_bits)[0] == 0


@pytest.mark.parametrize("format_bits", [4, 8])
def test_mixed_groups_and_mutated_input_refresh_match_independent_reference(packers, format_bits):
    rng = np.random.default_rng(7641)
    words = np.arange(65536, dtype=np.uint16)
    words[(words & 0x7F80) == 0x7F80] = 0
    rng.shuffle(words)
    for value in [words.reshape(256, 256), words[::-1].copy().reshape(256, 256)]:
        assert compare(packers, value, format_bits)[0] == 0


@pytest.mark.parametrize("format_bits", [4, 8])
def test_every_nonfinite_word_preserves_refusal_and_partial_publication(packers, format_bits):
    all_words = np.arange(65536, dtype=np.uint16)
    nonfinite = all_words[(all_words & 0x7F80) == 0x7F80]
    for bad in nonfinite:
        words = np.full((32, 96), 0x3F80, dtype=np.uint16)
        words[19, 71] = bad  # later group, after both FP4 nibble positions were written
        assert compare(packers, words, format_bits)[0] == -1


@pytest.mark.parametrize("m,k,format_bits", [(0, 32, 8), (15, 32, 8), (16, 16, 8),
    (16, 32, 4), (32, 31, 4), (32, 32, 7)])
def test_invalid_padded_extents_publish_nothing(packers, m, k, format_bits):
    words = np.zeros((32, 96), dtype=np.uint16)
    codes = np.full(4096, 0xA5, dtype=np.uint8)
    scales = np.full(128, 0xA5, dtype=np.uint8)
    function = packers["words"].mx_pack_activation_bf16
    assert function(words.ctypes.data, m, k, format_bits, codes.ctypes.data, scales.ctypes.data) == -1
    assert (codes == 0xA5).all() and (scales == 0xA5).all()


@pytest.mark.parametrize("missing", [0, 4, 5])
def test_missing_buffers_refuse_before_publication(packers, missing):
    words = np.zeros((32, 32), dtype=np.uint16)
    codes = np.full(1024, 0xA5, dtype=np.uint8)
    scales = np.full(32, 0xA5, dtype=np.uint8)
    arguments = [words.ctypes.data, 32, 32, 8, codes.ctypes.data, scales.ctypes.data]
    arguments[missing] = None
    assert packers["words"].mx_pack_activation_bf16(*arguments) == -1
    assert (codes == 0xA5).all() and (scales == 0xA5).all()
