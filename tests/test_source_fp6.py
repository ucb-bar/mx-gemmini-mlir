"""Source FP6 packed payload and per-row LUT validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mx_gemmini_support.source_fp6 import (_array,
                                           _require_equivalent_lut_aliases,
                                           read_source_fp6_payload)
from mx_gemmini_support.source_gemm import read_source_gemm


SOURCE = Path(os.environ.get("RADIANCE_FP6_SOURCE_ROOT", "/nonexistent"))
DRIVER = SOURCE / "kernels/gemm_mxgemmini/mxgemm.fp6.m128n128k2048.tm128tn128tk128.fullout.cpp"


def test_header_reader_rejects_nonliteral_and_wrong_sized_arrays():
    with pytest.raises(ValueError, match="nonliteral"):
        _array("static const uint8_t A[2][2] = {{1, 2}, {x, 4}\n};",
               name="A", ctype="uint8_t", dimensions="[2][2]", count=4, maximum=255)
    with pytest.raises(ValueError, match="wrong-sized"):
        _array("static const uint8_t A[2][2] = {{1, 2}, {3}\n};",
               name="A", ctype="uint8_t", dimensions="[2][2]", count=4, maximum=255)


def test_header_reader_accepts_only_literal_aligned_arrays():
    declaration = "static const uint8_t A[2][2] __attribute__((aligned(64))) = {{1, 2}, {3, 4}\n};"
    assert _array(declaration, name="A", ctype="uint8_t",
                  dimensions="[2][2]", count=4, maximum=255) == (1, 2, 3, 4)
    with pytest.raises(ValueError, match="declaration"):
        _array(declaration.replace("aligned(64)", "section(64)"), name="A",
               ctype="uint8_t", dimensions="[2][2]", count=4, maximum=255)
    with pytest.raises(ValueError, match="nonliteral"):
        _array(declaration.replace("{3, 4}", "{3, x}"), name="A",
               ctype="uint8_t", dimensions="[2][2]", count=4, maximum=255)


def test_header_reader_accepts_only_declared_symbolic_or_literal_dimensions():
    declaration = "static const uint8_t A[2][2] = {{1, 2}, {3, 4}\n};"
    dimensions = ("[MATMUL_M / 2][MATMUL_K]", "[2][2]")
    assert _array(declaration, name="A", ctype="uint8_t",
                  dimensions=dimensions, count=4, maximum=255) == (1, 2, 3, 4)
    with pytest.raises(ValueError, match="declaration"):
        _array(declaration.replace("[2][2]", "[1][4]"), name="A", ctype="uint8_t",
               dimensions=dimensions, count=4, maximum=255)


def test_fp6_repeated_lut_codes_preserve_original_indices() -> None:
    line = tuple([0, 0] + list(range(2, 16)))
    _require_equivalent_lut_aliases(bytes([0x10]), bytes([0x00]), (line,),
                                    axis_width=1, operand="activation")
    with pytest.raises(ValueError, match="changes decoded code"):
        _require_equivalent_lut_aliases(bytes([0x20]), bytes([0x00]), (line,),
                                        axis_width=1, operand="weight")


@pytest.mark.skipif(not DRIVER.is_file() or
                    not DRIVER.with_name("mxgemm.data.fp6.m128n128k2048.h").is_file(),
                    reason="requires the checked-in FP6 Radiance source header")
def test_real_source_fp6_indices_and_all_lut_banks_roundtrip():
    payload = read_source_fp6_payload(read_source_gemm(DRIVER))
    assert len(payload.activation_bytes) == len(payload.weight_bytes) == 131072
    assert len(payload.activation_lut_bytes) == 768
    assert len(payload.weight_lut_bytes) == 768
    assert len(payload.output_lut_bytes) == 768
    assert len(payload.activation_scale_bytes) == len(payload.weight_scale_bytes) == 8192
    assert len(payload.golden_bf16_bytes) == 32768
    assert len(set(payload.activation_lut_line0)) == 16
    assert len(set(payload.weight_lut_line0)) == 16
