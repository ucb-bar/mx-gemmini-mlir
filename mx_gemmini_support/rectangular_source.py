"""Checked Nicolas wire inputs and output oracle for rectangular FP8 pairs."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch

from .derived_plain_chain import MODEL_SHA256, _codes, _scales
from .quant_reference import quantize_bf16_fp8_output
from .source_fp6 import _array


FIRST_SHAPE = (64, 96, 64)
def derive_rectangular_resources(first_header: Path, b2_header: Path, *,
                                 model_path: Path,
                                 second_width: int = 64) -> dict[str, bytes]:
    """Use the unchanged MM1 golden, then model C2 from a checked B2 slice."""
    if (first_header.name != "matmul_fp8_64x96x64.h" or
            b2_header.name != "matmul_fp8_128x128_chain.h" or
            not model_path.is_file() or model_path.name != "fp8_matmul_model.py" or
            hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256):
        raise ValueError("rectangular pair needs Nicolas's pinned headers and mesh model")
    if second_width not in (32, 64):
        raise ValueError("rectangular pair MM2 width needs a selected source slice")
    first, second = first_header.read_text(), b2_header.read_text()
    if any(marker not in first for marker in
           ("#define MATMUL_M 64", "#define MATMUL_K 64", "#define MATMUL_N 96",
            "#define MATMUL_GK 2", "#define MATMUL_GN 3")) or any(
                marker not in second for marker in
                ("#define MATMUL_M 128", "#define MATMUL_K 128",
                 "#define MATMUL_N 128", "#define MATMUL_GK 4")):
        raise ValueError("rectangular source header geometry differs")

    def codes(text: str, name: str, dims: str, count: int) -> bytes:
        return bytes(_array(text, name=name, ctype="uint8_t",
                            dimensions=dims, count=count, maximum=255))

    a = codes(first, "A_in", "[MATMUL_M][MATMUL_K]", 64 * 64)
    a_scales = codes(first, "A_scales_row", "[MATMUL_GK][MATMUL_M]", 2 * 64)
    b1 = codes(first, "B_in", "[MATMUL_K][MATMUL_N]", 64 * 96)
    b1_scales = codes(first, "B_scales_col", "[MATMUL_GK][MATMUL_N]", 2 * 96)
    c1_codes = codes(first, "C_out", "[MATMUL_M][MATMUL_N]", 64 * 96)
    c1_scales = codes(first, "C_scales_out", "[MATMUL_M][MATMUL_GN]", 64 * 3)
    c1_bf16 = _array(first, name="C_out_bf16", ctype="uint16_t",
                     dimensions="[MATMUL_M][MATMUL_N]", count=64 * 96,
                     maximum=65535)
    b2_full = np.array(_array(second, name="B2_in", ctype="uint8_t",
                              dimensions="[MATMUL_K][MATMUL_N]", count=128 * 128,
                              maximum=255), dtype=np.uint8).reshape(128, 128)
    b2_scales_full = np.array(_array(
        second, name="B2_scales_col", ctype="uint8_t",
        dimensions="[MATMUL_GK][MATMUL_N]", count=4 * 128,
        maximum=255), dtype=np.uint8).reshape(4, 128)
    b2 = b2_full[:96, :second_width].tobytes()
    b2_scales = b2_scales_full[:3, :second_width].tobytes()

    spec = importlib.util.spec_from_file_location("nicolas_rectangular_mesh", model_path)
    if spec is None or spec.loader is None:
        raise ValueError("Nicolas rectangular mesh model cannot be loaded")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    kwargs = {"verbose": False, "prod_precision_list": [(4, 3)] * 16,
              "acc_precision_list": ([(4, 4)] * 8 + [(4, 5)] * 2 +
                                     [(4, 6)] * 5 + [(8, 7)])}

    def bf16_bytes(value: torch.Tensor) -> bytes:
        return value.to(torch.bfloat16).view(torch.int16).numpy().astype("<u2").tobytes()

    first_bf16 = bf16_bytes(model.tiled_matmul_hwlike(
        _codes(a, (64, 64)), _codes(b1, (64, 96)),
        _scales(a_scales, (2, 64), activation=True),
        _scales(b1_scales, (2, 96)), **kwargs))
    source_bf16 = np.array(c1_bf16, dtype="<u2").tobytes()
    if (first_bf16 != source_bf16 or
            quantize_bf16_fp8_output(first_bf16, 64, 96) !=
            (c1_codes, c1_scales)):
        raise ValueError("rectangular MM1 model differs from unchanged source goldens")
    second_bf16 = bf16_bytes(model.tiled_matmul_hwlike(
        _codes(c1_codes, (64, 96)), _codes(b2, (96, second_width)),
        _scales(c1_scales, (64, 3)),
        _scales(b2_scales, (3, second_width)), **kwargs))
    c2_codes, c2_scales = quantize_bf16_fp8_output(
        second_bf16, 64, second_width)
    return {
        "a1_activation": a, "a1_scales": a_scales,
        "b1_weight": b1, "b1_scales": b1_scales,
        "b2_weight": b2, "b2_scales": b2_scales,
        "c1_codes_ref": c1_codes, "c1_scales_ref": c1_scales,
        "c2_codes_ref": c2_codes, "c2_scales_ref": c2_scales,
    }
