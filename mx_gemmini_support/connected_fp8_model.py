"""Pinned independent FP8 MM2 oracle for source-derived connected MX graphs."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

from .quant_reference import quantize_bf16_fp8_output
from .source_attention_qk import (
    ACCUMULATOR_PRECISION, PRODUCT_PRECISION, decode_e4m3)


MODEL_SHA256 = "0750e78eadeaef36ed94857e92168056dd6196e72dafcd09c20b6db287452071"


def load_nicolas_fp8_model(software: Path):
    """Use exactly the model matching Nicolas's checked-in chain goldens."""
    path = software / "fp8_matmul_model.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("Nicolas's pinned FP8 model changed")
    spec = importlib.util.spec_from_file_location("nicolas_fp8_model", path)
    if spec is None or spec.loader is None:
        raise ValueError("Nicolas's FP8 model cannot be imported")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model


def model_c2(resources: dict[str, bytes], model, *, width: int,
             first_width: int = 64
             ) -> tuple[bytes, bytes, bytes]:
    """Return BF16, FP8 codes, and E8M0 scales for C1[M,M] × B2[M,N]."""
    import torch

    if (type(width) is not int or width < 32 or width % 32 or
            first_width not in (64, 128) or
            len(resources["c1_codes_ref"]) != first_width ** 2 or
            len(resources["c1_scales_ref"]) != first_width ** 2 // 32 or
            len(resources["b2_weight"]) != first_width * width or
            len(resources["b2_scales"]) != first_width * width // 32):
        raise ValueError("connected MM2 model needs complete FP8 operand blocks")

    def values(codes: bytes, rows: int, cols: int):
        return torch.tensor([decode_e4m3(code) for code in codes],
                            dtype=torch.float32).reshape(rows, cols)

    def scales(codes: bytes, rows: int, cols: int):
        return torch.tensor([math.ldexp(1.0, code - 127) for code in codes],
                            dtype=torch.float32).reshape(rows, cols)

    result = model.tiled_matmul_hwlike(
        values(resources["c1_codes_ref"], first_width, first_width),
        values(resources["b2_weight"], first_width, width),
        scales(resources["c1_scales_ref"], first_width, first_width // 32),
        scales(resources["b2_scales"], first_width // 32, width), verbose=False,
        prod_precision_list=PRODUCT_PRECISION,
        acc_precision_list=ACCUMULATOR_PRECISION)
    if not torch.isfinite(result).all():
        raise ValueError("connected MM2 reference contains nonfinite values")
    codes, bits = model.tensor_to_custom_fp_codes(result, "bf16")
    if bits != 16:
        raise ValueError("Nicolas MM2 model changed BF16 output width")
    bf16 = b"".join(int(code).to_bytes(2, "little")
                    for row in codes for code in row)
    quantized, scales_out = quantize_bf16_fp8_output(bf16, first_width, width)
    return bf16, quantized, scales_out
