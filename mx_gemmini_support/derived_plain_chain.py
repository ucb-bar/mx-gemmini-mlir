"""Derive a 16x96x96 connected fixture from Nicolas's 128-cubed wire data.

The unchanged 128-cubed source goldens first check the pinned mesh model.
The 96-wide outputs are model-derived references, never described as source
goldens or as RTL/FPGA qualification.
"""

from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path

import numpy as np
import torch

from .quant_reference import quantize_bf16_fp8_output


MODEL_SHA256 = "0750e78eadeaef36ed94857e92168056dd6196e72dafcd09c20b6db287452071"


def _codes(data: bytes, shape: tuple[int, int]) -> torch.Tensor:
    codes = np.frombuffer(data, dtype=np.uint8).reshape(shape).astype(np.int16)
    exponent, mantissa = (codes >> 3) & 15, codes & 7
    value = np.where(exponent == 0, mantissa * 2.0**-9,
                     (1.0 + mantissa / 8.0) * np.exp2(exponent - 7))
    if np.any((codes & 0x7f) == 0x7f):
        raise ValueError("derived FP8 fixture contains nonfinite source codes")
    return torch.from_numpy(np.where(codes & 0x80, -value, value).astype(np.float32))


def _scales(data: bytes, shape: tuple[int, int], *, activation: bool = False
            ) -> torch.Tensor:
    scales = np.frombuffer(data, dtype=np.uint8).reshape(shape)
    if activation:
        scales = scales.T
    if np.any(scales == 255):
        raise ValueError("derived FP8 fixture contains a nonfinite E8M0 scale")
    return torch.from_numpy(np.exp2(scales.astype(np.float32) - 127))


def _reference(model, resources: dict[str, bytes], m: int, width: int
               ) -> tuple[bytes, bytes, bytes, bytes]:
    a = _codes(resources["a1_activation"], (m, width))
    b1 = _codes(resources["b1_weight"], (width, width))
    b2 = _codes(resources["b2_weight"], (width, width))
    a_scales = _scales(resources["a1_scales"], (width // 32, m), activation=True)
    b1_scales = _scales(resources["b1_scales"], (width // 32, width))
    b2_scales = _scales(resources["b2_scales"], (width // 32, width))
    kwargs = {"verbose": False,
              "prod_precision_list": [(4, 3)] * 16,
              "acc_precision_list": ([(4, 4)] * 8 + [(4, 5)] * 2 +
                                     [(4, 6)] * 5 + [(8, 7)])}

    def quantized(output: torch.Tensor) -> tuple[bytes, bytes]:
        bf16 = output.to(torch.bfloat16).view(torch.int16).numpy()
        return quantize_bf16_fp8_output(bf16.astype("<u2").tobytes(), m, width)

    c1 = quantized(model.tiled_matmul_hwlike(a, b1, a_scales, b1_scales, **kwargs))
    c2 = quantized(model.tiled_matmul_hwlike(
        _codes(c1[0], (m, width)), b2,
        _scales(c1[1], (m, width // 32)), b2_scales, **kwargs))
    return (*c1, *c2)


def derive_16x96_from_128(full: dict[str, bytes], *, model_path: Path
                         ) -> dict[str, bytes]:
    """Check the 128 source oracle, then slice its wire inputs and recalculate."""
    if (not model_path.is_file() or model_path.name != "fp8_matmul_model.py" or
            hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256):
        raise ValueError("Nicolas's pinned FP8 mesh model is absent")
    spec = importlib.util.spec_from_file_location("nicolas_fp8_chain_model", model_path)
    if spec is None or spec.loader is None:
        raise ValueError("Nicolas FP8 mesh model cannot be loaded")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    m, full_width, width = 16, 128, 96
    expected = {"a1_activation": m * full_width, "a1_scales": m * 4,
                "b1_weight": full_width**2, "b1_scales": full_width * 4,
                "b2_weight": full_width**2, "b2_scales": full_width * 4,
                "c1_codes_ref": m * full_width, "c1_scales_ref": m * 4,
                "c2_codes_ref": m * full_width, "c2_scales_ref": m * 4}
    if any(len(full.get(name, b"")) != size for name, size in expected.items()):
        raise ValueError("128-cubed source fixture has missing or changed wire data")
    source_reference = _reference(model, full, m, full_width)
    if source_reference != tuple(full[name] for name in
                                 ("c1_codes_ref", "c1_scales_ref",
                                  "c2_codes_ref", "c2_scales_ref")):
        raise ValueError("pinned FP8 mesh model differs from unchanged 128 source goldens")

    def narrow_matrix(name: str) -> bytes:
        matrix = np.frombuffer(full[name], dtype=np.uint8).reshape(full_width, full_width)
        return matrix[:width, :width].tobytes()

    def narrow_scale(name: str) -> bytes:
        matrix = np.frombuffer(full[name], dtype=np.uint8).reshape(4, full_width)
        return matrix[:width // 32, :width].tobytes()

    resources = {
        "a1_activation": np.frombuffer(full["a1_activation"], dtype=np.uint8)
        .reshape(m, full_width)[:, :width].tobytes(),
        "a1_scales": full["a1_scales"][:m * (width // 32)],
        "b1_weight": narrow_matrix("b1_weight"),
        "b1_scales": narrow_scale("b1_scales"),
        "b2_weight": narrow_matrix("b2_weight"),
        "b2_scales": narrow_scale("b2_scales"),
    }
    c1, c1s, c2, c2s = _reference(model, resources, m, width)
    resources.update(c1_codes_ref=c1, c1_scales_ref=c1s,
                     c2_codes_ref=c2, c2_scales_ref=c2s)
    return resources
