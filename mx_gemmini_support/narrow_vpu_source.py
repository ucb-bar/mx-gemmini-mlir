"""Checked 32-column specialization of Nicolas's FP8 MX/VPU source chain."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .quant_reference import exact_bf16_x2, quantize_bf16_fp8_output
from .source_fp6 import _array
from .source_vector_chain import capture_nicolas_vpu_requant


def derive_narrow_vpu_resources(source: Path, header: Path, first_source: Path,
                                profile: dict) -> tuple[dict[str, bytes], dict]:
    """Audit the unchanged 64³ chain, then check its left 32-column output."""
    seam, full, facts = capture_nicolas_vpu_requant(
        source, header, profile, include_resident_matmul=True,
        first_source_path=first_source)
    narrow = dict(full)
    for name in ("b2_weight", "c2_codes_ref"):
        data = full[name]
        narrow[name] = b"".join(data[row * 64:row * 64 + 32]
                                for row in range(64))
    b2_scales = full["b2_scales"]
    narrow["b2_scales"] = b2_scales[:32] + b2_scales[64:96]
    narrow["c2_scales_ref"] = full["c2_scales_ref"][::2]
    text = header.read_text()
    bf16 = _array(text, name="C2_out_bf16", ctype="uint16_t",
                  dimensions="[MATMUL_M][MATMUL_N]", count=64 * 64,
                  maximum=65535)
    left_bf16 = b"".join(
        value.to_bytes(2, "little") for row in range(64)
        for value in bf16[row * 64:row * 64 + 32])
    codes, scales = quantize_bf16_fp8_output(exact_bf16_x2(left_bf16), 64, 32)
    if (codes, scales) != (narrow["c2_codes_ref"], narrow["c2_scales_ref"]):
        raise ValueError("narrow VPU/MM2 reference differs from source BF16 blocks")
    facts = {**facts, "second_width": 32,
             "full_seam_mlir_sha256": hashlib.sha256(seam.encode()).hexdigest(),
             "source_scope": "checked 64-cubed MM1/VPU/requant and first 32 MM2 columns",
             "resource_sha256": {name: hashlib.sha256(data).hexdigest()
                                 for name, data in sorted(narrow.items())}}
    return narrow, facts
