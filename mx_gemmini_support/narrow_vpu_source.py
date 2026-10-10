"""Checked 32-column specialization of Nicolas's FP8 MX/VPU source chain."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .connected_fp8_model import MODEL_SHA256, load_nicolas_fp8_model, model_c2
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


def derive_wide_vpu_resources(source: Path, header: Path, first_source: Path,
                              profile: dict, *, second_width: int
                              ) -> tuple[dict[str, bytes], dict]:
    """Derive distinct extra B2 columns from Nicolas's checked 64x64 source.

    The appended weight columns flip the sign of the source's first columns.
    Their complete MM2 reference comes from the pinned independent mesh model.
    No wider source kernel or golden is implied.
    """
    if second_width not in (96, 128):
        raise ValueError("derived wide VPU case needs 96 or 128 MM2 columns")
    _, full, facts = capture_nicolas_vpu_requant(
        source, header, profile, include_resident_matmul=True,
        first_source_path=first_source)
    model = load_nicolas_fp8_model(source.parents[1])
    original_bf16, original_codes, original_scales = model_c2(
        full, model, width=64)
    if (original_codes, original_scales) != (
            full["c2_codes_ref"], full["c2_scales_ref"]):
        raise ValueError("pinned mesh model differs from Nicolas's original C2 golden")
    c1_codes, c1_scales = quantize_bf16_fp8_output(
        exact_bf16_x2(full["c1_bf16"]), 64, 64)
    if (c1_codes, c1_scales) != (
            full["c1_codes_ref"], full["c1_scales_ref"]):
        raise ValueError("Nicolas's VPU C1 golden differs from BF16 ×2 reference")
    extra = second_width - 64
    wide = dict(full)
    b2 = full["b2_weight"]
    wide["b2_weight"] = b"".join(
        b2[row * 64:(row + 1) * 64] +
        bytes(code ^ 0x80 for code in b2[row * 64:row * 64 + extra])
        for row in range(64))
    scales = full["b2_scales"]
    wide["b2_scales"] = b"".join(
        scales[group * 64:(group + 1) * 64] +
        scales[group * 64:group * 64 + extra]
        for group in range(2))
    c2_bf16, c2_codes, c2_scales = model_c2(wide, model, width=second_width)
    wide["c2_codes_ref"] = c2_codes
    wide["c2_scales_ref"] = c2_scales
    wide["c2_bf16_ref"] = c2_bf16
    facts = {
        **facts,
        "second_width": second_width,
        "derived_weight_columns": extra,
        "model_sha256": MODEL_SHA256,
        "original_c2_bf16_sha256": hashlib.sha256(original_bf16).hexdigest(),
        "source_scope": ("checked Nicolas 64³ MM1/VPU plus sign-flipped "
                         "source B2 columns and model-derived wide MM2 outputs"),
        "resource_sha256": {name: hashlib.sha256(data).hexdigest()
                            for name, data in sorted(wide.items())},
    }
    return wide, facts
