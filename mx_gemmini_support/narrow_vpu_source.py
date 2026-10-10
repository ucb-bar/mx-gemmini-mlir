"""Checked 32-column specialization of Nicolas's FP8 MX/VPU source chain."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .connected_fp8_model import MODEL_SHA256, load_nicolas_fp8_model, model_c2
from .quant_reference import exact_bf16_x2, quantize_bf16_fp8_output
from .source_fp6 import _array
from .source_vector_chain import capture_nicolas_vpu_requant
from .target_profile import profile_sha256


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


def derive_square_128_vpu_resources(source: Path, header: Path, profile: dict
                                   ) -> tuple[dict[str, bytes], dict]:
    """Bind Nicolas's 128-cubed source operands to a derived VPU ×2 graph.

    Check the unchanged source's complete MM1/MM2 quantized goldens before
    deriving the VPU C1 and downstream C2 references. The checked source has
    no VPU operation; the new graph is an explicitly derived candidate.
    """
    if (source.name != "matmul_tiled_fp8_128x128_chain.c" or
            header.name != "matmul_fp8_128x128_chain.h" or
            '"include/matmul_fp8_128x128_chain.h"' not in source.read_text()):
        raise ValueError("derived 128-cubed VPU needs Nicolas's pinned source pair")
    text = header.read_text()
    if any(f"#define {name} {value}" not in text for name, value in
           (("MATMUL_M", 128), ("MATMUL_N", 128), ("MATMUL_K", 128),
            ("MATMUL_GK", 4), ("MATMUL_GN", 4))):
        raise ValueError("derived 128-cubed VPU source geometry differs")

    def codes(name: str, dimensions: str, count: int) -> bytes:
        return bytes(_array(text, name=name, ctype="uint8_t",
                            dimensions=dimensions, count=count, maximum=255))

    matrix = "[MATMUL_M][MATMUL_N]"
    k_matrix = "[MATMUL_K][MATMUL_N]"
    resources = {
        "a1_activation": codes("A_in", "[MATMUL_M][MATMUL_K]", 16384),
        "a1_scales": codes("A_scales_row", "[MATMUL_GK][MATMUL_M]", 512),
        "b1_weight": codes("B_in", k_matrix, 16384),
        "b1_scales": codes("B_scales_col", "[MATMUL_GK][MATMUL_N]", 512),
        "b2_weight": codes("B2_in", k_matrix, 16384),
        "b2_scales": codes("B2_scales_col", "[MATMUL_GK][MATMUL_N]", 512),
    }
    original_bf16 = b"".join(int(word).to_bytes(2, "little") for word in
        _array(text, name="C1_out_bf16", ctype="uint16_t",
               dimensions=matrix, count=16384, maximum=65535))
    original_c1 = codes("C1_out", matrix, 16384)
    original_c1_scales = codes("C1_scales_out", "[MATMUL_M][MATMUL_GN]", 512)
    if quantize_bf16_fp8_output(original_bf16, 128, 128) != (
            original_c1, original_c1_scales):
        raise ValueError("Nicolas 128-cubed MM1 source golden differs")
    model = load_nicolas_fp8_model(source.parents[1])
    original = {**resources, "c1_codes_ref": original_c1,
                "c1_scales_ref": original_c1_scales}
    _, original_c2, original_c2_scales = model_c2(
        original, model, width=128, first_width=128)
    if (original_c2 != codes("C2_out", matrix, 16384) or
            original_c2_scales != codes(
                "C2_scales_out", "[MATMUL_M][MATMUL_GN]", 512)):
        raise ValueError("pinned mesh model differs from Nicolas 128-cubed C2")
    resources["c1_bf16"] = original_bf16
    resources["c1_codes_ref"], resources["c1_scales_ref"] = (
        quantize_bf16_fp8_output(exact_bf16_x2(original_bf16), 128, 128))
    c2_bf16, resources["c2_codes_ref"], resources["c2_scales_ref"] = model_c2(
        resources, model, width=128, first_width=128)
    resources["c2_bf16_ref"] = c2_bf16
    facts = {
        "first_width": 128, "second_width": 128,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        "profile_sha256": profile_sha256(profile),
        "policy_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "examples/default-policy.yaml").read_bytes()
        ).hexdigest(),
        "model_sha256": MODEL_SHA256,
        "source_scope": ("unchanged Nicolas 128-cubed MM1 operands and BF16 "
                         "golden; derived BF16 VPU x2 and pinned-model MM2 outputs"),
        "resource_sha256": {name: hashlib.sha256(data).hexdigest()
                            for name, data in sorted(resources.items())},
    }
    return resources, facts
