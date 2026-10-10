"""Audit Nicolas's two-tile VPU chain and derive both exact source references."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .quant_reference import exact_bf16_x2, quantize_bf16_fp8_output
from .source_fp6 import _array, _bytes
from .source_vector_chain import _define
from .target_profile import profile_sha256


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit_chain_pipelined(source_path: Path, header_path: Path,
                          profile: dict) -> tuple[dict[str, bytes], dict]:
    """Read source wire data and independently recompute x2/x4 goldens."""
    if (source_path.name != "chain_pipelined.c" or
            header_path.name != "matmul_fp8_64x64_chain.h"):
        raise ValueError("selected Nicolas two-tile chain source pair differs")
    source_bytes, header_bytes = source_path.read_bytes(), header_path.read_bytes()
    source, header = source_bytes.decode("ascii"), header_bytes.decode("ascii")
    if ("\"include/matmul_fp8_64x64_chain.h\"" not in source or
            tuple(_define(header, name) for name in
                  ("MATMUL_M", "MATMUL_N", "MATMUL_K", "MATMUL_GN")) !=
            (64, 64, 64, 2)):
        raise ValueError("Nicolas two-tile chain dimensions or header differ")
    markers = (
        "static const uint32_t SP_BF16[T] = {0x1000, 0x1200};",
        "static const uint32_t SP_C1[T]   = {128, 1024};",
        "static const uint32_t SP_C2[T]   = {512, 1536};",
        "static const uint16_t FACT[T]    = {0x4000, 0x4080};",
        "gemmini_vpu_scalar(VPU_MULS, SP_BF16[t], SP_BF16[t], FACT[t], M * N / 8);",
        "gemmini_spad_requant(SP_C1[t], SP_BF16[t], M, N, 1, (uint64_t)c1_scales[t], 1);",
        "gemmini_mxquant_config_mvout_resident((uint64_t)c2_scales[t], tiles_I, tiles_J, tiles_K, 0, 0, 1);",
        "gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, SP_C1[t], BANK_NUM * BANK_ROWS, 0, SP_C2[t],",
        "mvin_tile(0); mvin_tile(1);",
        "vpu_tile(0); requant_tile(0);",
        "vpu_tile(1);",
        "mm_tile(0);",
        "requant_tile(1);",
        "mm_tile(1);",
    )
    if any(marker not in source for marker in markers):
        raise ValueError("Nicolas two-tile chain placement or issue order changed")
    if (profile["transport"] != "rocket_rocc" or
            profile["geometry"]["mesh_columns"] != 16 or
            not profile["resources"].get("vpu") or
            not profile["resources"].get("spad_requant") or
            profile["resources"]["scratchpad_bytes"] < 0x1400 * 16):
        raise ValueError("selected profile cannot run Nicolas's two-tile chain")

    c1_bf16 = _bytes(_array(header, name="C1_out_bf16", ctype="uint16_t",
                           dimensions="[MATMUL_M][MATMUL_N]", count=4096,
                           maximum=65535), 2)
    b2 = bytes(_array(header, name="B2_in", ctype="uint8_t",
                      dimensions="[MATMUL_K][MATMUL_N]", count=4096,
                      maximum=255))
    b2_scales = bytes(_array(header, name="B2_scales_col", ctype="uint8_t",
                             dimensions="[MATMUL_GK][MATMUL_N]", count=128,
                             maximum=255))
    resources = {"c1_bf16": c1_bf16, "b2_weight": b2, "b2_scales": b2_scales}
    for prefix in ("C1", "C2"):
        raw = c1_bf16 if prefix == "C1" else _bytes(_array(
            header, name="C2_out_bf16", ctype="uint16_t",
            dimensions="[MATMUL_M][MATMUL_N]", count=4096, maximum=65535), 2)
        source_codes = bytes(_array(header, name=f"{prefix}_out", ctype="uint8_t",
                                    dimensions="[MATMUL_M][MATMUL_N]",
                                    count=4096, maximum=255))
        source_scales = bytes(_array(header, name=f"{prefix}_scales_out", ctype="uint8_t",
                                     dimensions="[MATMUL_M][MATMUL_GN]",
                                     count=128, maximum=255))
        for tile, shift in enumerate((1, 2)):
            scaled = exact_bf16_x2(raw) if shift == 1 else exact_bf16_x2(exact_bf16_x2(raw))
            codes, scales = quantize_bf16_fp8_output(scaled, 64, 64)
            if (codes != source_codes or
                    any(got != original + shift for got, original
                        in zip(scales, source_scales))):
                raise ValueError(f"Nicolas {prefix} tile {tile} source golden differs")
            resources[f"{prefix.lower()}_codes_ref_{tile}"] = codes
            resources[f"{prefix.lower()}_scales_ref_{tile}"] = scales
    facts = {
        "source_sha256": _sha(source_bytes), "header_sha256": _sha(header_bytes),
        "profile_sha256": profile_sha256(profile),
        "source_scope": "preloaded BF16 C1; two VPU scalar branches and resident MM2 tiles; source tests three issue schedules",
        "tile_factors_bf16": [0x4000, 0x4080],
        "bf16_rows": [0x1000, 0x1200], "c1_rows": [128, 1024],
        "c2_rows": [512, 1536],
        "resource_sha256": {key: _sha(data) for key, data in sorted(resources.items())},
    }
    return resources, facts
