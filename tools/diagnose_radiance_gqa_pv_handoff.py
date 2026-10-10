"""Compare the experimental GQA P golden policy with the source Muon requantizer.

This is an analytical diagnostic, not a Muon execution result. It feeds the
same BF16 P values to both encoders and uses rounded torch.exp as a proxy for
Muon mu_fexp when selecting the source-style per-group E8M0 scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from mx_gemmini_support.source_attention_qk import (
    HARDWARE_SHIFT, decode_e4m3, read_gqa_qk_tile)
from mx_gemmini_support.source_attention_pv import (
    MUON_REQUANT_SOURCE_SHA256, source_e4m3_scaled)
from mx_gemmini_support.source_fp6 import _array
from tools.qualify_radiance_gqa_qk import HARDWARE_PATCH
from tools.qualify_radiance_ws_roster import _revision


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()
    prepared_dir, source_root, out = (args.prepared_dir.resolve(),
                                      args.source_root.resolve(), args.out_json.resolve())
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    prepared_index = json.loads((prepared_dir / "index.json").read_text())
    if (prepared_index.get("status") !=
            "generated_source_qk_candidate_matched_on_pinned_spike" or
            prepared_index.get("source_revision") != _revision(source_root) or
            prepared_index.get("generator_patch_sha256") != _sha(HARDWARE_PATCH)):
        raise ValueError("PV diagnostic requires a qualified generated GQA fixture")
    muon_source = source_root / "kernels/flash_attention_mx_gqa/flash_mx_impl.hpp"
    if _sha(muon_source) != MUON_REQUANT_SOURCE_SHA256:
        raise ValueError("source Muon P requantizer changed")
    patched_root = prepared_dir / "patched_source"
    read_gqa_qk_tile(patched_root, head=0, block=0, hardware_generated=True)
    sys.path[:0] = [str(patched_root / "kernels/flash_attention_mx_gqa"),
                    str(source_root / "lib/mxgemmini")]
    import numpy as np
    import torch
    import flash_attention_model as fa_model
    import fp8_matmul_model as low_level_model
    if (Path(fa_model.__file__).resolve() !=
            (patched_root / "kernels/flash_attention_mx_gqa/flash_attention_model.py").resolve() or
            Path(low_level_model.__file__).resolve() !=
            (source_root / "lib/mxgemmini/fp8_matmul_model.py").resolve()):
        raise ValueError("PV numerical model resolved to another checkout")
    fa_model.MX_HARDWARE_SHIFT = HARDWARE_SHIFT
    q, k, v = fa_model.make_inputs_gqa(64, 256, 64, 8, 2, 0)
    qa, qs = fa_model.mx_quantize_cols(q[0])
    kb, ks = fa_model.mx_quantize_cols(k[0, :64])
    score = fa_model.mx_gemm(qa, qs, kb.t().contiguous(), ks.t().contiguous())
    score = fa_model.q_bf16(score * 0.125)
    row_max = score.amax(dim=1, keepdim=True)
    probability = fa_model.q_bf16(torch.exp(score - row_max))
    model_p, model_scales = fa_model.mx_quantize_cols(probability)
    model_codes, bits = low_level_model.tensor_to_custom_fp_codes(
        model_p, low_level_model.INPUT_SPEC)
    model_scale_codes, scale_bits = low_level_model.tensor_to_custom_fp_codes(
        model_scales, low_level_model.SCALE_SPEC)
    if bits != 8 or scale_bits != 9:
        raise ValueError("experimental P model changed FP8 or E8M0 encoding")
    model_codes = np.asarray(model_codes, dtype=np.uint8)
    model_scale_codes = np.asarray(model_scale_codes, dtype=np.uint8)

    # Source Muon computes one scale per 32-column group from exp(block-S-max -
    # running-row-max). torch.exp is only a proxy for the Muon SFU here; the
    # following BF16-bit truncating encoder matches the C formula exactly.
    block_max = score.reshape(64, 2, 32).amax(dim=2)
    block_probability_max = fa_model.q_bf16(torch.exp(
        fa_model.q_bf16(block_max - row_max)))
    block_bits = (block_probability_max.to(torch.bfloat16).view(torch.int16)
                  .to(torch.int32).numpy() & 0xffff)
    scale_exponents = np.minimum(((block_bits >> 7) & 0xff) - 127, 0)
    source_scale_codes = (scale_exponents + 127).astype(np.uint8)
    p_bits = (probability.to(torch.bfloat16).view(torch.int16)
              .to(torch.int32).numpy() & 0xffff)
    source_codes = np.asarray([
        [source_e4m3_scaled(int(p_bits[row, col]),
                            int(scale_exponents[row, col // 32]))
         for col in range(64)] for row in range(64)], dtype=np.uint8)

    v_values, v_scales = fa_model.mx_quantize_cols(v[0, :64].t().contiguous())
    v_codes, v_bits = low_level_model.tensor_to_custom_fp_codes(
        v_values.t().contiguous(), low_level_model.INPUT_SPEC)
    v_scale_codes, v_scale_bits = low_level_model.tensor_to_custom_fp_codes(
        v_scales.t().contiguous(), low_level_model.SCALE_SPEC)
    if v_bits != 8:
        raise ValueError("generated V operand changed FP8 encoding")
    v_codes = np.asarray(v_codes, dtype=np.uint8)
    v_scale_codes = np.asarray(v_scale_codes, dtype=np.uint8)
    source_header = (patched_root /
                     "kernels/flash_attention_mx_gqa/include/fa_data.h").read_text()
    emitted_v = _array(source_header, name="V_in", ctype="uint8_t",
                       dimensions="[FA_NKV*FA_NBLK_USED*FA_BK][FA_D]",
                       count=2 * 2 * 64 * 64, maximum=255)
    emitted_v_scales = _array(source_header, name="V_scales", ctype="uint8_t",
                              dimensions="[FA_NKV*FA_NBLK_USED*FA_GKB][FA_D]",
                              count=2 * 2 * 2 * 64, maximum=255)
    if (v_scale_bits != 9 or v_codes.tobytes() != bytes(emitted_v[:4096]) or
            v_scale_codes.tobytes() != bytes(emitted_v_scales[:128])):
        raise ValueError("diagnostic V bytes differ from the emitted source header")
    p_abs_max = max(abs(decode_e4m3(int(code))) for code in source_codes.flat)
    v_abs_max = max(abs(decode_e4m3(int(code))) for code in v_codes.flat)
    result = {
        "schema": "mx_gemmini.gqa_pv_handoff_diagnostic.v1",
        "status": "source_muon_p_requant_policy_differs_from_experimental_model",
        "scope": "head0 block0 P operand; shared BF16 values; torch.exp proxy, not Muon SFU execution",
        "frontend_mlir_sha256": prepared_index["frontend_mlir_sha256"],
        "generator_patch_sha256": prepared_index["generator_patch_sha256"],
        "source_revision": prepared_index["source_revision"],
        "source_header_sha256": prepared_index["source_header_sha256"],
        "muon_requant_source_sha256": _sha(muon_source),
        "p_code_mismatches": int(np.count_nonzero(model_codes != source_codes)),
        "p_codes_compared": int(source_codes.size),
        "p_scale_mismatches": int(np.count_nonzero(
            model_scale_codes != source_scale_codes)),
        "p_scales_compared": int(source_scale_codes.size),
        "model_p_code_range": [int(model_codes.min()), int(model_codes.max())],
        "source_style_p_code_range": [int(source_codes.min()), int(source_codes.max())],
        "model_scale_code_range": [int(model_scale_codes.min()),
                                   int(model_scale_codes.max())],
        "source_style_scale_code_range": [int(source_scale_codes.min()),
                                          int(source_scale_codes.max())],
        "source_style_raw_pv_product_bound": p_abs_max * v_abs_max,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"P differs in {result['p_code_mismatches']}/{result['p_codes_compared']} "
          f"codes and {result['p_scale_mismatches']}/{result['p_scales_compared']} scales")


if __name__ == "__main__":
    main()
