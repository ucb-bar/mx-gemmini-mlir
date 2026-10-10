"""Source-bound PV bytes with an explicitly approximate Muon P producer.

The first GQA key block is fully visible. Its QK BF16 result feeds the source
Muon requantization formula, with rounded Torch exp standing in for mu_fexp.
This can qualify MX consumption on Spike, not Muon execution or final O.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import re

from .source_attention_qk import (HARDWARE_HEADER_SHA256, LOW_LEVEL_MODEL_SHA256,
                                  decode_e4m3, PRODUCT_PRECISION,
                                  ACCUMULATOR_PRECISION)
from .source_fp6 import _array, _bytes
from .source_payload import (ATTENTION_QK_CANDIDATE_ORIGIN, Resource,
                             load_bundle)


MUON_REQUANT_SOURCE_SHA256 = "c7dc2a63bf283a7cb3035983c26f76651569fa43a875d6fde06bcaf8991fa47f"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_e4m3_scaled(bf16: int, scale_exponent: int) -> int:
    """Translate pinned bf16_to_e4m3_scaled for the nonnegative P domain."""
    exp = (bf16 >> 7) & 0xff
    exponent = exp - 127 - scale_exponent
    mantissa = (bf16 >> 4) & 7
    if exp == 0 or exponent < -6:
        return 0
    if exponent > 8:
        exponent, mantissa = 8, 6
    if exponent == 8 and mantissa > 6:
        mantissa = 6
    return (((bf16 >> 8) & 0x80) | ((exponent + 7) << 3) | mantissa) & 0xff


def derive_first_pv_proxy(qk_bundle: Path, header_gz: Path, muon_source: Path, *,
                          torch, low_level_model) -> tuple[dict[str, Resource], dict]:
    """Derive one 64³ PV candidate from checked QK output and emitted V bytes."""
    import numpy as np

    qk_manifest, qk_resources = load_bundle(qk_bundle)
    if (qk_manifest.get("origin") != ATTENTION_QK_CANDIDATE_ORIGIN or
            qk_manifest.get("source_derivation", {}).get("stage") !=
            "gqa_qk_head0_block0" or
            qk_manifest.get("shape_mnk") != [64, 64, 64] or
            qk_manifest.get("source_header_sha256") != HARDWARE_HEADER_SHA256 or
            _sha(muon_source.read_bytes()) != MUON_REQUANT_SOURCE_SHA256 or
            _sha(Path(low_level_model.__file__).read_bytes()) != LOW_LEVEL_MODEL_SHA256):
        raise ValueError("PV proxy needs pinned QK, Muon requant, and MX numerical source")
    header_bytes = gzip.decompress(header_gz.read_bytes())
    if _sha(header_bytes) != HARDWARE_HEADER_SHA256:
        raise ValueError("PV proxy V arrays differ from the QK source header")
    header = header_bytes.decode("ascii")
    for macro, value in (("FA_SQ", 64), ("FA_BK", 64), ("FA_D", 64),
                         ("FA_QPOS0", 64), ("FA_SOFTMAX_SCALE_BF16", 0x3e00)):
        found = re.search(rf"^#define {macro}\s+(0x[0-9a-fA-F]+|\d+)\b", header,
                          re.MULTILINE)
        if found is None or int(found.group(1), 0) != value:
            raise ValueError(f"PV proxy source header changed {macro}")
    score_bits = np.frombuffer(qk_resources["golden_bf16"], dtype="<u2").copy()
    score = torch.from_numpy(score_bits).view(torch.bfloat16).reshape(64, 64).float()
    scaled = (score * 0.125).to(torch.bfloat16).float()
    row_max = scaled.amax(dim=1, keepdim=True)
    # The source computes one BF16 exp per probability and per 32-column max.
    # Torch supplies rounded exp here; its SFU bit pattern is not Muon evidence.
    probability = ((scaled - row_max).to(torch.bfloat16).float().exp()
                   .to(torch.bfloat16))
    block_max = scaled.reshape(64, 2, 32).amax(dim=2)
    block_probability_max = ((block_max - row_max).to(torch.bfloat16).float()
                             .exp().to(torch.bfloat16))
    block_bits = block_probability_max.view(torch.int16).to(torch.int32).numpy() & 0xffff
    scale_exponents = np.minimum(((block_bits >> 7) & 0xff) - 127, 0)
    activation_scales = (scale_exponents + 127).astype(np.uint8).T.copy()
    p_bits = probability.view(torch.int16).to(torch.int32).numpy() & 0xffff
    activation = np.asarray([
        [source_e4m3_scaled(int(p_bits[row, col]),
                            int(scale_exponents[row, col // 32]))
         for col in range(64)] for row in range(64)], dtype=np.uint8)
    v_codes = np.asarray(_array(
        header, name="V_in", ctype="uint8_t",
        dimensions="[FA_NKV*FA_NBLK_USED*FA_BK][FA_D]",
        count=2 * 2 * 64 * 64, maximum=255)[:4096], dtype=np.uint8)
    v_scales = np.asarray(_array(
        header, name="V_scales", ctype="uint8_t",
        dimensions="[FA_NKV*FA_NBLK_USED*FA_GKB][FA_D]",
        count=2 * 2 * 2 * 64, maximum=255)[:128], dtype=np.uint8)
    a = torch.tensor([decode_e4m3(int(code)) for code in activation.flat],
                     dtype=torch.float32).reshape(64, 64)
    b = torch.tensor([decode_e4m3(int(code)) for code in v_codes.flat],
                     dtype=torch.float32).reshape(64, 64)
    a_scales = torch.tensor([float(np.ldexp(1.0, int(code) - 127))
                             for code in activation_scales.flat],
                            dtype=torch.float32).reshape(2, 64).T
    b_scales = torch.tensor([float(np.ldexp(1.0, int(code) - 127))
                             for code in v_scales.flat],
                            dtype=torch.float32).reshape(2, 64)
    golden = low_level_model.tiled_matmul_hwlike(
        a, b, a_scales, b_scales, verbose=False,
        prod_precision_list=PRODUCT_PRECISION,
        acc_precision_list=ACCUMULATOR_PRECISION)
    if not torch.isfinite(golden).all():
        raise ValueError("PV proxy overflows Nicolas's MX numerical path")
    codes, width = low_level_model.tensor_to_custom_fp_codes(golden, "bf16")
    if width != 16:
        raise ValueError("PV proxy numerical model changed BF16 encoding")
    resources = {
        "activation": Resource(activation.tobytes(), (64, 64), 8, "row_major_codes"),
        "activation_scales": Resource(activation_scales.tobytes(), (2, 64), 8,
                                      "k_group_row_e8m0"),
        "weight": Resource(v_codes.tobytes(), (64, 64), 8, "row_major_codes"),
        "weight_scales": Resource(v_scales.tobytes(), (2, 64), 8,
                                  "k_group_column_e8m0"),
        "output_scales": Resource(bytes([127] * 128), (2, 64), 8,
                                  "n_group_row_e8m0"),
        "golden_bf16": Resource(_bytes(tuple(code for row in codes for code in row), 2),
                                (64, 64), 16, "row_major_bf16"),
    }
    policy = {
        "schema": "mx_gemmini.attention_pv_proxy.v1",
        "stage": "gqa_pv_head0_block0",
        "exp_policy": "torch_exp_bf16_proxy_for_mu_fexp",
        "causal_mask": "first_key_block_fully_visible",
        "output_oracle": "dim16_reduced_precision_product_and_accumulator",
        "source_header_sha256": _sha(header_bytes),
        "qk_golden_bf16_sha256": _sha(qk_resources["golden_bf16"]),
        "muon_requant_source_sha256": _sha(muon_source.read_bytes()),
        "source_arrays_sha256": {name: _sha(resources[name].data) for name in (
            "activation", "activation_scales", "weight", "weight_scales")},
    }
    return resources, policy
