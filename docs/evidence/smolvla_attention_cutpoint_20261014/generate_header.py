"""Generate Nicolas-style MX operands and a block-major SmolVLA mask."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def f32_words(values: np.ndarray) -> str:
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return ",\n".join(
        "    { " + ", ".join("0x%08x" % int(word) for word in row) + " }"
        for row in words
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocc-tools", type=Path, required=True,
                        help="Directory containing the pinned gen_matmul_llama.py")
    parser.add_argument("--cutpoint", type=Path,
                        default=Path(__file__).with_name("cutpoint.npz"))
    parser.add_argument("--head-index", type=int, default=0,
                        help="Attention head from the captured 15-head flow step")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).with_name("smol_head0.h"))
    args = parser.parse_args()
    if not 0 <= args.head_index < 15:
        parser.error("--head-index must be in 0..14")

    sys.path.insert(0, str(args.rocc_tools.resolve()))
    import gen_matmul_llama as generator

    with np.load(args.cutpoint, allow_pickle=False) as arrays:
        q = arrays["q"][args.head_index]
        kt = arrays["kt"][args.head_index]
        v = arrays["v"][args.head_index]
        p = arrays["p"][args.head_index]
        captured_mask = arrays["mask"][0, 0]
    keys = kt.shape[1]
    assert keys in (113, 163)
    assert q.shape == (50, 64) and kt.shape == (64, keys)
    assert v.shape == (keys, 64) and captured_mask.shape == (50, keys)
    physical_keys = ((keys + 63) // 64) * 64
    blocks = physical_keys // 64

    padded_q = np.zeros((64, 64), dtype=np.float32)
    padded_kt = np.zeros((64, physical_keys), dtype=np.float32)
    padded_v = np.zeros((physical_keys, 64), dtype=np.float32)
    padded_q[:50] = q
    padded_kt[:, :keys] = kt
    padded_v[:keys] = v
    mask = np.zeros((64, physical_keys), dtype=bool)
    mask[:50, :keys] = captured_mask
    mask[50:, 0] = True  # give discarded padded rows one finite softmax entry
    mask_blocks = np.stack([
        np.where(mask[:, block * 64:(block + 1) * 64], 0, 0xFF80).astype(np.uint16)
        for block in range(blocks)
    ])

    fmt = generator.FORMATS["fp8"]
    q_codes, q_scales, _ = generator.quantize(padded_q, axis="row", f=fmt)
    kt_codes, kt_scales, _ = generator.quantize(padded_kt, axis="col", f=fmt)
    v_codes, v_scales, _ = generator.quantize(padded_v, axis="col", f=fmt)
    reference = np.zeros((64, 64), dtype=np.float32)
    reference[:50] = p @ v

    def array(name: str, values: np.ndarray, width: int, typ: str, dims: str) -> str:
        return (f"static const {typ} {name}{dims} __attribute__((aligned(64))) = {{\n"
                f"{generator._rows(values, width)}\n}};\n")

    mask_rows = ",\n".join(
        "  {\n" + generator._rows(block, 4) + "\n  }" for block in mask_blocks
    )
    parts = [
        "#ifndef SMOL_ATTENTION_HEAD0_H", "#define SMOL_ATTENTION_HEAD0_H",
        "#include <stdint.h>", "#define ATTN_SQ 64", f"#define ATTN_SK {physical_keys}",
        "#define ATTN_D 64", "#define ATTN_VALID_Q 50", "#define ATTN_HEADS 1",
        "#define ATTN_CAUSAL 0", "#define ATTN_ARBITRARY_MASK 1",
        "#define ATTN_SCALE_IN_Q 0",
        array("Q_IN", q_codes, 2, "uint8_t", "[ATTN_SQ][ATTN_D]"),
        array("Q_SCALES", q_scales.T, 2, "uint8_t", "[ATTN_D/32][ATTN_SQ]"),
        array("KT_IN", kt_codes, 2, "uint8_t", "[ATTN_D][ATTN_SK]"),
        array("KT_SCALES", kt_scales, 2, "uint8_t", "[ATTN_D/32][ATTN_SK]"),
        array("V_IN", v_codes, 2, "uint8_t", "[ATTN_SK][ATTN_D]"),
        array("V_SCALES", v_scales, 2, "uint8_t", "[ATTN_SK/32][ATTN_D]"),
        (f"static const uint16_t MASK_BLOCK_BF16[{blocks}][ATTN_SQ][64] "
         f"__attribute__((aligned(64))) = {{\n{mask_rows}\n}};"),
        f"static const uint32_t O_REF_F_F32[ATTN_SQ][ATTN_D] = {{\n{f32_words(reference)}\n}};",
        "#endif",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts) + "\n")
    print(f"wrote {args.out}; head {args.head_index}; keys {keys}; "
          f"valid mask fraction {captured_mask.mean():.6f}")


if __name__ == "__main__":
    main()
