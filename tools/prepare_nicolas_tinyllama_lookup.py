"""Give Nicolas's one-pass TinyLlama source the compiler's token lookup.

The source generator stores precomputed BF16 embeddings in its model blob.
For an end-to-end timing boundary, this tool instead embeds the checkpoint's
FP32 token table and makes the source program gather and round it inside the
existing timed region. The selected BF16 values must exactly match the
source generator's precomputed inputs before any source program is built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


WEIGHT = "lm.model.embed_tokens.weight"
SYMBOL = "_binary_embedding_fp32_bin_start"
PINNED_SOURCE_SHA256 = (
    "ffd14dc19420dfb803cb55e49013a7aaf957ce6dd703b4a375212eed2fec9254"
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fp32_embedding(path: Path) -> tuple[np.memmap, int, int]:
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    item = header[WEIGHT]
    if item["dtype"] != "F32" or item["shape"] != [32000, 2048]:
        raise ValueError("unexpected TinyLlama token table")
    first, last = item["data_offsets"]
    if last - first != 32000 * 2048 * 4:
        raise ValueError("token table has an unexpected byte length")
    start = 8 + header_size + first
    table = np.memmap(path, dtype="<f4", mode="r", offset=start,
                      shape=(32000, 2048))
    return table, start, last - first


def to_bf16_f32(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype="<f4").view("<u4")
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16)
    return (rounded.astype("<u4") << 16).view("<f4")


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"source anchor occurs {source.count(old)} times: {old!r}")
    return source.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-c", type=Path, required=True)
    parser.add_argument("--source-activations", type=Path, required=True,
                        help="Nicolas's 160-token model.npz")
    parser.add_argument("--compiler-bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.compiler_bundle / "inputs.npz", allow_pickle=False) as data:
        ids = np.asarray(data["in0"].reshape(-1), dtype=np.int64)
    with np.load(args.source_activations, allow_pickle=False) as data:
        source_ids = data["meta_token_ids"].reshape(-1)
        expected = data["embed_out"]
    if (ids.shape != (160,) or expected.shape != (160, 2048)
            or not np.array_equal(ids, source_ids)
            or np.any(ids < 0) or np.any(ids >= 32000)):
        raise ValueError("source and compiler token inputs differ")

    weight_path = args.compiler_bundle / "weights.safetensors"
    table, start, length = fp32_embedding(weight_path)
    actual = to_bf16_f32(table[ids])
    if not np.array_equal(actual, expected):
        raise ValueError("FP32 lookup does not reproduce source BF16 embeddings")

    if digest(args.source_c) != PINNED_SOURCE_SHA256:
        raise ValueError("Nicolas source differs from the pinned 160-token patch")
    source = args.source_c.read_text()
    if "#define LLAMA_M   160" not in (args.source_c.parent /
                                        "llama_model_m160.h").read_text():
        raise ValueError("source header is not the 160-token program")
    constants = ", ".join(str(int(item)) for item in ids)
    declarations = (
        f"\nextern const unsigned char {SYMBOL}[];\n"
        f"static const uint32_t lookup_token_ids[160] = {{{constants}}};\n"
        "static void lookup_embeddings(uint16_t *out) {\n"
        f"  const float *table = (const float *){SYMBOL};\n"
        "  for (int row = 0; row < LLAMA_M; row++) {\n"
        "    const float *word = table + (size_t)lookup_token_ids[row] * LLAMA_D;\n"
        "    for (int col = 0; col < LLAMA_D; col++)\n"
        "      out[(size_t)row * LLAMA_D + col] = mx_f32_to_bf16_rne(word[col]);\n"
        "  }\n"
        "}\n"
    )
    source = replace_once(source, "#include LLAMA_MODEL_HEADER\n",
                          "#include LLAMA_MODEL_HEADER\n" + declarations)
    source = replace_once(source,
                          "  memcpy(X_hw, EMBED_OUT, sizeof(X_hw));\n",
                          "  lookup_embeddings(X_hw);\n")
    logit_tail = (
        '      printf("\\n");\n'
        '    }\n'
        '#endif\n'
        '  }\n'
        '#else\n'
        '  printf("grade final hidden vs model: rel_fro %d ppm\\n",\n'
    )
    source = replace_once(source, logit_tail, (
        '      printf("\\n");\n'
        '    }\n'
        '    printf("source_argmax_160");\n'
        '    for (int m = 0; m < LLAMA_M; m++) {\n'
        '      int best = 0;\n'
        '      float value = mx_bf16_to_f32(logits_hw[(size_t)m * LLAMA_V]);\n'
        '      for (int v = 1; v < LLAMA_V; v++) {\n'
        '        float candidate = mx_bf16_to_f32(logits_hw[(size_t)m * LLAMA_V + v]);\n'
        '        if (candidate > value) { best = v; value = candidate; }\n'
        '      }\n'
        '      printf(" %d", best);\n'
        '    }\n'
        '    printf("\\n");\n'
        '#endif\n'
        '  }\n'
        '#else\n'
        '  printf("grade final hidden vs model: rel_fro %d ppm\\n",\n'
    ))

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=False)
    table_path = out / "embedding_fp32.bin"
    with weight_path.open("rb") as input_stream, table_path.open("wb") as output:
        input_stream.seek(start)
        remaining = length
        while remaining:
            block = input_stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError("truncated checkpoint embedding table")
            output.write(block)
            remaining -= len(block)
    source_path = out / "llama_model_lookup.c"
    source_path.write_text(source)
    receipt = {
        "schema": "mx_gemmini.nicolas_tinyllama_lookup_source.v1",
        "scope": "token IDs to logits; FP32 gather and BF16 rounding inside source timer",
        "source_c_sha256": digest(args.source_c),
        "lookup_source_sha256": digest(source_path),
        "compiler_weights_sha256": digest(weight_path),
        "compiler_inputs_sha256": digest(args.compiler_bundle / "inputs.npz"),
        "source_activations_sha256": digest(args.source_activations),
        "embedding_table_sha256": digest(table_path),
        "embedding_bytes": length,
        "token_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "matched_bf16_elements": int(expected.size),
        "bf16_mismatches": 0,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                 sort_keys=True) + "\n")
    print(json.dumps({"matched_bf16_elements": int(expected.size),
                      "embedding_bytes": length, "out": str(out)}))


if __name__ == "__main__":
    main()
