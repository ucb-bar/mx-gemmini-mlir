"""Verify that a model2MLIR one-pass bundle and Nicolas source use one input.

The source generator takes checkpoint tensors from its earlier layer capture
and 160-token activations from a later capture. This check binds both to the
receipt-bound MLIR bundle before their Spike cycle counts are compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


LAYER_WEIGHTS = {
    "Wq": "self_attn.q_proj.weight",
    "Wk": "self_attn.k_proj.weight",
    "Wv": "self_attn.v_proj.weight",
    "Wo": "self_attn.o_proj.weight",
    "Wg": "mlp.gate_proj.weight",
    "Wu": "mlp.up_proj.weight",
    "Wd": "mlp.down_proj.weight",
    "w_in_ln": "input_layernorm.weight",
    "w_post_ln": "post_attention_layernorm.weight",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64, copy=False).ravel()
    bb = b.astype(np.float64, copy=False).ravel()
    return float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-weights", type=Path, required=True,
                        help="directory with Nicolas's layer0..21.npz and model.npz")
    parser.add_argument("--source-activations", type=Path, required=True,
                        help="directory with Nicolas's 160-token model.npz")
    parser.add_argument("--compiler-bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    receipt_path = args.compiler_bundle / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema"] == "m2m.capture-receipt.v1"
    for name in ("model.mlir", "weights.safetensors", "inputs.npz", "golden.npy"):
        artifact = args.compiler_bundle / name
        record = receipt["artifacts"][name]
        assert not artifact.is_symlink() and artifact.stat().st_size == record["bytes"]
        assert sha(artifact) == record["sha256"], name

    with np.load(args.compiler_bundle / "inputs.npz", allow_pickle=False) as inputs:
        compiler_ids = inputs["in0"].reshape(-1).copy()
    with np.load(args.source_activations / "model.npz", allow_pickle=False) as source:
        source_ids = source["meta_token_ids"].reshape(-1).copy()
        source_logits = source["logits"].copy()
    assert compiler_ids.shape == source_ids.shape == (160,)
    assert np.array_equal(compiler_ids, source_ids)

    weights_path = args.compiler_bundle / "weights.safetensors"
    matched = 0
    with safe_open(weights_path, framework="np") as compiler:
        for layer in range(22):
            with np.load(args.source_weights / f"layer{layer}.npz", allow_pickle=False) as source:
                for source_name, suffix in LAYER_WEIGHTS.items():
                    expected = source[source_name]
                    actual = compiler.get_tensor(f"lm.model.layers.{layer}.{suffix}")
                    if expected.ndim == 2:
                        actual = actual.T
                    assert np.array_equal(expected, actual), (layer, source_name)
                    matched += 1
        with np.load(args.source_weights / "model.npz", allow_pickle=False) as source:
            for source_name, name in (("w_final_ln", "lm.model.norm.weight"),
                                      ("lm_head", "lm.lm_head.weight")):
                expected = source[source_name]
                actual = compiler.get_tensor(name)
                if expected.ndim == 2:
                    actual = actual.T
                assert np.array_equal(expected, actual), source_name
                matched += 1
    assert matched == 200

    compiler_logits = np.load(args.compiler_bundle / "golden.npy", allow_pickle=False)
    assert compiler_logits.shape == (1, 160, 32000)
    assert source_logits.shape == (160, 32000)
    cosines = [_cosine(source_logits[index], compiler_logits[0, index])
               for index in range(127, 160)]
    tops = [int(np.argmax(source_logits[index]) ==
                np.argmax(compiler_logits[0, index]))
            for index in range(127, 160)]
    report = {
        "schema": "mx_gemmini.nicolas_tinyllama_onepass_input_identity.v1",
        "scope": "one 160-token causal forward, all 22 layers and LM head",
        "source_and_compiler_token_ids_exact": True,
        "source_and_compiler_fp32_weights_exact": True,
        "matched_fp32_weight_tensors": matched,
        "token_ids_sha256": hashlib.sha256(compiler_ids.tobytes()).hexdigest(),
        "source_activation_sha256": sha(args.source_activations / "model.npz"),
        "source_weight_capture_sha256": {
            path.name: sha(path) for path in sorted(args.source_weights.glob("*.npz"))},
        "compiler_capture_receipt_sha256": sha(receipt_path),
        "compiler_weights_sha256": receipt["artifacts"]["weights.safetensors"]["sha256"],
        "compiler_mlir_sha256": receipt["artifacts"]["model.mlir"]["sha256"],
        "source_bf16_vs_compiler_fp32_prefill_cosine": cosines[0],
        "source_bf16_vs_compiler_fp32_late_cosine_mean": float(np.mean(cosines[1:])),
        "source_bf16_vs_compiler_fp32_late_cosine_min": min(cosines[1:]),
        "source_bf16_vs_compiler_fp32_top_matches": sum(tops),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "matched_fp32_weight_tensors",
        "source_bf16_vs_compiler_fp32_prefill_cosine",
        "source_bf16_vs_compiler_fp32_late_cosine_mean",
        "source_bf16_vs_compiler_fp32_top_matches")}, indent=2))


if __name__ == "__main__":
    main()
