"""Match all two-layer source checkpoint tensors to model2MLIR's capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


HERE = Path(__file__).resolve().parent
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


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True,
                        help="Nicolas capture directory containing layer0/1 and model.npz")
    parser.add_argument("--compiler-stage", type=Path, required=True,
                        help="model2MLIR two-layer prefill stage")
    args = parser.parse_args()
    receipt = json.loads((HERE / "l2_receipt.json").read_text())
    for name, expected in receipt["source_capture_npz_sha256"].items():
        assert file_hash(args.source_capture / name) == expected, name
    weights = args.compiler_stage / "weights.safetensors"
    inputs = args.compiler_stage / "inputs.npz"
    assert file_hash(weights) == receipt["compiler_checkpoint_weights_sha256"]
    assert file_hash(inputs) == receipt["input_ids_npz_sha256"]
    count = 0
    with safe_open(weights, framework="np") as tensors:
        for layer in range(2):
            with np.load(args.source_capture / f"layer{layer}.npz") as source:
                for source_name, suffix in LAYER_WEIGHTS.items():
                    captured = source[source_name]
                    compiled = tensors.get_tensor(
                        f"lm.model.layers.{layer}.{suffix}")
                    if captured.ndim == 2:
                        compiled = compiled.T
                    assert np.array_equal(captured, compiled), (layer, source_name)
                    count += 1
        with np.load(args.source_capture / "model.npz") as source:
            for source_name, weight_name in (
                ("w_final_ln", "lm.model.norm.weight"),
                ("lm_head", "lm.lm_head.weight"),
            ):
                captured = source[source_name]
                compiled = tensors.get_tensor(weight_name)
                if captured.ndim == 2:
                    compiled = compiled.T
                assert np.array_equal(captured, compiled), source_name
                count += 1
            with np.load(inputs) as input_data:
                assert np.array_equal(source["meta_token_ids"],
                                      input_data["in0"].reshape(-1))
    assert count == receipt["exact_fp32_checkpoint_tensors_matched"]
    print(f"verified {count} exact FP32 checkpoint tensors and all 64 token IDs")


if __name__ == "__main__":
    main()
