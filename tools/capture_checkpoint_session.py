"""Capture a local causal checkpoint with a fixed prefill/decode input stream.

The supplied capture contributes token IDs and reusable checkpoint storage.
Both program graphs, lifted constants and floating references are generated
again from the loaded model. Existing capture receipts remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint-dir", "weights-source", "input-capture", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.out_dir.exists() or args.threads <= 0:
        parser.error("output must be fresh and threads must be positive")

    import m2m
    import numpy as np
    import torch
    from m2m.capture.bundle import (
        capture_session_trajectory,
        write_multi_program_bundle,
    )
    from m2m.capture.bundle_integrity import verify_bundle_integrity
    from m2m.capture.causal_session import make_causal_program_capture
    from m2m.capture.torch_mlir_bridge import module_to_text
    from m2m.transforms.externalize import externalize_weights
    from transformers import AutoModelForCausalLM

    source = args.input_capture.resolve(strict=True)
    verify_bundle_integrity(source)
    paths = [
        source / "stages/prefill/inputs.npz",
        source / "stages/decode/session_inputs.npz",
    ]
    with np.load(paths[0], allow_pickle=False) as values:
        prefill = values["in0"]
    with np.load(paths[1], allow_pickle=False) as values:
        decode = values["decode_tokens"]
    if (
        prefill.shape != (1, 128)
        or decode.shape != (32, 1, 1)
        or prefill.dtype != np.int64
        or decode.dtype != np.int64
    ):
        raise ValueError("expected the original 128+32 token stream")
    tokens = np.concatenate([prefill.reshape(-1), decode.reshape(-1)])
    out = args.out_dir.absolute()
    out.mkdir()
    np.save(out / "token_ids.npy", tokens)
    os.environ["M2M_CAPTURE_SESSION_TOKEN_IDS"] = str(out / "token_ids.npy")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    print("load local checkpoint", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.checkpoint_dir.resolve(strict=True)),
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    )
    if tokens.min() < 0 or tokens.max() >= model.config.vocab_size:
        raise ValueError("token stream exceeds the checkpoint vocabulary")
    # Give the exported LM head independent storage, preserving its exact values.
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    capture, _ = make_causal_program_capture(
        model.eval(),
        env_prefix="M2M_CAPTURE_SESSION",
        checkpoint=args.checkpoint_id,
        full_checkpoint=True,
        prefill_tokens=128,
        decode_tokens=32,
    )
    external = capture.external_runtime_session()
    programs = external.bundle_programs()
    print("capture floating trajectory", flush=True)
    decoded = programs[1]
    decoded["session"]["quality"]["reference_values"] = capture_session_trajectory(
        decoded["model"], tuple(decoded["inputs"]), decoded["session"]
    )
    source_weights = args.weights_source.resolve(strict=True)
    weight_digest = sha(source_weights)
    records = {}
    for program in programs:
        name = program["name"]
        print("convert", name, flush=True)
        converted = m2m.convert(
            program["model"], tuple(program["inputs"]), backend="fx_importer"
        )
        if (
            not converted.ok
            or converted.module is None
            or converted.exported_program is None
        ):
            raise ValueError(
                "capture did not produce a complete reusable program: " + name
            )
        stage = out / "capture/stages" / name
        stage.mkdir(parents=True)
        records[name] = externalize_weights(
            converted.exported_program,
            converted.module,
            str(stage / "weights.safetensors"),
            reuse_source=source_weights,
        )
        converted.mlir_text = module_to_text(converted.module)
        program["conversion_result"] = converted
    metadata = dict(external.metadata)
    metadata["checkpoint_revision"] = args.checkpoint_revision
    summary = write_multi_program_bundle(programs, metadata, out / "capture")
    verify_bundle_integrity(out / "capture")
    if sha(source_weights) != weight_digest:
        raise ValueError("reused checkpoint changed during capture")
    report = {
        "schema": "mx_gemmini.checkpoint_session_capture.v1",
        "scope": "fresh_graphs_constants_and_float_references_not_quantized_execution",
        "checkpoint_id": args.checkpoint_id,
        "checkpoint_revision": args.checkpoint_revision,
        "config_sha256": sha(args.checkpoint_dir / "config.json"),
        "weights_sha256": weight_digest,
        "producer_sha256": sha(Path(__file__)),
        "input_sources_sha256": {str(p): sha(p) for p in paths},
        "token_ids_sha256": sha(out / "token_ids.npy"),
        "weight_reuse": records,
        "summary": summary,
    }
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print("fresh session captured", flush=True)


if __name__ == "__main__":
    main()
