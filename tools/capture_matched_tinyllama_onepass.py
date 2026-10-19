"""Recapture the matched one-pass input while reusing exact checkpoint storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--weights-source", type=Path, required=True)
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.out_dir.exists() or args.threads <= 0:
        parser.error("output must be fresh and threads positive")
    import m2m
    import numpy as np
    import torch
    from m2m.capture.bundle import write_bundle
    from m2m.capture.torch_mlir_bridge import module_to_text
    from m2m.transforms.externalize import externalize_weights
    from workloads.tiny_llama.loader import get_model_and_inputs

    archive = args.reference_archive.resolve(strict=True)
    identity = json.loads((archive / "input_identity.json").read_text())
    weights = args.weights_source.resolve(strict=True)
    if sha(weights) != identity["compiler_weights_sha256"]:
        raise ValueError("checkpoint bytes differ from the archived independent source comparison")
    with np.load(args.input_ids, allow_pickle=False) as values:
        ids = values["in0"].reshape(-1).astype(np.int64)
    with np.load(archive / "input_ids.npz", allow_pickle=False) as values:
        expected_ids = values["in0"].reshape(-1)
    if not np.array_equal(ids, expected_ids):
        raise ValueError("token IDs differ from the archived comparison")
    out = args.out_dir.resolve()
    out.mkdir()
    ids_path = out / "token_ids.npy"
    np.save(ids_path, ids)
    os.environ.update(M2M_SEQ=str(len(ids)), M2M_LLAMA_TOKEN_IDS=str(ids_path),
                      M2M_LLAMA_ATTENTION="eager", M2M_LLAMA_LAYERS="",
                      M2M_LLAMA_CHECKPOINT_LAYERS="", M2M_LLAMA_SESSION="",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    torch.set_num_threads(args.threads)
    model, inputs = get_model_and_inputs()
    converted = m2m.convert(model, inputs, backend="fx_importer")
    if not converted.ok or converted.module is None or converted.exported_program is None:
        raise ValueError("matched model did not convert to complete reusable typed IR")
    bundle = out / "capture"
    bundle.mkdir()
    reused = externalize_weights(converted.exported_program, converted.module,
                                 str(bundle / "weights.safetensors"), reuse_source=weights)
    converted.mlir_text = module_to_text(converted.module)
    write_bundle(model, inputs, bundle, conversion_result=converted, capture_regions=False)
    golden = np.load(bundle / "golden.npy", allow_pickle=False).reshape(len(ids), -1)
    reference = np.load(archive / "fp32_late_logits.npy", allow_pickle=False)
    top = np.load(archive / "fp32_argmax_160.npy", allow_pickle=False).reshape(-1)
    if not np.array_equal(np.argmax(golden, axis=-1), top):
        raise ValueError("recaptured FP32 top tokens differ from the archived forward")
    late = golden[-len(reference):]
    difference = float(np.max(np.abs(late - reference)))
    # This verifies the float capture correspondence, not quantized model quality.
    if not np.allclose(late, reference, rtol=1e-5, atol=3e-5):
        raise ValueError("recaptured FP32 logits differ from the archived reference")
    report = {
        "schema": "mx_gemmini.matched_onepass_recapture.v1",
        "weights": reused, "shared_checkpoint_inode": os.path.samefile(weights, bundle / "weights.safetensors"),
        "reference_identity_sha256": sha(archive / "input_identity.json"),
        "source_mlir_sha256": sha(bundle / "model.mlir"), "golden_sha256": sha(bundle / "golden.npy"),
        "input_ids_sha256": sha(args.input_ids), "tokens": len(ids),
        "fp32_top_tokens_equal": True, "fp32_late_max_absolute_difference": difference,
        "threads": args.threads,
    }
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
