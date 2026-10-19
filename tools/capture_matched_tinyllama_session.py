"""Freshly capture matched prefill/decode programs with shared checkpoint storage.

Never repair a changed old capture by editing its receipt. Fresh source-derived
conversions and FP32 trajectory references retain their own new identities.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tools.capture_matched_tinyllama_onepass import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-source", type=Path, required=True)
    parser.add_argument("--reference-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("output must be fresh")
    import m2m
    import numpy as np
    import torch
    from m2m.capture.bundle import (
        capture_session_trajectory,
        write_multi_program_bundle,
    )
    from m2m.capture.torch_mlir_bridge import module_to_text
    from m2m.transforms.externalize import externalize_weights
    from workloads.tiny_llama.loader import get_model_and_inputs

    archive = args.reference_archive.resolve(strict=True)
    identity = json.loads((archive / "input_identity.json").read_text())
    weights = args.weights_source.resolve(strict=True)
    if sha(weights) != identity["compiler_weights_sha256"]:
        raise ValueError("session checkpoint differs from the independent-forward comparison")
    with np.load(archive / "input_ids.npz", allow_pickle=False) as values:
        ids = values["in0"].reshape(-1).astype(np.int64)
    if ids.size != 160:
        raise ValueError("matched session requires the original 160 token IDs")
    out = args.out_dir.resolve()
    out.mkdir()
    np.save(out / "token_ids.npy", ids)
    os.environ.update(M2M_LLAMA_TOKEN_IDS=str(out / "token_ids.npy"),
                      M2M_LLAMA_ATTENTION="eager", M2M_LLAMA_LAYERS="",
                      M2M_LLAMA_CHECKPOINT_LAYERS="", M2M_LLAMA_SESSION="e2e",
                      M2M_PREFILL_TOKENS="128", M2M_DECODE_TOKENS="32",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    torch.set_num_threads(2)
    capture, _ = get_model_and_inputs()
    external = capture.external_runtime_session()
    programs = external.bundle_programs()
    decode = programs[1]
    reference = capture_session_trajectory(decode["model"], tuple(decode["inputs"]), decode["session"])
    decode["session"]["quality"]["reference_values"] = reference
    reused = {}
    for program in programs:
        name = program["name"]
        print("convert", name, flush=True)
        converted = m2m.convert(program["model"], tuple(program["inputs"]), backend="fx_importer")
        if not converted.ok or converted.module is None or converted.exported_program is None:
            raise ValueError("session program did not produce reusable complete typed IR: " + name)
        stage = out / "capture/stages" / name
        stage.mkdir(parents=True)
        reused[name] = externalize_weights(converted.exported_program, converted.module,
                                           str(stage / "weights.safetensors"), reuse_source=weights)
        converted.mlir_text = module_to_text(converted.module)
        program["conversion_result"] = converted
        print("weights bound", name, flush=True)
    summary = write_multi_program_bundle(programs, dict(external.metadata), out / "capture")
    from merlin.llvmlower.session_bundle import load

    session = load(out / "capture")
    if [(p.name, p.steps) for p in session.programs] != [("prefill", 1), ("decode", 32)]:
        raise ValueError("new session schedule differs from the requested 128+32 trajectory")
    if sha(weights) != identity["compiler_weights_sha256"] or any(
            sha(p.bundle / "weights.safetensors") != identity["compiler_weights_sha256"]
            or not os.path.samefile(weights, p.bundle / "weights.safetensors")
            for p in session.programs):
        raise ValueError("session checkpoint changed during capture or is not shared storage")
    report = {"schema": "mx_gemmini.matched_cached_session_capture.v1",
              "scope": "fresh_fp32_128_prefill_and_32_cached_decodes_not_quantized_execution",
              "reference_identity_sha256": sha(archive / "input_identity.json"),
              "weights_sha256": sha(weights), "reused": reused, "summary": summary,
              "programs": {p.name: {"model_sha256": sha(p.bundle / "model.mlir"),
                  "receipt_sha256": sha(p.bundle / "capture_receipt.json")} for p in session.programs}}
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
