"""Capture current model2MLIR small LLaMA and emit its MX precision proposal."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.model_precision_plan import build_model_precision_plan
from mx_gemmini_support.target_profile import load_profile, profile_sha256


MODEL2MLIR_REVISION = "e9ded36eb85abf2d9097ac4dc11457c825853388"
LOADER_SHA256 = "aa859e6ffb6ec765c372511a97d63aba6205b8a5dd092cb2ac5b36864d9c2c5b"
GRAPH_SHA256 = "cf7c4a36bb72a319289fadd296514aa201f9bb187f3e2547496404a3ec7d6099"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    source = args.model2mlir_root.resolve()
    out = args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    loader = source / "workloads/small_llama/loader.py"
    if _sha(loader.read_bytes()) != LOADER_SHA256:
        raise ValueError("small LLaMA model2MLIR workload source differs")
    if (source / ".git").exists():
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=source, text=True).strip()
        if revision != MODEL2MLIR_REVISION:
            raise ValueError("small LLaMA needs pinned model2MLIR revision")
    sys.path.insert(0, str(source))
    import m2m
    from m2m.coverage import opaque_report
    import torch
    from workloads.small_llama.loader import get_model_and_inputs

    if Path(m2m.__file__).resolve().parents[1] != source:
        raise ValueError("model2MLIR imported from another source tree")
    torch.manual_seed(0)
    model, inputs = get_model_and_inputs()
    capture = m2m.convert(model, inputs, backend="fx_importer")
    if (not capture.ok or capture.path_taken != "fx_importer" or
            opaque_report(capture.mlir_text)):
        raise ValueError("small LLaMA model2MLIR capture is opaque or failed")
    graph = capture.mlir_text.encode()
    if _sha(graph) != GRAPH_SHA256:
        raise ValueError("small LLaMA model2MLIR graph differs")
    profile = load_profile(args.profile.resolve(), rtl_root=args.rtl_root.resolve())
    worklist = build_model2mlir_worklist(capture.mlir_text, profile)
    plan = build_model_precision_plan(worklist, profile)
    roles = {role: sum(site["role"] == role for site in plan["sites"])
             for role in ("qk", "pv")}
    if (worklist["contraction_region_count"] != 19 or
            roles != {"qk": 2, "pv": 2} or
            plan["selected_format_counts"] !=
            {"host": 6, "mxfp4": 9, "mxfp8": 4}):
        raise ValueError("small LLaMA precision proposal differs")
    out.mkdir(parents=True)
    (out / "small_llama.model2mlir.mlir.gz").write_bytes(gzip.compress(graph, mtime=0))
    (out / "worklist.json").write_text(json.dumps(worklist, indent=2, sort_keys=True) + "\n")
    (out / "precision_proposal.json").write_text(json.dumps(plan, indent=2,
                                                          sort_keys=True) + "\n")
    receipt = {
        "schema": "mx_gemmini.small_llama_latest_model2mlir_precision_probe.v1",
        "status": "frontend_and_precision_proposal_only_no_executable_model",
        "model2mlir_revision": MODEL2MLIR_REVISION,
        "model2mlir_loader_sha256": LOADER_SHA256,
        "capture_backend": "fx_importer",
        "opaque_operation_count": 0,
        "profile_sha256": profile_sha256(profile),
        "frontend_mlir_sha256": GRAPH_SHA256,
        "compressed_frontend_sha256": _sha((out / "small_llama.model2mlir.mlir.gz").read_bytes()),
        "worklist_sha256": _sha((out / "worklist.json").read_bytes()),
        "precision_proposal_sha256": _sha((out / "precision_proposal.json").read_bytes()),
        "contraction_regions": 19,
        "precision_counts": plan["selected_format_counts"],
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                  sort_keys=True) + "\n")
    print("small LLaMA: 19 contractions, FP4 9, FP8 attention 4, host 6")


if __name__ == "__main__":
    main()
