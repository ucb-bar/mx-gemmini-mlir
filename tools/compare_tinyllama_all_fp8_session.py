"""Check a complete all-FP8 TinyLlama session against captured FP32 logits.

The capture uses fixed synthetic tokens. This reports numerical drift for all
33 outputs, not generation quality or a compiler-versus-source speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.compare_tinyllama_session import metrics


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare(run_path: Path, link_path: Path, prefill_path: Path,
            decode_path: Path, session_root: Path) -> dict:
    run = json.loads(run_path.read_text())
    link = json.loads(link_path.read_text())
    builds = {
        "prefill": json.loads(prefill_path.read_text()),
        "decode": json.loads(decode_path.read_text()),
    }
    if (run.get("schema") != "mx_gemmini.merlin_session_spike_run.v1"
            or run.get("complete") is not True or run.get("exit_code") != 0
            or run.get("status") != "complete_numeric_qualification_pending"
            or run.get("link_receipt_sha256") != sha(link_path)
            or run.get("timing_config") != "e4m3_vpu"
            or run.get("timing_overrides") != "mem.host_tracking=0"
            or run.get("timing_mode") != "both"
            or link.get("schema") != "mx_gemmini.linked_merlin_session.v1"
            or Path(link["session_root"]).resolve() != session_root.resolve()
            or [row["name"] for row in link["programs"]]
            != ["prefill", "decode"]
            or [row["steps"] for row in link["programs"]] != [1, 32]
            or link.get("dump_outputs") != [
                {"program": 0, "name": "prefill", "output": 0, "bytes": 128000},
                {"program": 1, "name": "decode", "output": 0, "bytes": 128000}]
            or link.get("dump_limit_bytes") != 128000):
        raise ValueError("session identities, timing, or output taps differ")
    for name, row in zip(("prefill", "decode"), link["programs"]):
        build = builds[name]
        manifest = json.loads(Path(build["catalog_manifest"]).read_text())
        if (build.get("schema") != "mx_gemmini.merlin_stage_probe.v1"
                or build.get("profile_sha256") != link["profile_sha256"]
                or build.get("capture_model_sha256") !=
                row["capture_model_sha256"]
                or build.get("selected_operations") != 200
                or build.get("prepack_checkpoint_weights") is not True
                or build.get("enabled_vpu_classes") !=
                ["add", "elementwise", "reduce", "softmax", "unary"]
                or len(manifest["bindings"]) != 200
                or any(binding["target_dtypes"][:2] !=
                       ["f8E4M3FN", "f8E4M3FN"]
                       for binding in manifest["bindings"])):
            raise ValueError(f"{name} is not the selected all-FP8 MX+VPU build")
        for obj in ("model.o", "device_catalog/mx_catalog.o"):
            if sha((prefill_path if name == "prefill" else decode_path).parent
                   / obj) != link["stage_object_sha256"][name][obj]:
                raise ValueError(f"{name} object differs from linked session")
    if (builds["prefill"]["checkpoint_weights_sha256"] !=
            builds["decode"]["checkpoint_weights_sha256"]
            or len(run.get("output_digests", [])) != 99
            or len(run.get("step_performance", [])) != 33
            or sum(row["cycles"] for row in run["step_performance"])
            != run["total_performance"]["cycles"]):
        raise ValueError("session weights, outputs, or cycle totals differ")

    golden = np.load(session_root / "stages/prefill/golden.npy",
                     allow_pickle=False)
    with np.load(session_root / "stages/decode/session_quality_fp32.npz",
                 allow_pickle=False) as archive:
        decode = archive["logits"]
    if (golden.shape != (1, 32000) or decode.shape != (32, 1, 32000)
            or golden.dtype != decode.dtype or golden.dtype != np.float32):
        raise ValueError("captured FP32 references have unexpected shapes")
    expected = {(0, 0, 0)} | {(1, step, 0) for step in range(32)}
    dumps = {(row["program"], row["step"], row["output"]): row
             for row in run.get("output_dumps", [])}
    if len(run.get("output_dumps", [])) != 33 or set(dumps) != expected:
        raise ValueError("all 33 session logits must be dumped")
    digests = {(row["program"], row["step"], row["output"]): row
               for row in run["output_digests"]}
    observed = {}
    for key, row in dumps.items():
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("output dump path escapes the run directory")
        path = run_path.parent / relative
        if (row["bytes"] != 128000 or path.stat().st_size != 128000
                or sha(path) != row["sha256"]
                or row["sha256"] != digests[key]["sha256"]):
            raise ValueError("output dump differs from its receipt or digest")
        observed[key] = np.frombuffer(path.read_bytes(), dtype="<f4")\
            .reshape(1, 32000).copy()
    prefill = metrics(observed[0, 0, 0], golden)
    decoded = [{"step": step,
                **metrics(observed[1, step, 0], decode[step])}
               for step in range(32)]
    return {
        "schema": "mx_gemmini.tinyllama_all_fp8_session_fp32.v1",
        "scope": "128-token prefill plus 32 cached decode calls",
        "reference_kind": "captured_unquantized_fp32",
        "numerical_qualification": "not_established",
        "run_receipt_sha256": sha(run_path),
        "link_receipt_sha256": sha(link_path),
        "prefill_build_receipt_sha256": sha(prefill_path),
        "decode_build_receipt_sha256": sha(decode_path),
        "prefill_reference_sha256":
            sha(session_root / "stages/prefill/golden.npy"),
        "decode_reference_sha256":
            sha(session_root / "stages/decode/session_quality_fp32.npz"),
        "total_modeled_cycles": run["total_performance"]["cycles"],
        "prefill_metrics": prefill,
        "decode_step_metrics": decoded,
        "decode_summary": {
            "steps": 32,
            "cosine_mean": float(np.mean([row["cosine"] for row in decoded])),
            "cosine_min": min(row["cosine"] for row in decoded),
            "top_token_matches": sum(row["top_token_equal"] for row in decoded),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-receipt", "link-receipt", "prefill-build-receipt",
                 "decode-build-receipt", "session-root", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("--out must be a new file")
    report = compare(args.run_receipt, args.link_receipt,
                     args.prefill_build_receipt, args.decode_build_receipt,
                     args.session_root)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cycles": report["total_modeled_cycles"],
                      "prefill_cosine": report["prefill_metrics"]["cosine"],
                      "decode": report["decode_summary"]}, sort_keys=True))


if __name__ == "__main__":
    main()
