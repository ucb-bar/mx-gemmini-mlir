"""Compare TinyLlama prefill VPU routes on the same captured FP8 MX work.

The full-session prefill may be inspected before its later decode steps finish.
Only two completed standalone stage runs support a cycle ratio; the linked
session prefill has a different timing boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from tools.compare_tinyllama_session import metrics


_IDENTITY = (
    "capture_model_sha256", "checkpoint_weights_sha256", "profile_sha256",
    "precision_overrides_sha256", "selected_operations",
    "expected_selected_operations", "prepack_checkpoint_weights",
)
_VPU_CLASSES = ("softmax", "add", "elementwise", "unary", "reduce")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def full_prefill(log: str) -> tuple[np.ndarray, str, int]:
    """Verify and decode the one completed session prefill output."""
    value = re.findall(r"^SESSION_HEX 0 0 0 (\d+) ([0-9a-f]+)$", log, re.M)
    digest = re.findall(r"^SESSION_SHA 0 0 0 (\d+) ([0-9a-f]{64})$", log, re.M)
    cycles = re.findall(r"^SESSION_PERF 0 0 (\d+) \d+$", log, re.M)
    if len(value) != 1 or len(digest) != 1 or len(cycles) != 1:
        raise ValueError("full session has no complete, unique prefill result")
    raw = bytes.fromhex(value[0][1])
    if (int(value[0][0]) != 128000 or int(digest[0][0]) != len(raw)
            or hashlib.sha256(raw).hexdigest() != digest[0][1]):
        raise ValueError("full-session prefill output digest differs")
    return np.frombuffer(raw, dtype="<f4").reshape(1, 32000).copy(), digest[0][1], int(cycles[0])


def stage_output(log: str, run: dict) -> tuple[str, int, int]:
    """Check a complete digest-only stage result and its one top token."""
    digest = re.findall(r"^OUT_SHA256 f32le 32000 128000 ([0-9a-f]{64})$",
                        log, re.M)
    top = re.findall(r"^ARGMAX 1 (\d+)$", log, re.M)
    cycles = re.findall(r"^METRIC cycles (\d+)$", log, re.M)
    evidence = run.get("output_digests") or {}
    first = evidence.get("first_typed") or {}
    indexed = evidence.get("indexed") or []
    if (len(digest) != 1 or len(top) != 1 or len(cycles) != 1
            or first.get("sha256") != digest[0]
            or first.get("bytes") != 128000
            or not indexed or indexed[0].get("sha256") != digest[0]
            or indexed[0].get("index") != 0
            or indexed[0].get("shape") != [1, 32000]):
        raise ValueError("stage output differs from its typed receipt")
    return digest[0], int(top[0]), int(cycles[0])


def compare(safe_build_path: Path, safe_run_path: Path,
            full_build_path: Path, full_link_path: Path,
            full_log_path: Path, stage_path: Path,
            reference_path: Path, full_run_path: Path | None = None) -> dict:
    if reference_path.resolve() != (stage_path / "golden.npy").resolve():
        raise ValueError("reference must be the captured prefill golden")
    safe_build = json.loads(safe_build_path.read_text())
    full_build = json.loads(full_build_path.read_text())
    safe_run = json.loads(safe_run_path.read_text())
    full_link = json.loads(full_link_path.read_text())
    if (any(safe_build.get(key) != full_build.get(key) for key in _IDENTITY)
            or safe_build.get("selected_operations") != 200
            or safe_build.get("enabled_vpu_classes") != ["add", "reduce", "unary"]
            or full_build.get("enabled_vpu_classes") != list(sorted(_VPU_CLASSES))
            or safe_build.get("status") !=
            "linked_diagnostic_partial_vpu_numeric_qualification_pending"
            or full_build.get("status") !=
            "linked_experimental_numeric_qualification_pending"):
        raise ValueError("builds do not isolate the intended VPU classes")
    safe_catalog = json.loads(Path(safe_build["catalog_manifest"]).read_text())
    full_catalog = json.loads(Path(full_build["catalog_manifest"]).read_text())
    if (safe_catalog["bindings"] != full_catalog["bindings"]
            or safe_catalog["kernels"] != full_catalog["kernels"]
            or len(safe_catalog["bindings"]) != 200
            or any(row["target_dtypes"][:2] != ["f8E4M3FN", "f8E4M3FN"]
                   for row in safe_catalog["bindings"])):
        raise ValueError("the 200 FP8 MX matrix bindings differ")
    if (safe_run.get("done") is not True
            or safe_run.get("exit_code") != 0
            or safe_run.get("diagnostic_partial_vpu") is not True
            or safe_run.get("build_receipt_sha256") != sha(safe_build_path)
            or safe_run.get("spike_environment", {}).get("GEMMINI_PERF_CONFIG")
            != "e4m3_vpu"
            or safe_run.get("spike_environment", {}).get("GEMMINI_PERF_SET")
            != "mem.host_tracking=0"
            or "--isa=rv64gc_zicntr" not in safe_run.get("command", [])):
        raise ValueError("safe VPU run is incomplete or uses different Spike settings")
    if (full_link.get("schema") != "mx_gemmini.linked_merlin_session.v1"
            or full_link.get("profile_sha256") != full_build["profile_sha256"]
            or full_link["programs"][0]["name"] != "prefill"
            or full_link["programs"][0]["capture_model_sha256"] !=
            full_build["capture_model_sha256"]
            or (Path(full_link["session_root"]) / "stages/prefill").resolve()
            != stage_path.resolve()
            or sha(stage_path / "model.mlir") !=
            safe_build["capture_model_sha256"]):
        raise ValueError("full-session link does not contain the matched prefill")
    linked = full_link["stage_object_sha256"]["prefill"]
    for name in ("model.o", "device_catalog/mx_catalog.o"):
        if linked[name] != sha(full_build_path.parent / name):
            raise ValueError(f"full session uses a different {name}")
    safe_log_path = safe_run_path.parent / "spike.log"
    safe_log = safe_log_path.read_text()
    if safe_run.get("log_sha256") != sha(safe_log_path):
        raise ValueError("safe VPU log differs from its receipt")
    reference = np.load(reference_path, allow_pickle=False)
    safe_digest, safe_top, safe_cycles = stage_output(safe_log, safe_run)
    full_log_bytes = full_log_path.read_bytes()
    if b"SESSION_BEGIN 1 0" not in full_log_bytes:
        raise ValueError("linked session has not finished the prefill step")
    full_prefix = full_log_bytes.split(b"SESSION_BEGIN 1 0", 1)[0]
    full, full_digest, full_cycles = full_prefill(
        full_prefix.decode("utf-8", errors="replace"))
    report = {
        "schema": "mx_gemmini.tinyllama_vpu_route_comparison.v3",
        "scope": "128-token prefill, first 1x32000 FP32 output",
        "numerical_scope": "stage receipts emit full-output hashes and top tokens only",
        "linked_cycle_ratio_valid": False,
        "linked_cycle_boundary_difference":
            "safe run is a standalone stage; full run is a linked session",
        "identity": {key: safe_build[key] for key in _IDENTITY},
        "safe_build_sha256": sha(safe_build_path),
        "full_build_sha256": sha(full_build_path),
        "safe_run_sha256": sha(safe_run_path),
        "full_link_sha256": sha(full_link_path),
        "full_prefill_log_prefix_sha256":
            hashlib.sha256(full_prefix).hexdigest(),
        "captured_input_sha256": sha(stage_path / "inputs.npz"),
        "reference_sha256": sha(reference_path),
        "linked_full_prefill_output_sha256": full_digest,
        "safe_prefill_output_sha256": safe_digest,
        "safe_prefill_top_token": safe_top,
        "fp32_reference_top_token": int(np.argmax(reference)),
        "safe_vs_fp32_top_token_equal":
            safe_top == int(np.argmax(reference)),
        "safe_matches_linked_full_output": safe_digest == full_digest,
        "safe_standalone_cycles": safe_cycles,
        "linked_full_prefill_cycles": full_cycles,
        "safe_vpu_lowered_regions": {
            name: safe_build[f"vpu_{name}_lowered_regions"]
            for name in _VPU_CLASSES},
        "full_vpu_lowered_regions": {
            name: full_build[f"vpu_{name}_lowered_regions"]
            for name in _VPU_CLASSES},
        "linked_full_vs_fp32": metrics(full, reference),
    }
    if full_run_path is not None:
        full_run = json.loads(full_run_path.read_text())
        full_stage_log_path = full_run_path.parent / "spike.log"
        full_stage_log = full_stage_log_path.read_text()
        timing_keys = ("GEMMINI_MODE", "GEMMINI_PERF_CONFIG",
                       "GEMMINI_PERF_SET")
        if (full_run.get("done") is not True
                or full_run.get("exit_code") != 0
                or full_run.get("build_receipt_sha256") != sha(full_build_path)
                or full_run.get("spike_sha256") != safe_run.get("spike_sha256")
                or full_run.get("extension_sha256") !=
                safe_run.get("extension_sha256")
                or any((full_run.get("spike_environment") or {}).get(key) !=
                       (safe_run.get("spike_environment") or {}).get(key)
                       for key in timing_keys)
                or full_run.get("command", [])[:-1] !=
                safe_run.get("command", [])[:-1]
                or full_run.get("log_sha256") != sha(full_stage_log_path)):
            raise ValueError("standalone runs differ in input or Spike settings")
        full_stage_digest, full_stage_top, full_stage_cycles = stage_output(
            full_stage_log, full_run)
        if safe_cycles <= 0 or full_stage_cycles <= 0:
            raise ValueError("standalone stage cycles are incomplete")
        report.update({
            "full_standalone_run_sha256": sha(full_run_path),
            "full_standalone_output_sha256": full_stage_digest,
            "full_standalone_top_token": full_stage_top,
            "full_standalone_matches_linked_full_output":
                full_stage_digest == full_digest,
            "safe_matches_full_standalone_output":
                safe_digest == full_stage_digest,
            "equal_output_speedup_valid": safe_digest == full_stage_digest,
            "standalone_cycle_ratio_valid": True,
            "standalone_cycles": {
                "safe_vpu": safe_cycles,
                "full_vpu": full_stage_cycles,
                "safe_over_full": safe_cycles / full_stage_cycles,
            },
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("safe_build", "safe_run", "full_build", "full_link",
                 "full_log", "stage", "reference", "out"):
        parser.add_argument("--" + name.replace("_", "-"), required=True,
                            type=Path)
    parser.add_argument("--full-run", type=Path,
                        help="completed standalone full-VPU run for equal-boundary cycles")
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("--out must be a new file")
    report = compare(args.safe_build, args.safe_run, args.full_build,
                     args.full_link, args.full_log, args.stage,
                     args.reference, args.full_run)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out),
                      "safe_top": report["safe_prefill_top_token"],
                      "safe_matches_full":
                      report["safe_matches_linked_full_output"],
                      "full_cosine": report["linked_full_vs_fp32"]["cosine"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
