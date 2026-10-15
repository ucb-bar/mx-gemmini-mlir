"""Verify the two standalone FP8 TinyLlama VPU routes and their output scope."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import runpy
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "tinyllama_all_fp8_full_session_20261014"
sys.path.insert(0, str(HERE.parents[2]))

from tools.compare_tinyllama_session import metrics  # noqa: E402
from tools.compare_tinyllama_vpu_routes import (  # noqa: E402
    full_prefill, stage_output,
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify() -> None:
    manifest = json.loads((HERE / "archive_manifest.json").read_text())
    assert manifest["schema"] == "mx_gemmini.tinyllama_fp8_vpu_route_archive.v1"
    for name, digest in manifest["files"].items():
        assert sha(HERE / name) == digest, name
    runpy.run_path(str(BASELINE / "verify.py"))["verify"]()

    report = json.loads((HERE / "comparison.json").read_text())
    safe_build = json.loads((HERE / "safe_build.json").read_text())
    full_build = json.loads((BASELINE / "prefill_build.json").read_text())
    safe_catalog = json.loads((HERE / "safe_catalog.json").read_text())
    full_catalog = json.loads((BASELINE / "prefill_catalog.json").read_text())
    safe_run = json.loads((HERE / "safe_run.json").read_text())
    full_run = json.loads((HERE / "full_run.json").read_text())
    link = json.loads((BASELINE / "link.json").read_text())
    assert report["schema"] == "mx_gemmini.tinyllama_vpu_route_comparison.v3"
    for name, path in (
        ("safe_build_sha256", HERE / "safe_build.json"),
        ("full_build_sha256", BASELINE / "prefill_build.json"),
        ("safe_run_sha256", HERE / "safe_run.json"),
        ("full_standalone_run_sha256", HERE / "full_run.json"),
        ("full_link_sha256", BASELINE / "link.json"),
        ("reference_sha256", BASELINE / "prefill_golden.npy"),
    ):
        assert report[name] == sha(path), name
    assert link["profile_sha256"] == safe_build["profile_sha256"]
    assert safe_catalog["bindings"] == full_catalog["bindings"]
    assert safe_catalog["kernels"] == full_catalog["kernels"]
    assert len(safe_catalog["bindings"]) == 200
    assert all(row["target_dtypes"][:2] == ["f8E4M3FN", "f8E4M3FN"]
               for row in safe_catalog["bindings"])
    for key in (
        "capture_model_sha256", "checkpoint_weights_sha256",
        "precision_overrides_sha256", "selected_operations",
    ):
        assert safe_build[key] == full_build[key] == report["identity"][key]
    assert safe_build["enabled_vpu_classes"] == ["add", "reduce", "unary"]
    assert full_build["enabled_vpu_classes"] == [
        "add", "elementwise", "reduce", "softmax", "unary"]
    assert report["safe_vpu_lowered_regions"] == {
        "add": 88, "elementwise": 0, "reduce": 45,
        "softmax": 0, "unary": 156,
    }
    assert report["full_vpu_lowered_regions"] == {
        "add": 88, "elementwise": 358, "reduce": 45,
        "softmax": 22, "unary": 156,
    }

    for run, run_path, build_path, log_path in (
        (safe_run, HERE / "safe_run.json", HERE / "safe_build.json",
         HERE / "safe_spike.log"),
        (full_run, HERE / "full_run.json", BASELINE / "prefill_build.json",
         HERE / "full_spike.log"),
    ):
        assert run["done"] and run["exit_code"] == 0
        assert run["build_receipt_sha256"] == sha(build_path)
        assert run["log_sha256"] == sha(log_path)
        assert run["spike_environment"]["GEMMINI_PERF_CONFIG"] == "e4m3_vpu"
        assert run["spike_environment"]["GEMMINI_PERF_SET"] == \
            "mem.host_tracking=0"
        assert sha(run_path) in (
            report["safe_run_sha256"], report["full_standalone_run_sha256"])
    assert safe_run["spike_sha256"] == full_run["spike_sha256"]
    assert safe_run["extension_sha256"] == full_run["extension_sha256"]
    assert safe_run["command"][:-1] == full_run["command"][:-1]

    safe_digest, safe_top, safe_cycles = stage_output(
        (HERE / "safe_spike.log").read_text(), safe_run)
    full_digest, full_top, full_cycles = stage_output(
        (HERE / "full_spike.log").read_text(), full_run)
    assert safe_digest == report["safe_prefill_output_sha256"]
    assert full_digest == report["full_standalone_output_sha256"]
    assert safe_top == report["safe_prefill_top_token"] == 362
    assert full_top == report["full_standalone_top_token"] == 29889
    assert safe_cycles == report["safe_standalone_cycles"] == 10_804_276_937
    assert full_cycles == report["standalone_cycles"]["full_vpu"] == \
        12_276_460_508
    assert report["standalone_cycles"]["safe_over_full"] == \
        safe_cycles / full_cycles

    full_log = gzip.decompress((BASELINE / "spike.log.gz").read_bytes())
    prefix = full_log.split(b"SESSION_BEGIN 1 0", 1)[0]
    assert hashlib.sha256(prefix).hexdigest() == \
        report["full_prefill_log_prefix_sha256"]
    full_output, linked_digest, linked_cycles = full_prefill(
        prefix.decode("utf-8"))
    assert linked_digest == full_digest == \
        report["linked_full_prefill_output_sha256"]
    assert linked_cycles == report["linked_full_prefill_cycles"]
    reference = np.load(BASELINE / "prefill_golden.npy", allow_pickle=False)
    observed = metrics(full_output, reference)
    for key, value in observed.items():
        expected = report["linked_full_vs_fp32"][key]
        if isinstance(value, float):
            assert abs(value - expected) < 1e-12, key
        else:
            assert value == expected, key
    assert safe_digest != full_digest
    assert report["standalone_cycle_ratio_valid"] is True
    assert report["equal_output_speedup_valid"] is False
    print("same FP8 work and Spike settings, distinct outputs, and both cycles verified")


if __name__ == "__main__":
    verify()
