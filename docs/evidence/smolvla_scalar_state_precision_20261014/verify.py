"""Check the portable SmolVLA scalar-state Spike evidence."""

from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path
import struct

import numpy as np


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "smolvla_reference_step_replay_20261014"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify() -> dict:
    receipt = json.loads((HERE / "receipt.json").read_text())
    if receipt.get("schema") != "mx_gemmini.smolvla_scalar_state_precision.v1":
        raise ValueError("unexpected receipt schema")
    for name, digest in receipt["artifact_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError(f"changed artifact: {name}")
    old_catalog = json.loads((HERE / "old_catalog.json").read_text())
    new_catalog = json.loads((HERE / "new_catalog.json").read_text())
    for field in ("bindings", "covered_contractions", "source_sha256",
                  "profile_sha256", "vpu_add", "vpu_reduce", "vpu_softmax",
                  "vpu_unary"):
        if old_catalog[field] != new_catalog[field]:
            raise ValueError(f"unexpected catalog change: {field}")
    old_regions = set(old_catalog["vpu_elementwise"]["source_regions"])
    new_regions = set(new_catalog["vpu_elementwise"]["source_regions"])
    if old_regions - new_regions != {"add_96#add0"} or new_regions - old_regions:
        raise ValueError("scalar-state route is not the only removed VPU region")
    if (len(old_catalog["vpu_elementwise"]["kernels"]) != 19
            or len(new_catalog["vpu_elementwise"]["kernels"]) != 18):
        raise ValueError("unexpected VPU kernel count")

    old_link = json.loads((PRIOR / "link.json").read_text())
    prior_receipt = json.loads((PRIOR / "receipt.json").read_text())
    for name in ("link.json", "isolated_step9.bin", "fp32_step9.bin"):
        if sha(PRIOR / name) != prior_receipt["artifact_sha256"][name]:
            raise ValueError(f"preceding replay artifact changed: {name}")
    new_link = json.loads((HERE / "new_link.json").read_text())
    for field in ("profile_sha256", "shared_weight_sha256", "programs"):
        if old_link[field] != new_link[field]:
            raise ValueError(f"replay identity changed: {field}")
    if new_link["profile_sha256"] != new_catalog["profile_sha256"]:
        raise ValueError("replay profile differs from catalog")
    old_objects = old_link["stage_object_sha256"]["flow_denoise"]
    new_objects = new_link["stage_object_sha256"]["flow_denoise"]
    if (old_objects["device/device_catalog_shim.o"]
            != new_objects["device/device_catalog_shim.o"]):
        raise ValueError("MX shim object changed")
    if (old_objects["model.o"] == new_objects["model.o"]
            or old_objects["device_catalog/mx_catalog.o"]
            == new_objects["device_catalog/mx_catalog.o"]):
        raise ValueError("scalar routing did not rebuild model and catalog")

    run = json.loads((HERE / "new_run.json").read_text())
    if (run.get("status") != "complete_numeric_qualification_pending"
            or run["link_receipt_sha256"] != sha(HERE / "new_link.json")
            or run["timing_config"] != "e4m3_vpu"):
        raise ValueError("new Spike run is not bound to the selected profile")
    if (hashlib.sha256(gzip.decompress(
            (HERE / "new_spike.log.gz").read_bytes())).hexdigest()
            != run["log_sha256"]
            or sha(HERE / "new_perf.txt") != run["timing_report_sha256"]):
        raise ValueError("Spike log or performance report changed")
    dumps = {(row["program"], row["step"], row["output"]): row
             for row in run["output_dumps"]}
    for output, name, size in ((0, "new_step9.bin", 6400),
                               (2, "new_timestep.bin", 4)):
        row = dumps[0, 0, output]
        if row["bytes"] != size or row["sha256"] != sha(HERE / name):
            raise ValueError(f"new output {output} does not match Spike receipt")
    old_output = PRIOR / "isolated_step9.bin"
    if sha(HERE / "new_step9.bin") != sha(old_output):
        raise ValueError("flow result changed after scalar route fix")
    timestep = struct.unpack("<f", (HERE / "new_timestep.bin").read_bytes())[0]
    expected = float(np.float32(np.float32(0.09999992698431015)
                                + np.float32(-0.1)))
    if timestep != expected:
        raise ValueError("new timestep is not the FP32 source update")
    old = np.fromfile(old_output, dtype="<f4").astype(np.float64)
    reference = np.fromfile(PRIOR / "fp32_step9.bin", dtype="<f4").astype(np.float64)
    cosine = float(np.dot(old, reference) /
                   (np.linalg.norm(old) * np.linalg.norm(reference)))
    return {"removed_vpu_region": "add_96#add0",
            "flow_output_equal": True, "flow_vs_fp32_cosine": cosine,
            "timestep_f32": timestep,
            "isolated_spike_cycles": run["total_performance"]["cycles"]}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
