"""Archive a completed isolated FP32-state SmolVLA flow-step Spike run."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from verify import HERE, metrics, sha, timestep_diagnostic, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--link", type=Path, required=True)
    parser.add_argument("--derived-session", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run.read_text())
    if run.get("status") != "complete_numeric_qualification_pending":
        raise ValueError("isolated replay has not finished")
    if run.get("link_receipt_sha256") != sha(args.link):
        raise ValueError("isolated run is not bound to its link")
    dumps = run.get("output_dumps", [])
    if (len(dumps) != 1 or (dumps[0]["program"], dumps[0]["step"], dumps[0]["output"])
            != (0, 0, 0)):
        raise ValueError("isolated replay needs the flow output dump")
    source = HERE.parent / "smolvla_full_action_values_20261014"
    full_run = json.loads((source / "tap_run.json").read_text())
    full_link = json.loads((source / "tap_link.json").read_text())
    link = json.loads(args.link.read_text())
    if (link["stage_object_sha256"]["flow_denoise"]
            != full_link["stage_object_sha256"]["flow_denoise"]
            or link["shared_weight_sha256"] != full_link["shared_weight_sha256"]
            or link["profile_sha256"] != full_link["profile_sha256"]):
        raise ValueError("isolated replay changed the flow compiler objects or profile")
    (HERE / "run.json").write_bytes(args.run.read_bytes())
    (HERE / "link.json").write_bytes(args.link.read_bytes())
    (HERE / "full_run.json").write_bytes((source / "tap_run.json").read_bytes())
    (HERE / "full_link.json").write_bytes((source / "tap_link.json").read_bytes())
    (HERE / "derived_replay_receipt.json").write_bytes(
        (args.derived_session / "derived_replay_receipt.json").read_bytes())
    with np.load(args.derived_session / "stages/flow_denoise/session_quality_fp32.npz",
                 allow_pickle=False) as archive:
        expected = np.ascontiguousarray(archive["actions"], dtype="<f4")
    if expected.shape != (1, 1, 50, 32):
        raise ValueError("isolated reference output has wrong shape")
    (HERE / "fp32_step9.bin").write_bytes(expected.tobytes())
    output_path = args.run.parent / dumps[0]["path"]
    shutil.copyfile(output_path, HERE / "isolated_step9.bin")
    shutil.copyfile(source / "output_dumps/p1_s9_o0.bin", HERE / "full_step9.bin")
    (HERE / "spike.log.gz").write_bytes(gzip.compress(
        (args.run.parent / "spike.log").read_bytes(), mtime=0))
    shutil.copyfile(args.run.parent / "gemmini_perf.txt",
                    HERE / "gemmini_perf_report.txt")
    isolated = np.frombuffer((HERE / "isolated_step9.bin").read_bytes(), dtype="<f4")
    full = np.frombuffer((HERE / "full_step9.bin").read_bytes(), dtype="<f4")
    ref = expected.reshape(-1)
    names = ("run.json", "link.json", "full_run.json", "full_link.json",
             "derived_replay_receipt.json", "fp32_step9.bin", "isolated_step9.bin",
             "full_step9.bin", "spike.log.gz", "gemmini_perf_report.txt")
    receipt = {"schema": "mx_gemmini.smolvla_reference_step_replay_result.v1",
               "scope": "one flow step at captured FP32 state input; no prefix or action stage",
               "reference_step": 9, "cycle_ratio_valid": False,
               "metrics": {"isolated_vs_fp32": metrics(isolated, ref),
                           "full_vs_fp32": metrics(full, ref)},
               "timestep_diagnostic": timestep_diagnostic(full_run),
               "artifact_sha256": {name: sha(HERE / name) for name in names}}
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verify(), sort_keys=True))


if __name__ == "__main__":
    main()
