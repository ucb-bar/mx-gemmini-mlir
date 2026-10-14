"""Archive the completed SmolVLA compiled-state / FP32-prefix control."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from verify import FULL, HERE, REFERENCE, metrics, sha, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--link", type=Path, required=True)
    parser.add_argument("--derived-session", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run.read_text())
    if (run.get("status") != "complete_numeric_qualification_pending"
            or run.get("complete") is not True
            or run["link_receipt_sha256"] != sha(args.link)):
        raise ValueError("factorial Spike run is incomplete or link differs")
    inputs = args.derived_session / "stages/flow_denoise/inputs.npz"
    derived = args.derived_session / "derived_replay_receipt.json"
    if json.loads(derived.read_text())["derived_inputs_sha256"] != sha(inputs):
        raise ValueError("derived control input changed")
    paths = {"run.json": args.run, "link.json": args.link,
             "derived_replay_receipt.json": derived,
             "flow_step9.bin": args.run.parent / "output_dumps/p0_s0_o0.bin",
             "timestep_step9.bin": args.run.parent / "output_dumps/p0_s0_o2.bin",
             "gemmini_perf.txt": args.run.parent / "gemmini_perf.txt"}
    HERE.mkdir(parents=True, exist_ok=True)
    for name, source in paths.items():
        shutil.copyfile(source, HERE / name)
    (HERE / "spike.log.gz").write_bytes(gzip.compress(
        (args.run.parent / "spike.log").read_bytes(), mtime=0))
    names = [*paths, "spike.log.gz"]
    expected = np.fromfile(REFERENCE / "fp32_step9.bin", dtype="<f4")
    factor = np.fromfile(HERE / "flow_step9.bin", dtype="<f4")
    full = np.fromfile(FULL / "output_dumps/p1_s9_o0.bin", dtype="<f4")
    receipt = {"schema": "mx_gemmini.smolvla_prefix_factor_result.v1",
               "scope": "compiled step-8 state and time; captured FP32 prefix cache",
               "cycle_ratio_valid": False,
               "numerical_qualification": "not_established",
               "metrics": {"factor_vs_fp32": metrics(factor, expected),
                           "full_vs_fp32": metrics(full, expected),
                           "factor_vs_full": metrics(factor, full)},
               "artifact_sha256": {name: sha(HERE / name) for name in names}}
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                   sort_keys=True) + "\n")
    print(json.dumps(verify(), sort_keys=True))


if __name__ == "__main__":
    main()
