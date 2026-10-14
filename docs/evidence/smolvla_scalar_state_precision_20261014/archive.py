"""Archive the SmolVLA scalar-state route and its one-step Spike replay."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

from verify import HERE, sha, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-catalog", type=Path, required=True)
    parser.add_argument("--new-catalog", type=Path, required=True)
    parser.add_argument("--new-link", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.new_run.read_text())
    if run.get("status") != "complete_numeric_qualification_pending":
        raise ValueError("new Spike replay has not completed")
    if run.get("link_receipt_sha256") != sha(args.new_link):
        raise ValueError("new replay is not bound to its link receipt")
    copies = {
        "old_catalog.json": args.old_catalog,
        "new_catalog.json": args.new_catalog,
        "new_link.json": args.new_link,
        "new_run.json": args.new_run,
        "new_step9.bin": args.new_run.parent / "output_dumps/p0_s0_o0.bin",
        "new_timestep.bin": args.new_run.parent / "output_dumps/p0_s0_o2.bin",
        "new_perf.txt": args.new_run.parent / "gemmini_perf.txt",
    }
    HERE.mkdir(parents=True, exist_ok=True)
    for name, source in copies.items():
        shutil.copyfile(source, HERE / name)
    (HERE / "new_spike.log.gz").write_bytes(gzip.compress(
        (args.new_run.parent / "spike.log").read_bytes(), mtime=0))
    names = list(copies) + ["new_spike.log.gz"]
    receipt = {
        "schema": "mx_gemmini.smolvla_scalar_state_precision.v1",
        "scope": "one isolated flow step on captured FP32 state and prefix cache",
        "artifact_sha256": {name: sha(HERE / name) for name in names},
    }
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                   sort_keys=True) + "\n")
    print(json.dumps(verify(), sort_keys=True))


if __name__ == "__main__":
    main()
