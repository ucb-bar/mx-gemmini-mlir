"""Archive a completed full SmolVLA run with the FP32 scalar-state route."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

from verify import HERE, sha, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--link", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run.read_text())
    if (run.get("status") != "complete_numeric_qualification_pending"
            or run.get("complete") is not True
            or run.get("link_receipt_sha256") != sha(args.link)):
        raise ValueError("full Spike run is incomplete or link identity differs")
    comparison = json.loads(args.comparison.read_text())
    if (comparison.get("run_receipt_sha256") != sha(args.run)
            or comparison.get("link_receipt_sha256") != sha(args.link)):
        raise ValueError("FP32 comparison does not bind to this run")
    HERE.mkdir(parents=True, exist_ok=True)
    files = {"run.json": args.run, "link.json": args.link,
             "fp32_comparison.json": args.comparison,
             "gemmini_perf.txt": args.run.parent / "gemmini_perf.txt"}
    for name, path in files.items():
        shutil.copyfile(path, HERE / name)
    (HERE / "spike.log.gz").write_bytes(gzip.compress(
        (args.run.parent / "spike.log").read_bytes(), mtime=0))
    names = [*files, "spike.log.gz"]
    (HERE / "output_dumps").mkdir(exist_ok=True)
    for row in run["output_dumps"]:
        relative = Path(row["path"])
        if relative.parent != Path("output_dumps"):
            raise ValueError("output dump path escapes expected archive directory")
        source = args.run.parent / relative
        if sha(source) != row["sha256"]:
            raise ValueError(f"output digest differs: {relative}")
        shutil.copyfile(source, HERE / relative)
        names.append(str(relative))
    receipt = {"schema": "mx_gemmini.smolvla_scalar_full_action.v1",
               "scope": "one prefix, ten flow steps, one action with FP32 scalar-state route",
               "numerical_qualification": "not_established",
               "cycle_ratio_valid": False,
               "artifact_sha256": {name: sha(HERE / name) for name in names}}
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                   sort_keys=True) + "\n")
    print(json.dumps(verify(), sort_keys=True))


if __name__ == "__main__":
    main()
