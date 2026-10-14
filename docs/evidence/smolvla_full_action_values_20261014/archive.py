"""Archive a finished full SmolVLA value run after checking its identities."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

from verify import FAST, HERE, sha, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tap-run", "tap-link", "old-run", "old-link",
                 "comparison", "goldens", "tap-log", "tap-perf"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    source_paths = {
        "tap_run.json": args.tap_run,
        "tap_link.json": args.tap_link,
        "old_run.json": args.old_run,
        "old_link.json": args.old_link,
        "fp32_comparison.json": args.comparison,
        "session_goldens.npz": args.goldens,
        "tap_perf_report.txt": args.tap_perf,
    }
    tap = json.loads(args.tap_run.read_text())
    old = json.loads(args.old_run.read_text())
    fast_path = FAST / "compiler/mx_session_run.json"
    fast = json.loads(fast_path.read_text())
    tap_link = json.loads(args.tap_link.read_text())
    old_link = json.loads(args.old_link.read_text())
    fast_link_path = FAST / "compiler_link.json"
    fast_link = json.loads(fast_link_path.read_text())
    report = json.loads(args.comparison.read_text())
    assert tap["status"] == old["status"] == fast["status"] == (
        "complete_numeric_qualification_pending")
    assert tap["output_digests"] == old["output_digests"] == fast["output_digests"]
    assert len(tap["output_digests"]) == 33
    assert tap_link["stage_object_sha256"] == old_link["stage_object_sha256"]
    assert (tap_link["shared_weight_sha256"]
            == old_link["shared_weight_sha256"]
            == fast_link["shared_weight_sha256"])
    assert tap["link_receipt_sha256"] == sha(args.tap_link)
    assert old["link_receipt_sha256"] == sha(args.old_link)
    assert report["run_receipt_sha256"] == sha(args.tap_run)
    assert report["link_receipt_sha256"] == sha(args.tap_link)
    assert report["reference_sha256"] == sha(args.goldens)
    assert report["final_action_matches_flow_bytes"] is True
    dumps = tap.get("output_dumps", [])
    assert len(dumps) == 11
    for row in dumps:
        relative = Path(row["path"])
        assert relative.parent == Path("output_dumps")
        assert sha(args.tap_run.parent / relative) == row["sha256"]
    for name, path in source_paths.items():
        shutil.copyfile(path, HERE / name)
    log = args.tap_log.read_bytes()
    (HERE / "tap_spike.log.gz").write_bytes(gzip.compress(log, mtime=0))
    artifact_names = [*source_paths, "tap_spike.log.gz"]
    (HERE / "output_dumps").mkdir(exist_ok=True)
    for row in dumps:
        relative = Path(row["path"])
        source = args.tap_run.parent / relative
        shutil.copyfile(source, HERE / relative)
        artifact_names.append(str(relative))
    receipt = {
        "schema": "mx_gemmini.smolvla_full_action_values.v1",
        "scope": "one prefix, ten flow steps, one final action",
        "numerical_qualification": "not_established",
        "cycle_ratio_valid": False,
        "cycle_ratio_reason": "tapped slow objects differ from optimized compiler objects",
        "output_digests_match_fast_compiler": True,
        "flow_outputs_compared": 10,
        "fast_compiler_cycles": fast["total_performance"]["cycles"],
        "tap_cycles": tap["total_performance"]["cycles"],
        "fast_run_sha256": sha(fast_path),
        "fast_link_sha256": sha(fast_link_path),
        "final_action_metrics": report["final_action_metrics"],
        "artifact_sha256": {name: sha(HERE / name) for name in artifact_names},
    }
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    verify()
    print("archived full SmolVLA action-value diagnostic")


if __name__ == "__main__":
    main()
