"""Archive a completed full TinyLlama logits run after identity checks."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

from verify import FAST, HERE, sha, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tap-run", "tap-link", "comparison", "prefill-reference",
                 "decode-reference", "tap-log", "tap-perf"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "tap_run.json": args.tap_run,
        "tap_link.json": args.tap_link,
        "fp32_comparison.json": args.comparison,
        "prefill_reference.npy": args.prefill_reference,
        "decode_reference.npz": args.decode_reference,
        "tap_perf_report.txt": args.tap_perf,
    }
    tap = json.loads(args.tap_run.read_text())
    link = json.loads(args.tap_link.read_text())
    fast_path = FAST / "compiler/mx_session_run.json"
    fast_link_path = FAST / "compiler_link.json"
    fast = json.loads(fast_path.read_text())
    fast_link = json.loads(fast_link_path.read_text())
    report = json.loads(args.comparison.read_text())
    assert tap["complete"] is True and tap["exit_code"] == 0
    assert tap["status"] == "complete_numeric_qualification_pending"
    assert tap["output_digests"] == fast["output_digests"]
    assert len(tap["output_digests"]) == 99
    assert link["stage_object_sha256"] == fast_link["stage_object_sha256"]
    assert link["shared_weight_sha256"] == fast_link["shared_weight_sha256"]
    assert tap["link_receipt_sha256"] == sha(args.tap_link)
    assert tap["log_sha256"] == sha(args.tap_log)
    assert report["run_receipt_sha256"] == sha(args.tap_run)
    assert report["link_receipt_sha256"] == sha(args.tap_link)
    assert report["fast_run_receipt_sha256"] == sha(fast_path)
    assert report["fast_link_receipt_sha256"] == sha(fast_link_path)
    assert report["prefill_reference_sha256"] == sha(args.prefill_reference)
    assert report["decode_reference_sha256"] == sha(args.decode_reference)
    assert report["output_digests_match_fast_compiler"] is True
    dumps = tap.get("output_dumps", [])
    assert len(dumps) == 33
    for row in dumps:
        relative = Path(row["path"])
        assert relative.parent == Path("output_dumps")
        source = args.tap_run.parent / relative
        assert row["bytes"] == source.stat().st_size == 128000
        assert sha(source) == row["sha256"]

    for name, source in sources.items():
        shutil.copyfile(source, HERE / name)
    (HERE / "tap_spike.log.gz").write_bytes(
        gzip.compress(args.tap_log.read_bytes(), mtime=0))
    artifacts = [*sources, "tap_spike.log.gz"]
    (HERE / "output_dumps").mkdir(exist_ok=True)
    for row in dumps:
        relative = Path(row["path"])
        shutil.copyfile(args.tap_run.parent / relative, HERE / relative)
        artifacts.append(str(relative))
    receipt = {
        "schema": "mx_gemmini.tinyllama_full_logits.v1",
        "scope": "one prefill and 32 decode steps",
        "numerical_qualification": "not_established",
        "output_digests_match_fast_compiler": True,
        "fast_compiler_cycles": fast["total_performance"]["cycles"],
        "tap_cycles": tap["total_performance"]["cycles"],
        "fast_run_sha256": sha(fast_path),
        "fast_link_sha256": sha(fast_link_path),
        "prefill_metrics": report["prefill_metrics"],
        "decode_summary": report["decode_summary"],
        "artifact_sha256": {name: sha(HERE / name) for name in artifacts},
    }
    (HERE / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    verify()
    print("archived full TinyLlama logits diagnostic")


if __name__ == "__main__":
    main()
