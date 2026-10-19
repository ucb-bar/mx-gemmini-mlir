"""Check a completed run against the published model's complete output evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.spike_timing import reject_deadlocked_timing
from tools.model_release import ROOT, evidence, reader, sha


def check(name, out, *, stage_build=None, link_path=None, run_path=None):
    if out.exists():
        raise ValueError("result path must be fresh")
    item, files = evidence(name)
    if item["kind"] == "forward":
        if stage_build is None or link_path or run_path:
            raise ValueError("forward needs --stage-build only")
        observed = json.loads((stage_build / "mx_stage_probe.json").read_text())
        record = item["stages"]["forward"]
        if (observed["capture_model_sha256"] != record["capture_model_sha256"]
                or observed["checkpoint_weights_sha256"] != record["weights_sha256"]
                or observed["execution_policy"] != json.loads((ROOT / record["policy"]["path"]).read_text())
                or observed["spike"]["exit_code"] != 0):
            raise ValueError("forward build identity or terminal status differs")
        raw = (stage_build / "spike.log").read_bytes()
        reject_deadlocked_timing(raw.decode())
        parser = reader(ROOT / "docs/evidence/matched_forward_20261018/verify.py")
        actual, metrics = parser.parse(raw)
        expected, _ = parser.parse(files["candidate/spike.log"])
        if actual.tobytes() != expected.tobytes():
            raise ValueError("forward output words differ from the published run")
        count = actual.size
        cycles = int(metrics["cycles"])
        console_sha = hashlib.sha256(raw).hexdigest()
    else:
        if not link_path or not run_path or stage_build:
            raise ValueError("session needs --link-receipt and --run-receipt only")
        from merlin.llvmlower.session_bundle import load
        from merlin.llvmlower.session_outputs import observed_output_dump_bytes

        from tools.compare_linked_session_cycles import _load

        link, run = _load(link_path, run_path)
        baseline_link = json.loads(files["candidate/link/mx_session_link.json"])
        baseline_run = json.loads(files["candidate/run/mx_session_run.json"])
        for key in ("programs", "bindings", "profile_sha256", "shared_weight_sha256",
                    "performance_schema", "preparation_performance_schema", "execution_timing_scope"):
            if link.get(key) != baseline_link.get(key):
                raise ValueError("session work or execution boundary differs: " + key)
        if sha(Path(link["elf"])) != link["elf_sha256"]:
            raise ValueError("actual linked ELF differs from its receipt")
        session_root = Path(link["session_root"])
        for key, raw in files.items():
            if key.startswith("capture/"):
                if (session_root / key.removeprefix("capture/")).read_bytes() != raw:
                    raise ValueError("captured session input changed: " + key)
        if run["output_digests"] != baseline_run["output_digests"]:
            raise ValueError("session output or persistent-state digest differs")
        raw = (run_path.parent / "spike.log").read_text()
        dumps = observed_output_dump_bytes(raw, load(session_root), link["dump_outputs"], run["output_digests"],
                                           byte_limit=link["dump_limit_bytes"])
        if len(dumps) != len(baseline_run["output_dumps"]):
            raise ValueError("complete output tensor roster differs")
        count = 0
        for row in baseline_run["output_dumps"]:
            key = (row["program"], row["step"], row["output"])
            expected = files["candidate/run/" + row["path"]]
            if dumps[key] != expected:
                raise ValueError("complete output tensor differs")
            count += len(expected) // 4
        cycles = run["comparison_performance"]["cycles"]
        console_sha = sha(run_path.parent / "spike.log")
    result = {"schema": "mx_gemmini.reproduced_model_outputs.v1", "model": name,
              "status": "complete_output_parity", "complete_f32_words_compared": count,
              "cycles": cycles, "published_cycles": item["cycles"], "console_sha256": console_sha,
              "scope": "complete outputs preserve the published numerical result; timing ratio requires separately matched tool and runtime identities"}
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tinyllama-forward", "tinyllama-cached", "smolvla"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage-build", type=Path)
    parser.add_argument("--link-receipt", type=Path)
    parser.add_argument("--run-receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.model, args.out, stage_build=args.stage_build,
                           link_path=args.link_receipt, run_path=args.run_receipt), indent=2))


if __name__ == "__main__":
    main()
