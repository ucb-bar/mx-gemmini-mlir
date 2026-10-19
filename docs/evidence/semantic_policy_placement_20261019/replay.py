"""Compare actual old and corrected default policies on recorded source inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import types
from pathlib import Path

from mx_gemmini_support import execution_policy
from mx_gemmini_support.target_profile import load_profile, profile_sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())


def compare(paths):
    before = (HERE / "execution_policy_before.py.txt").read_bytes()
    assert sha(before) == "31ce63493b9063f6ee54968fca92be477114cf43782f2ddee4573bfe3e42438d"
    legacy = types.ModuleType("mx_gemmini_support._frozen_policy_3662440")
    legacy.__package__ = "mx_gemmini_support"
    exec(compile(before, "execution_policy_before.py.txt", "exec"), legacy.__dict__)
    profile = load_profile(PROFILE)
    rows = []
    for path in paths:
        raw = path.read_bytes()
        source = (gzip.decompress(raw) if path.suffix == ".gz" else raw).decode()
        policy = execution_policy.make_policy(source, profile, preserve_singleton_reductions=True)
        previous = legacy.make_policy(source, profile, preserve_singleton_reductions=True)
        assert policy == previous, str(path)
        row = {"path": str(path), "file_sha256": sha(raw), "source_sha256": sha(source.encode()),
               "policy_sha256": canonical(policy), "matrix_sites": len(policy["matrix_formats"]),
               "vpu_site_counts": {name: len(sites) for name, sites in policy["vpu_placements"].items()},
               "changed_placements": []}
        rows.append(row)
        print(path.name, row["matrix_sites"], row["vpu_site_counts"], flush=True)
    return {"scope": "default_policy_preservation_only_no_recompiled_or_new_runtime_claim",
            "before_revision": "3662440", "before_source_sha256": sha(before),
            "policy_source_sha256": sha(Path(execution_policy.__file__).read_bytes()),
            "profile_sha256": profile_sha256(profile), "inputs": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.record:
        archive = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984"
        base = ROOT.parent / ".tmp/mx_model_goal_20261018"
        paths = [archive / f"{name}.mlir.gz" for name in (
            "tinyllama", "deepseek_r1_distill_qwen_1_5b", "gemma2_2b", "smolvla")]
        paths += [base / "onepass160/capture/model.mlir"]
        paths += [base / f"smolvla_{stage}_latest_optimized/input_read_view/model.generic.mlir"
                  for stage in ("prefix", "flow")]
        result = compare(paths)
        with (HERE / "report.json").open("x") as output:
            output.write(json.dumps(result, indent=2) + "\n")
    else:
        expected = json.loads((HERE / "report.json").read_text())
        result = compare([Path(row["path"]) for row in expected["inputs"]])
        assert result == expected
