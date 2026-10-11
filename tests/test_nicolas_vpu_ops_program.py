"""The full Nicolas VPU source schedule must remain one checked object graph."""

from __future__ import annotations

import hashlib
import json

from mx_gemmini_support.command_ir import Fence
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vpu_sequence_program import lower_vpu_sequence
from tools.compile_object import classify
from tools.qualify_nicolas_vpu_ops_program import (
    BASELINE, FP4_PROFILE, PROFILE, _abi, _driver, _mlir, _schedule)


def test_full_source_schedule_lowers_with_snapshots_and_war_reload() -> None:
    baseline = json.loads(BASELINE.read_text())
    operations, captures, references, reloads = _schedule()
    labels = [name for label in baseline["checks"] for name in
              (["dual X", "dual Y"] if label == "dual X,Y" else [label])]
    assert [item["label"] for item in captures] == labels
    assert len(operations) == len(captures) == 30
    assert reloads == [{"input": "a2", "row": 0, "after_operation": 26}]
    schedule = {"operations": operations, "captures": captures, "reloads": reloads}
    digest = hashlib.sha256(json.dumps(schedule, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    for path in (PROFILE, FP4_PROFILE):
        profile = load_profile(path)
        graph = _mlir(operations, baseline["source_sha256"],
                      baseline["reference_sha256"], digest,
                      profile_sha256(profile))
        assert classify(graph, profile)[0] == "vpu_sequence"
        plan = lower_vpu_sequence(graph, profile, _abi(captures, reloads))
        assert len(plan.operations) == 30
        assert len(plan.captures) == 30
        assert sum(isinstance(command, Fence) for command in plan.commands) == 26
        assert sum(item.rows * 8 for item in plan.captures) == 13056
    driver = _driver(captures, references)
    assert driver.count("mx_issue(A, A2, B, P, X,") == 1
    assert "gemmini_" not in driver
    assert "vpu_ref_exec(VPU_EXPSUM" in driver
    assert "vpu_ref_exec(VPU_RMAX" in driver
