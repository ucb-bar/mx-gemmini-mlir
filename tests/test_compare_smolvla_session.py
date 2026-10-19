"""A complete SmolVLA action comparison binds each step to its run receipt."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tools.compare_smolvla_session import compare_session


@pytest.mark.parametrize("changed_final_zero_sign", [False, True])
def test_flow_and_action_comparison_checks_every_dump(tmp_path, changed_final_zero_sign) -> None:
    session = tmp_path / "session"
    stage = session / "stages/flow_denoise"
    stage.mkdir(parents=True)
    reference = np.arange(10 * 50 * 32, dtype=np.float32).reshape(10, 1, 50, 32) / 1000
    reference[9, 0, 0, 0] = 0.0
    np.savez(stage / "session_goldens.npz", actions=reference)
    link_path = tmp_path / "link.json"
    link_path.write_text(json.dumps({
        "session_root": str(session),
        "programs": [{"name": name, "steps": steps} for name, steps in (
            ("prefix_encode", 1), ("flow_denoise", 10), ("action_decode", 1))],
    }))
    dump_dir = tmp_path / "output_dumps"
    dump_dir.mkdir()
    dumps = []
    for program, step in [(1, index) for index in range(10)] + [(2, 0)]:
        value = reference[step if program == 1 else 9].copy()
        if program == 1 and step == 3:
            value += 0.25
        if program == 2 and changed_final_zero_sign:
            value[0, 0, 0] = -0.0
        relative = f"output_dumps/p{program}_s{step}_o0.bin"
        raw = value.tobytes()
        (tmp_path / relative).write_bytes(raw)
        dumps.append({"program": program, "step": step, "output": 0,
                      "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "path": relative})
    timestep = np.float32(0.6).tobytes()
    relative = "output_dumps/p1_s3_o2.bin"
    (tmp_path / relative).write_bytes(timestep)
    dumps.append({"program": 1, "step": 3, "output": 2,
                  "bytes": len(timestep),
                  "sha256": hashlib.sha256(timestep).hexdigest(),
                  "path": relative})
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps({
        "status": "complete_numeric_qualification_pending",
        "link_receipt_sha256": hashlib.sha256(link_path.read_bytes()).hexdigest(),
        "output_dumps": dumps,
    }))
    if changed_final_zero_sign:
        with pytest.raises(ValueError, match="final flow state"):
            compare_session(run_path, link_path, session)
        return
    report = compare_session(run_path, link_path, session)
    assert report["final_action_metrics"]["max_abs"] == 0
    assert report["flow_step_metrics"][3]["max_abs"] == 0.25
    assert report["numerical_qualification"] == "not_established"
    (dump_dir / "p1_s3_o0.bin").write_bytes(bytes(6400))
    with pytest.raises(ValueError, match="differs from its run receipt"):
        compare_session(run_path, link_path, session)
