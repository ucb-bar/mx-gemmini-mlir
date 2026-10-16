"""MX quality arrays follow the complete session's verified dump order."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from merlin.llvmlower.session_bundle import MultiProgramSession, Program
from tools.assemble_session_output import assemble


def test_assemble_prefill_and_decode_logits(tmp_path):
    programs = tuple(Program(name, tmp_path, steps, (), (([1, 3], "f32"),))
                     for name, steps in (("prefill", 1), ("decode", 2)))
    session = MultiProgramSession(tmp_path, {}, programs, (), "decode", 0)
    dumps = []
    for program, step, values in ((0, 0, [1, 2, 3]),
                                  (1, 0, [4, 5, 6]),
                                  (1, 1, [7, 8, 9])):
        data = np.asarray(values, dtype="<f4").tobytes()
        name = f"p{program}_s{step}_o0.bin"
        (tmp_path / name).write_bytes(data)
        dumps.append({"program": program, "step": step, "output": 0,
                      "path": name, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    run = {"schema": "mx_gemmini.merlin_session_spike_run.v1",
           "complete": True, "output_dumps": dumps,
           "output_digests": [{key: row[key] for key in
                               ("program", "step", "output", "bytes", "sha256")}
                              for row in dumps]}
    output = assemble(session, run, tmp_path,
                      programs=("prefill", "decode"), output=0,
                      squeeze_leading_one=True, omit_single_step_axis=False)
    assert output.shape == (3, 3)
    np.testing.assert_array_equal(output, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    changed = dict(run, output_dumps=[dict(row) for row in dumps])
    changed["output_dumps"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="guest digest"):
        assemble(session, changed, tmp_path, programs=("prefill", "decode"),
                 output=0, squeeze_leading_one=True,
                 omit_single_step_axis=False)
    changed = dict(run, output_digests=run["output_digests"] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        assemble(session, changed, tmp_path, programs=("prefill", "decode"),
                 output=0, squeeze_leading_one=True,
                 omit_single_step_axis=False)
