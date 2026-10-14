"""The MX session linker keeps stage scheduling target agnostic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.link_merlin_session import _selected_output_dumps, _session_main


def test_bare_metal_session_reclaims_temporary_allocations_per_step() -> None:
    source = _session_main([2, 1], weights_base=0x200000000)
    assert source.index("merlin_arena_reset();") < source.index("merlin_session_run_step(")
    assert source.count("merlin_arena_reset();") == 1
    assert source.index("read_cycle();") < source.index("merlin_session_run_step(")
    assert source.index("SESSION_PERF ") < source.index("report(program, step, output")
    assert "SESSION_TOTAL " in source


def test_selected_output_is_dumped_after_digest_and_timing() -> None:
    source = _session_main([2, 1], weights_base=0x200000000,
                           dump_outputs=((1, 0),))
    assert source.index("SESSION_PERF ") < source.index("report(program, step, output")
    assert source.index("report(program, step, output") < source.index(
        "if ((program == 1 && output == 0)) dump_output")
    assert "SESSION_HEX " in source


def test_large_logits_dump_requires_explicit_bounded_limit() -> None:
    session = SimpleNamespace(
        program_names=("prefill", "decode"),
        programs=(SimpleNamespace(steps=1, outputs=(([1, 32000], "f32"),)),
                  SimpleNamespace(steps=32, outputs=(([1, 32000], "f32"),))),
    )
    requests = ("prefill:0", "decode:0")
    with pytest.raises(ValueError, match="console byte limit"):
        _selected_output_dumps(session, requests, 16_384)
    selected = _selected_output_dumps(session, requests, 128_000)
    assert [row["bytes"] for row in selected] == [128_000, 128_000]
    with pytest.raises(ValueError, match="8 MiB"):
        _selected_output_dumps(SimpleNamespace(
            program_names=("prefill", "decode"),
            programs=(session.programs[0],
                      SimpleNamespace(steps=65, outputs=session.programs[1].outputs)),
        ), requests, 128_000)
