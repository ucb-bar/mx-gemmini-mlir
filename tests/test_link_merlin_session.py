"""The MX session linker keeps stage scheduling target agnostic."""

from __future__ import annotations

from tools.link_merlin_session import _session_main


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
