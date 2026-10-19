"""The MX session linker keeps stage scheduling target agnostic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.link_merlin_session import (
    _selected_output_dumps,
    _session_main,
    _session_math_contract,
)


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


def test_native_session_math_needs_no_selection():
    assert _session_math_contract({"first": {}, "second": {}}, None) is None


def test_guarded_stages_cannot_silently_lose_their_wrapper():
    from merlin.runtime.host_math import GuardedF32Call

    guard = GuardedF32Call("expf", "body", 0x42A00000, "a" * 64)
    receipts = {"first": {"host_math_guard": guard.to_dict()}, "identity": {}}
    with pytest.raises(ValueError, match="explicit session selection"):
        _session_math_contract(receipts, None)
    assert _session_math_contract(receipts, guard) == guard
    other = GuardedF32Call("expf", "body", 0x42000000, "a" * 64)
    with pytest.raises(ValueError, match="explicit session selection"):
        _session_math_contract(receipts, other)
    with pytest.raises(ValueError, match="typed guarded contract"):
        _session_math_contract(receipts, guard.to_dict())


def test_incompatible_stage_math_contracts_refuse_a_global_wrap():
    from merlin.runtime.host_math import GuardedF32Call

    guard = GuardedF32Call("expf", "body", 0x42A00000, "a" * 64)
    other = GuardedF32Call("expf", "body", 0x42A00000, "b" * 64)
    with pytest.raises(ValueError, match="stage second math"):
        _session_math_contract({"first": {"host_math_guard": guard.to_dict()},
                                "second": {"host_math_guard": other.to_dict()}}, guard)
