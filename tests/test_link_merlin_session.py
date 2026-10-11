"""A linked session must use each stage's exact generated weight layout."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.link_merlin_session import _session_main, _shared_weight_build


def _fixture(tmp_path: Path, values: dict[str, bytes]):
    builds = {}
    rows = []
    programs = []
    for name, value in values.items():
        build = tmp_path / f"build_{name}"
        generated = tmp_path / f"generated_{name}"
        (build / "cgen").mkdir(parents=True)
        generated.mkdir()
        (build / "cgen/weights.bin").write_bytes(value)
        builds[name] = build
        programs.append(SimpleNamespace(name=name))
        rows.append({"name": name, "generated": str(generated),
                     "weights_bytes": len(value),
                     "weights_sha256": hashlib.sha256(value).hexdigest()})
    return tuple(programs), builds, {"programs": rows}


def test_identical_weights_and_weightless_stage_share_one_blob(tmp_path: Path) -> None:
    programs, builds, generated = _fixture(
        tmp_path, {"prefix": b"same bytes", "flow": b"same bytes", "action": b""})
    assert _shared_weight_build(programs, builds, generated) == "prefix"


def test_different_nonempty_stage_weights_refuse_link(tmp_path: Path) -> None:
    programs, builds, generated = _fixture(tmp_path, {"prefill": b"first", "decode": b"other"})
    with pytest.raises(ValueError, match="different nonempty weight blobs"):
        _shared_weight_build(programs, builds, generated)


def test_stage_build_must_match_merlin_generated_abi(tmp_path: Path) -> None:
    programs, builds, generated = _fixture(tmp_path, {"prefill": b"captured"})
    (builds["prefill"] / "cgen/weights.bin").write_bytes(b"rewritten")
    with pytest.raises(ValueError, match="weight layout differs"):
        _shared_weight_build(programs, builds, generated)


def test_bare_metal_session_reclaims_temporary_allocations_per_step() -> None:
    source = _session_main([2, 1], weights_base=0x200000000)
    assert source.index("merlin_arena_reset();") < source.index("merlin_session_run_step(")
    assert source.count("merlin_arena_reset();") == 1
