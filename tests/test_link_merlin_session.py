"""A linked session must use each stage's exact generated weight layout."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.link_merlin_session import _shared_weight_build


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
        (generated / "weights.bin").write_bytes(value)
        builds[name] = build
        programs.append(SimpleNamespace(name=name))
        rows.append({"name": name, "generated": str(generated),
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
