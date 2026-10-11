"""Check the source-specific transfer equivalence, including rejection."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from mx_gemmini_support.target_profile import load_profile
from tools.qualify_nicolas_plain_matrix_object import (
    CASES, _audit_alternate_fp8_mvin,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/evidence/nicolas_direct_matrix_suite_9df3384_266c593/fp8_32x32x32/object"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"


def test_pinned_alternate_source_matches_existing_compiler_transfers(tmp_path: Path) -> None:
    case = CASES["fp8_32x32x32_alternate_mvin"]
    profile = load_profile(PROFILE)
    shutil.copy(BASELINE / "physical_program.json", tmp_path / "physical_program.json")
    shutil.copy(BASELINE / "mx_issue.o", tmp_path / "mx_issue.o")
    report = _audit_alternate_fp8_mvin(tmp_path, profile, case.source_sha256)
    assert report["source_transfer_pairs"] == report["compiler_transfer_pairs"]
    assert report["compiler_transfer_pairs"]["move_weight"] == [
        (0, 16320), (16, 16336), (512, 16352), (528, 16368)]
    assert report["object_sha256"] == (
        "69926e33aa470b821571ec5fb402f1270615d6972f16d1d39f53ee5da8bb7a1d")
    physical = json.loads((tmp_path / "physical_program.json").read_text())
    weight = next(step["command"] for step in physical["steps"]
                  if step["phase"] == "move_weight" and
                  step["command"].get("funct") == 2)
    weight["rs1"]["byte_offset"] = 16
    (tmp_path / "physical_program.json").write_text(json.dumps(physical))
    with pytest.raises(ValueError, match="move_weight differs"):
        _audit_alternate_fp8_mvin(tmp_path, profile, case.source_sha256)
