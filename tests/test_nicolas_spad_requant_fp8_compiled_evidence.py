"""Audit both archived Nicolas flat+tiled FP8 source replays."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_spad_requant_fp8_compiled_9d202b8_266c593"
COMPILER = "9d202b80b4876af1d3e078c7a545f69e0e215277"
SOURCE = "51af741fcfd14b031daa03b12ce6a3f03ee12f9d1f79bad5cabf3089087da508"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dual_fp8_requant_generated_object_matches_source_on_both_profiles() -> None:
    inventory = json.loads((ROOT / "docs/evidence/nicolas_mx_source_inventory_266c593/index.json").read_text())
    source = next(entry for entry in inventory["entries"] if entry["name"] == "spad_requant")
    assert source["source_sha256"] == SOURCE
    assert any("nicolas_spad_requant_fp8_compiled_9d202b8" in ref["path"]
               for ref in source["evidence_references"])
    for profile_name, folder_name in (
            ("MxE4M3VpuGemminiRocketConfig", "e4m3_vpu"),
            ("MxE4M3Fp4VpuGemminiRocketConfig", "e4m3_fp4_vpu")):
        folder = ARCHIVE / folder_name
        receipt = json.loads((folder / "receipt.json").read_text())
        physical = json.loads((folder / "physical_program.json").read_text())
        profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593"
                               / f"{profile_name}.json")
        assert receipt["status"] == "source_and_compiled_flat_tiled_fp8_matched_on_pinned_spike"
        assert receipt["compiler_revision"] == COMPILER
        assert receipt["rtl_revision"] == inventory["rtl_revision"]
        assert receipt["source_sha256"] == SOURCE
        assert receipt["profile_sha256"] == profile_sha256(profile)
        assert receipt["allocated_data_section_bytes"] == 0
        assert receipt["command_count"] == 37
        for side, run_dir in (("source_spike", "source"),
                              ("compiler_spike", "compiled")):
            outcome = receipt[side]
            assert outcome["exit_code"] == 0
            assert outcome["compared_fp8_codes"] == 4096
            assert outcome["compared_e8m0_scales"] == 128
            assert outcome["elf_sha256"] == _sha(folder / run_dir / "program.elf")
            assert outcome["spike_log_sha256"] == _sha(folder / run_dir / "spike.log")
        for name, archived in (("connected.mlir", "connected_mlir_sha256"),
                               ("physical_program.json", "physical_program_sha256"),
                               ("mx_issue.c", "issuer_c_sha256"),
                               ("mx_issue.o", "object_sha256"),
                               ("compiler_driver.c", "driver_sha256"),
                               ("compiler_driver.patch", "driver_patch_sha256")):
            assert receipt[archived] == _sha(folder / name)
        assert physical["profile_sha256"] == profile_sha256(profile)
        commands = physical["commands"]
        assert Counter(command["funct"] for command in commands
                       if command["kind"] == "command") == {
                           0: 2, 2: 16, 3: 16, 7: 1, 34: 2}
        assert [command["rs1"]["buffer"] for command in commands
                if command.get("funct") == 34] == ["scales_hw", "scales_hw2"]
        driver = (folder / "compiler_driver.c").read_text()
        assert "mxr_quant_block(&X[m][32 * b]" in driver
        assert "gemmini_spad_requant(" not in driver
        assert "gemmini_extended_mvin(" not in driver
        assert "gemmini_extended_mvout(" not in driver
        assert "gemmini_flush(" not in driver
        assert "spad_requant PASSED" in (folder / "compiled/spike.log").read_text()
