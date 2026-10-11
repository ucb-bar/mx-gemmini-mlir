"""Audit rectangular packed-output replays and unequal LUT bank sizes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_plain_matrix_object import (
    MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_e4m3_lut_requant_rect_public_75524c6_266c593"
COMPILER = "75524c6d0666be404ceba247814290e866076c8a"
CASES = (("64x128x128", (32, 64, 32)), ("128x64x128", (64, 32, 64)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rectangular_source_lut_banks_and_all_outputs_match() -> None:
    inventory = json.loads((ROOT / "docs/evidence/nicolas_mx_source_inventory_266c593/index.json").read_text())
    source_hashes = {entry["name"]: entry["source_sha256"] for entry in inventory["entries"]}
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593"
                           / "MxDim32AllGemminiRocketConfig.json")
    for shape, lines in CASES:
        folder = ARCHIVE / f"e4m3_{shape}_dim32"
        receipt = json.loads((folder / "receipt.json").read_text())
        object_manifest = json.loads((folder / "object/object_manifest.json").read_text())
        physical = json.loads((folder / "object/physical_program.json").read_text())
        resources = json.loads((folder / "physical/resource_manifest.json").read_text())
        source = f"matmul_tiled_fp8_e4m3_lut_{shape}_requant_dim32"
        assert receipt["status"] == "source_golden_matched_on_pinned_spike"
        assert receipt["spike_exit_code"] == 0
        assert receipt["compiler_revision"] == COMPILER
        assert receipt["rtl_revision"] == RTL_REVISION
        assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
        assert receipt["mxq_revision"] == MXQ_REVISION
        assert receipt["profile_sha256"] == profile_sha256(profile)
        assert receipt["source_driver_sha256"] == source_hashes[source]
        assert receipt["compared_packed_lut_bytes"] == 4096
        assert receipt["compared_e8m0_scales"] == 256
        assert "0 E4M3 packed-LUT-index mismatches, 0 E8M0 scale mismatches" in (
            folder / "physical/spike.log").read_text()
        for resource, count in zip(("activation_lut", "weight_lut", "output_lut"), lines):
            assert resources["resources"][resource]["shape"] == [count, 4]
        issued_lines = {step["command"]["rs1"]["buffer"]:
                        step["command"]["rs2"]["immediate"] & 0xffffffff
                        for step in physical["steps"] if step["phase"] == "upload_lut"}
        assert issued_lines == dict(zip(
            ("activation_lut", "weight_lut", "output_lut"), lines))
        assert physical["plan"]["quant_output_layout"] == "packed_even_odd_m_lut_indices"
        slots = {slot["name"]: slot for slot in object_manifest["buffer_abi"]}
        assert object_manifest["allocated_data_section_bytes"] == 0
        assert slots["output_quantized"]["minimum_bytes"] == 4096
        assert slots["scratch_output_scales"]["minimum_bytes"] == 256
        assert receipt["public_object_sha256"] == _sha(folder / "object/mx_issue.o")
        assert receipt["elf_sha256"] == _sha(folder / "physical/asymmetric_program.elf")
        assert receipt["spike_log_sha256"] == _sha(folder / "physical/spike.log")
        for name, digest in receipt["files_sha256"].items():
            assert digest == _sha(folder / name)
        for name, digest in receipt["physical_inputs_sha256"].items():
            assert digest == _sha(folder / "physical" / name)
