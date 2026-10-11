"""Audit the source-bound packed E4M3 LUT-index public-object replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_plain_matrix_object import (
    MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_e4m3_lut_requant_public_74bceeb_266c593"
COMPILER = "74bceebfb16b647425de0d322fcfa60118d0d817"
SOURCE = "matmul_tiled_fp8_e4m3_lut_64x64_requant"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_packed_lut_public_object_matches_all_source_codes_and_scales() -> None:
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    object_manifest = json.loads((ARCHIVE / "object/object_manifest.json").read_text())
    commands = json.loads((ARCHIVE / "object/physical_program.json").read_text())
    resources = json.loads((ARCHIVE / "physical/resource_manifest.json").read_text())
    inventory = json.loads((ROOT / "docs/evidence/nicolas_mx_source_inventory_266c593/index.json").read_text())
    source_hash = next(entry["source_sha256"] for entry in inventory["entries"]
                       if entry["name"] == SOURCE)
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593"
                           / "MxE4M3LutGemminiRocketConfig.json")

    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["spike_exit_code"] == 0
    assert receipt["compiler_revision"] == COMPILER
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["mxq_revision"] == MXQ_REVISION
    assert receipt["source_driver_sha256"] == source_hash
    assert receipt["profile_sha256"] == profile_sha256(profile)
    assert receipt["compared_packed_lut_bytes"] == 2048
    assert receipt["compared_e8m0_scales"] == 128
    assert "0 E4M3 packed-LUT-index mismatches, 0 E8M0 scale mismatches" in (
        ARCHIVE / "physical/spike.log").read_text()

    assert '"mx_gemmini.readout_quantized"' in (ARCHIVE / "asymmetric_bound.mlir").read_text()
    assert 'output_projection = "lut"' in (ARCHIVE / "asymmetric_bound.mlir").read_text()
    assert commands["output_format"] == "fp8_e4m3"
    assert commands["plan"]["quant_output_layout"] == "packed_even_odd_m_lut_indices"
    assert commands["source_golden_preserving"] is False
    assert resources["resources"]["golden_lut_indices"]["bytes"] == 2048
    assert resources["resources"]["golden_output_scales"]["bytes"] == 128

    slots = {slot["name"]: slot for slot in object_manifest["buffer_abi"]}
    assert object_manifest["allocated_data_section_bytes"] == 0
    assert object_manifest["embedded_operand_bytes"] == 0
    assert object_manifest["embedded_golden_bytes"] == 0
    assert slots["output_quantized"]["role"] == "write"
    assert slots["output_quantized"]["minimum_bytes"] == 2048
    assert slots["output_quantized"]["layout"] == "packed_even_odd_m_lut_indices"
    assert slots["scratch_output_scales"]["role"] == "write"
    assert slots["scratch_output_scales"]["minimum_bytes"] == 128
    assert receipt["public_object_sha256"] == _sha(ARCHIVE / "object/mx_issue.o")
    assert receipt["elf_sha256"] == _sha(ARCHIVE / "physical/asymmetric_program.elf")
    assert receipt["spike_log_sha256"] == _sha(ARCHIVE / "physical/spike.log")
    for name, digest in receipt["files_sha256"].items():
        assert digest == _sha(ARCHIVE / name)
    for name, digest in receipt["physical_inputs_sha256"].items():
        assert digest == _sha(ARCHIVE / "physical" / name)
