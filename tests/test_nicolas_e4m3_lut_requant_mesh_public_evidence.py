"""Audit all four DIM8 and DIM32 packed LUT-index output replays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_plain_matrix_object import (
    MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_e4m3_lut_requant_mesh_public_4553528_266c593"
COMPILER = "45535285f67aa4f5c60e5c05dc810132df74efdf"
CASES = ((8, 64), (32, 64), (8, 128), (32, 128))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_four_mesh_requant_source_programs_match() -> None:
    inventory = json.loads((ROOT / "docs/evidence/nicolas_mx_source_inventory_266c593/index.json").read_text())
    source_hashes = {entry["name"]: entry["source_sha256"] for entry in inventory["entries"]}
    total_codes = total_scales = 0
    for dim, side in CASES:
        folder = ARCHIVE / f"e4m3_dim{dim}_{side}x{side}"
        receipt = json.loads((folder / "receipt.json").read_text())
        object_manifest = json.loads((folder / "object/object_manifest.json").read_text())
        physical = json.loads((folder / "object/physical_program.json").read_text())
        profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593"
                               / f"MxDim{dim}AllGemminiRocketConfig.json")
        source = f"matmul_tiled_fp8_e4m3_lut_{side}x{side}_requant_dim{dim}"
        packed_bytes, scales = side * side // 2, side * side // 32

        assert receipt["status"] == "source_golden_matched_on_pinned_spike"
        assert receipt["spike_exit_code"] == 0
        assert receipt["compiler_revision"] == COMPILER
        assert receipt["rtl_revision"] == RTL_REVISION
        assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
        assert receipt["mxq_revision"] == MXQ_REVISION
        assert receipt["source_driver_sha256"] == source_hashes[source]
        assert receipt["profile_sha256"] == profile_sha256(profile)
        assert receipt["compared_packed_lut_bytes"] == packed_bytes
        assert receipt["compared_e8m0_scales"] == scales
        assert "0 E4M3 packed-LUT-index mismatches, 0 E8M0 scale mismatches" in (
            folder / "physical/spike.log").read_text()
        assert 'output_projection = "lut"' in (folder / "asymmetric_bound.mlir").read_text()
        assert physical["output_format"] == "fp8_e4m3"
        assert physical["plan"]["mesh_dim"] == dim
        assert physical["plan"]["quant_output_layout"] == "packed_even_odd_m_lut_indices"
        slots = {slot["name"]: slot for slot in object_manifest["buffer_abi"]}
        assert object_manifest["allocated_data_section_bytes"] == 0
        assert slots["output_quantized"]["minimum_bytes"] == packed_bytes
        assert slots["output_quantized"]["layout"] == "packed_even_odd_m_lut_indices"
        assert slots["scratch_output_scales"]["minimum_bytes"] == scales
        assert slots["scratch_output_scales"]["role"] == "write"
        assert receipt["public_object_sha256"] == _sha(folder / "object/mx_issue.o")
        assert receipt["elf_sha256"] == _sha(folder / "physical/asymmetric_program.elf")
        assert receipt["spike_log_sha256"] == _sha(folder / "physical/spike.log")
        for name, digest in receipt["files_sha256"].items():
            assert digest == _sha(folder / name)
        for name, digest in receipt["physical_inputs_sha256"].items():
            assert digest == _sha(folder / "physical" / name)
        total_codes += packed_bytes
        total_scales += scales
    assert (total_codes, total_scales) == (20480, 1280)
