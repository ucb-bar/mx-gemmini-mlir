"""Audit ten E2M3/E5M2 source-bound packed LUT-index object replays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_plain_matrix_object import (
    MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_e2m3_e5m2_lut_requant_mesh_public_b94c3af_266c593"
COMPILER = "b94c3afb1635094e854794d434b6f4ac35b8d953"
SHAPES = ((8, "64x64", (64, 64, 64)),
          (32, "64x64", (64, 64, 64)),
          (32, "128x128", (128, 128, 128)),
          (32, "64x128x128", (64, 128, 128)),
          (32, "128x64x128", (128, 64, 128)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_ten_source_programs_match_packed_codes_and_scales() -> None:
    inventory = json.loads((ROOT / "docs/evidence/nicolas_mx_source_inventory_266c593/index.json").read_text())
    source_hashes = {entry["name"]: entry["source_sha256"] for entry in inventory["entries"]}
    total_codes = total_scales = 0
    for fmt, prefix, output, words in (
            ("e2m3", "fp6_e2m3_lut", "fp6_e2m3", 3),
            ("e5m2", "fp8_e5m2", "fp8_e5m2", 4)):
        for dim, shape_name, (m, n, k) in SHAPES:
            folder = ARCHIVE / f"{fmt}_dim{dim}_{shape_name}"
            receipt = json.loads((folder / "receipt.json").read_text())
            object_manifest = json.loads((folder / "object/object_manifest.json").read_text())
            physical = json.loads((folder / "object/physical_program.json").read_text())
            resources = json.loads((folder / "physical/resource_manifest.json").read_text())
            profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593"
                                   / f"MxDim{dim}AllGemminiRocketConfig.json")
            source = f"matmul_tiled_{prefix}_{shape_name}_requant_dim{dim}"
            packed_bytes, scales = m * n // 2, m * n // 32

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
            assert f"0 {fmt.upper()} packed-LUT-index mismatches, 0 E8M0 scale mismatches" in (
                folder / "physical/spike.log").read_text()
            mlir = (folder / "asymmetric_bound.mlir").read_text()
            assert f'output_format = "{output}"' in mlir
            assert 'output_projection = "lut"' in mlir
            assert physical["output_format"] == output
            assert physical["plan"]["shape_mnk"] == [m, n, k]
            assert physical["plan"]["quant_output_layout"] == "packed_even_odd_m_lut_indices"
            for resource, lines in (("activation_lut", m // 2),
                                    ("weight_lut", n // 2),
                                    ("output_lut", m // 2)):
                assert resources["resources"][resource]["shape"] == [lines, words]
            slots = {slot["name"]: slot for slot in object_manifest["buffer_abi"]}
            assert object_manifest["allocated_data_section_bytes"] == 0
            assert slots["output_quantized"]["minimum_bytes"] == packed_bytes
            assert slots["scratch_output_scales"]["minimum_bytes"] == scales
            assert receipt["public_object_sha256"] == _sha(folder / "object/mx_issue.o")
            assert receipt["elf_sha256"] == _sha(folder / "physical/asymmetric_program.elf")
            assert receipt["spike_log_sha256"] == _sha(folder / "physical/spike.log")
            for name, digest in receipt["files_sha256"].items():
                assert digest == _sha(folder / name)
            for name, digest in receipt["physical_inputs_sha256"].items():
                assert digest == _sha(folder / "physical" / name)
            total_codes += packed_bytes
            total_scales += scales
    assert (total_codes, total_scales) == (40960, 2560)
