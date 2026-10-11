"""Audit the source-bound native DRAM-loop object and full Spike comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.native_dram import bind_native_dram, selected_native_dram
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_plain_matrix_object import (
    CASES, MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_fp8_native_dram_public_1ef6f85_266c593"
COMPILER = "1ef6f85b239543d8f65507add21a733e42f3cf1e"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_and_compiler_native_dram_loop_match() -> None:
    case = CASES["fp8_128x128x128_native_dram"]
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    audit = json.loads((EVIDENCE / "native_dram_equivalence.json").read_text())
    physical = json.loads((EVIDENCE / "object/physical_program.json").read_text())
    object_manifest = json.loads((EVIDENCE / "object/object_manifest.json").read_text())
    manifest, resources = load_bundle(EVIDENCE / "bundle")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    mlir = (EVIDENCE / "payload_bound.mlir").read_text()

    assert receipt["compiler_revision"] == COMPILER
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["mxq_revision"] == MXQ_REVISION
    assert receipt["profile_sha256"] == profile_sha256(profile)
    assert receipt["case"] == case.key
    assert receipt["source_driver_sha256"] == manifest["source_driver_sha256"] == (
        case.source_sha256)
    assert receipt["source_header_sha256"] == case.header_sha256
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["native_dram_loop"] is True
    assert receipt["outputs_checked"] == 16384 and receipt["mismatches"] == 0
    assert len(resources["golden_bf16"]) == 32768
    assert selected_native_dram(mlir, profile, manifest)
    assert audit["source_driver_sha256"] == case.source_sha256
    assert audit["physical_program_sha256"] == _sha(
        EVIDENCE / "object/physical_program.json")
    assert receipt["native_dram_equivalence_sha256"] == _sha(
        EVIDENCE / "native_dram_equivalence.json")
    assert audit["native_command_functs"] == [9, 10, 11, 12, 13, 8]
    assert audit["native_pointer_buffers"] == ["activation", "weight", "output_bf16"]
    assert audit["loop_bounds_ijk"] == [8, 8, 8]
    assert audit["scale_bytes_per_side"] == 512
    assert audit["explicit_operand_dma_commands"] == 0
    assert audit["explicit_output_dma_commands"] == 0
    assert physical["plan"]["execution_transport"] == "native_dram_loop"
    assert object_manifest["allocated_data_section_bytes"] == 0
    assert object_manifest["embedded_operand_bytes"] == 0
    assert receipt["object_sha256"] == _sha(EVIDENCE / "object/mx_issue.o")
    assert receipt["spike_log_sha256"] == _sha(EVIDENCE / "run/spike.log")
    assert "0 mismatches / 16384 BF16 values" in (EVIDENCE / "run/spike.log").read_text()
    assert receipt["source_baseline"]["source_spike_exit_code"] == 0
    assert receipt["source_baseline"]["source_golden_bf16_values_checked"] == 16384
    assert receipt["source_baseline"]["source_elf_sha256"] == _sha(
        EVIDENCE / "source_baseline/program.elf")
    assert receipt["source_baseline"]["source_spike_log_sha256"] == _sha(
        EVIDENCE / "source_baseline/spike.log")
    assert "fp8 WS native-loop matmul test PASSED (no mismatches)." in (
        EVIDENCE / "source_baseline/spike.log").read_text()


def test_native_schedule_rejects_other_source_and_tampered_selection() -> None:
    manifest, _ = load_bundle(EVIDENCE / "bundle")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    mlir = (EVIDENCE / "payload_bound.mlir").read_text()
    with pytest.raises(ValueError, match="not bound"):
        selected_native_dram(mlir, profile,
                             {**manifest, "source_driver_sha256": "0" * 64})
    with pytest.raises(ValueError, match="not bound"):
        selected_native_dram(mlir.replace('mx.native_dram_loop = "single"',
                                          'mx.native_dram_loop = "other"'),
                             profile, manifest)
    with pytest.raises(ValueError, match="one source-bound"):
        bind_native_dram(mlir, profile, manifest)


@pytest.mark.parametrize("chunks", [2, 4])
def test_native_column_chunk_source_and_compiler_match(chunks: int) -> None:
    directory = ROOT / "docs/evidence/nicolas_fp8_native_nc_public_57b26ce_266c593" / (
        f"nc{chunks}")
    case = CASES[f"fp8_128x128x128_native_dram_nc{chunks}"]
    receipt = json.loads((directory / "receipt.json").read_text())
    audit = json.loads((directory / "native_dram_equivalence.json").read_text())
    physical = json.loads((directory / "object/physical_program.json").read_text())
    object_manifest = json.loads((directory / "object/object_manifest.json").read_text())
    manifest, resources = load_bundle(directory / "bundle")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    mlir = (directory / "payload_bound.mlir").read_text()
    assert receipt["compiler_revision"] == "57b26cef780947aa125622b24b154ae055599b35"
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["mxq_revision"] == MXQ_REVISION
    assert receipt["profile_sha256"] == profile_sha256(profile)
    assert receipt["case"] == case.key
    assert receipt["source_driver_sha256"] == manifest["source_driver_sha256"] == (
        case.source_sha256)
    assert receipt["source_header_sha256"] == case.header_sha256
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["native_dram_loop"] is True
    assert receipt["outputs_checked"] == 16384 and receipt["mismatches"] == 0
    assert len(resources["golden_bf16"]) == 32768
    assert selected_native_dram(mlir, profile, manifest) == chunks
    assert audit["source_driver_sha256"] == case.source_sha256
    assert audit["native_chunk_count"] == chunks
    assert audit["native_command_functs"] == [9, 10, 11, 12, 13, 8] * chunks
    assert audit["scale_uploads"] == 2 * chunks
    assert audit["b_spad_ids"] == [1 + (c & 1) for c in range(chunks)]
    assert audit["explicit_operand_dma_commands"] == 0
    assert audit["explicit_output_dma_commands"] == 0
    assert audit["physical_program_sha256"] == _sha(
        directory / "object/physical_program.json")
    assert receipt["native_dram_equivalence_sha256"] == _sha(
        directory / "native_dram_equivalence.json")
    assert physical["plan"]["native_dram_n_chunks"] == chunks
    assert object_manifest["allocated_data_section_bytes"] == 0
    assert receipt["object_sha256"] == _sha(directory / "object/mx_issue.o")
    assert receipt["spike_log_sha256"] == _sha(directory / "run/spike.log")
    assert "0 mismatches / 16384 BF16 values" in (directory / "run/spike.log").read_text()
    assert receipt["source_baseline"]["source_spike_exit_code"] == 0
    assert receipt["source_baseline"]["source_golden_bf16_values_checked"] == 16384
    assert receipt["source_baseline"]["source_elf_sha256"] == _sha(
        directory / "source_baseline/program.elf")
    assert receipt["source_baseline"]["source_spike_log_sha256"] == _sha(
        directory / "source_baseline/spike.log")
    assert f"native multi-loop ({chunks} chunks) test PASSED" in (
        directory / "source_baseline/spike.log").read_text()


@pytest.mark.parametrize("chunks", [2, 4])
def test_loop_managed_scales_source_and_compiler_match(chunks: int) -> None:
    directory = ROOT / "docs/evidence/nicolas_fp8_native_ls_public_126f138_266c593" / (
        f"ls{chunks}")
    case = CASES[f"fp8_128x128x128_native_dram_ls{chunks}"]
    receipt = json.loads((directory / "receipt.json").read_text())
    audit = json.loads((directory / "native_dram_equivalence.json").read_text())
    physical = json.loads((directory / "object/physical_program.json").read_text())
    object_manifest = json.loads((directory / "object/object_manifest.json").read_text())
    manifest, resources = load_bundle(directory / "bundle")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    mlir = (directory / "payload_bound.mlir").read_text()
    assert receipt["compiler_revision"] == "126f13874535211bc154c97af1deee3027ec75b2"
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["mxq_revision"] == MXQ_REVISION
    assert receipt["profile_sha256"] == profile_sha256(profile)
    assert receipt["case"] == case.key
    assert receipt["source_driver_sha256"] == manifest["source_driver_sha256"] == (
        case.source_sha256)
    assert receipt["source_header_sha256"] == case.header_sha256
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["native_dram_loop"] is True
    assert receipt["native_dram_scale_mode"] == "loop"
    assert receipt["outputs_checked"] == 16384 and receipt["mismatches"] == 0
    assert len(resources["golden_bf16"]) == 32768
    assert selected_native_dram(mlir, profile, manifest) == f"ls{chunks}"
    assert audit["source_driver_sha256"] == case.source_sha256
    assert audit["native_chunk_count"] == chunks
    assert audit["native_command_functs"] == [9, 10, 11, 12, 13, 8] * chunks
    assert audit["scale_uploads"] == 0
    assert audit["loop_scale_configurations"] == 2 * chunks
    assert audit["b_spad_ids"] == [1 + (c & 1) for c in range(chunks)]
    assert audit["explicit_operand_dma_commands"] == 0
    assert audit["explicit_output_dma_commands"] == 0
    assert audit["physical_program_sha256"] == _sha(
        directory / "object/physical_program.json")
    assert receipt["native_dram_equivalence_sha256"] == _sha(
        directory / "native_dram_equivalence.json")
    assert physical["plan"]["native_dram_scale_mode"] == "loop_managed"
    assert "scratch_output_scales" not in [slot["name"] for slot in
                                           object_manifest["buffer_abi"]]
    assert object_manifest["allocated_data_section_bytes"] == 0
    assert receipt["object_sha256"] == _sha(directory / "object/mx_issue.o")
    assert receipt["spike_log_sha256"] == _sha(directory / "run/spike.log")
    assert "0 mismatches / 16384 BF16 values" in (directory / "run/spike.log").read_text()
    assert receipt["source_baseline"]["source_spike_exit_code"] == 0
    assert receipt["source_baseline"]["source_golden_bf16_values_checked"] == 16384
    assert receipt["source_baseline"]["source_elf_sha256"] == _sha(
        directory / "source_baseline/program.elf")
    assert receipt["source_baseline"]["source_spike_log_sha256"] == _sha(
        directory / "source_baseline/spike.log")
    assert f"native multi-loop ({chunks} chunks) test PASSED" in (
        directory / "source_baseline/spike.log").read_text()


def test_native_k_tiled_accumulation_matches_source() -> None:
    directory = ROOT / "docs/evidence/nicolas_fp8_native_kt_public_af0b4ad_266c593"
    case = CASES["fp8_128x128x128_native_dram_kt2"]
    receipt = json.loads((directory / "receipt.json").read_text())
    audit = json.loads((directory / "native_dram_equivalence.json").read_text())
    physical = json.loads((directory / "object/physical_program.json").read_text())
    obj = json.loads((directory / "object/object_manifest.json").read_text())
    manifest, resources = load_bundle(directory / "bundle")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    assert receipt["compiler_revision"] == "af0b4ad333ab462df9d96db0f0d227c3a16ded89"
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["mxq_revision"] == MXQ_REVISION
    assert receipt["case"] == case.key
    assert receipt["source_driver_sha256"] == manifest["source_driver_sha256"] == (
        case.source_sha256)
    assert receipt["source_header_sha256"] == case.header_sha256
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["mismatches"] == 0 and receipt["outputs_checked"] == 16384
    assert receipt["native_dram_scale_mode"] == "loop"
    assert len(resources["golden_bf16"]) == 32768
    assert selected_native_dram((directory / "payload_bound.mlir").read_text(),
                                profile, manifest) == "kt2"
    assert audit["source_driver_sha256"] == case.source_sha256
    assert audit["native_chunk_count"] == audit["k_tiles_per_chunk"] == 2
    assert audit["loop_scale_configurations"] == 8
    assert audit["native_command_functs"] == [9, 10, 11, 12, 13, 8] * 4
    assert audit["accumulating_loops"] == audit["store_loops"] == 2
    assert audit["explicit_operand_dma_commands"] == 0
    assert audit["explicit_output_dma_commands"] == 0
    assert audit["physical_program_sha256"] == _sha(
        directory / "object/physical_program.json")
    assert receipt["native_dram_equivalence_sha256"] == _sha(
        directory / "native_dram_equivalence.json")
    assert physical["plan"]["native_dram_k_tiles"] == 2
    assert obj["allocated_data_section_bytes"] == 0
    assert receipt["object_sha256"] == _sha(directory / "object/mx_issue.o")
    assert receipt["spike_log_sha256"] == _sha(directory / "run/spike.log")
    assert "0 mismatches / 16384 BF16 values" in (directory / "run/spike.log").read_text()
    assert receipt["source_baseline"]["source_spike_exit_code"] == 0
    assert receipt["source_baseline"]["source_golden_bf16_values_checked"] == 16384
    assert receipt["source_baseline"]["source_elf_sha256"] == _sha(
        directory / "source_baseline/program.elf")
    assert receipt["source_baseline"]["source_spike_log_sha256"] == _sha(
        directory / "source_baseline/spike.log")
    assert "native multi-loop (2 chunks) test PASSED" in (
        directory / "source_baseline/spike.log").read_text()


@pytest.mark.parametrize("variant,scale_mode", [
    ("nc_2d", "direct_2d"), ("nc_wait", "wait"),
])
def test_native_scale_control_matches_source(variant: str,
                                             scale_mode: str) -> None:
    directory = (ROOT / "docs/evidence/"
                 "nicolas_fp8_native_scale_control_public_c2e25ce_266c593" /
                 variant)
    case = CASES[f"fp8_128x128x128_native_dram_{variant}"]
    receipt = json.loads((directory / "receipt.json").read_text())
    audit = json.loads((directory / "native_dram_equivalence.json").read_text())
    physical = json.loads((directory / "object/physical_program.json").read_text())
    obj = json.loads((directory / "object/object_manifest.json").read_text())
    manifest, resources = load_bundle(directory / "bundle")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    assert receipt["compiler_revision"] == "c2e25ce2c8d5b9701b02837d307b922ae16cd749"
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["mxq_revision"] == MXQ_REVISION
    assert receipt["profile_sha256"] == profile_sha256(profile)
    assert receipt["case"] == case.key
    assert receipt["source_driver_sha256"] == manifest["source_driver_sha256"] == (
        case.source_sha256)
    assert receipt["source_header_sha256"] == case.header_sha256
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["native_dram_loop"] is True
    assert receipt["native_dram_scale_mode"] == scale_mode
    assert receipt["mismatches"] == 0 and receipt["outputs_checked"] == 16384
    assert len(resources["golden_bf16"]) == 32768
    selected = "nc2_2d" if variant == "nc_2d" else "nc2_wait"
    assert selected_native_dram((directory / "payload_bound.mlir").read_text(),
                                profile, manifest) == selected
    assert audit["source_driver_sha256"] == case.source_sha256
    assert audit["native_chunk_count"] == 2
    assert audit["native_command_functs"] == [9, 10, 11, 12, 13, 8] * 2
    assert audit["scale_uploads"] == 4
    assert audit["b_spad_ids"] == [1, 2]
    assert audit["explicit_operand_dma_commands"] == 0
    assert audit["explicit_output_dma_commands"] == 0
    assert audit.get("scale_config_wait_bit") == (True if variant == "nc_wait"
                                                   else None)
    assert audit["physical_program_sha256"] == _sha(
        directory / "object/physical_program.json")
    assert receipt["native_dram_equivalence_sha256"] == _sha(
        directory / "native_dram_equivalence.json")
    assert physical["plan"]["native_dram_n_chunks"] == 2
    assert physical["plan"].get("native_dram_scale_wait") == (
        True if variant == "nc_wait" else None)
    assert obj["allocated_data_section_bytes"] == 0
    assert receipt["object_sha256"] == _sha(directory / "object/mx_issue.o")
    assert receipt["spike_log_sha256"] == _sha(directory / "run/spike.log")
    assert "0 mismatches / 16384 BF16 values" in (
        directory / "run/spike.log").read_text()
    assert receipt["source_baseline"]["source_spike_exit_code"] == 0
    assert receipt["source_baseline"]["source_golden_bf16_values_checked"] == 16384
    assert receipt["source_baseline"]["source_elf_sha256"] == _sha(
        directory / "source_baseline/program.elf")
    assert receipt["source_baseline"]["source_spike_log_sha256"] == _sha(
        directory / "source_baseline/spike.log")
    assert "native multi-loop (2 chunks) test PASSED" in (
        directory / "source_baseline/spike.log").read_text()
    if variant == "nc_2d":
        assert _sha(directory / "object/mx_issue.o") == _sha(
            ROOT / "docs/evidence/nicolas_fp8_native_nc_public_57b26ce_266c593/"
            "nc2/object/mx_issue.o")


def test_native_relu_store_matches_source() -> None:
    directory = ROOT / "docs/evidence/nicolas_fp8_native_relu_public_c729276_266c593"
    case = CASES["fp8_128x128x128_native_dram_relu"]
    receipt = json.loads((directory / "receipt.json").read_text())
    audit = json.loads((directory / "native_dram_equivalence.json").read_text())
    physical = json.loads((directory / "object/physical_program.json").read_text())
    obj = json.loads((directory / "object/object_manifest.json").read_text())
    manifest, resources = load_bundle(directory / "bundle")
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    mlir = (directory / "payload_bound.mlir").read_text()
    assert receipt["compiler_revision"] == "c7292768d0184f06b82dcfc87a3419a8a4831a8f"
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["mxq_revision"] == MXQ_REVISION
    assert receipt["profile_sha256"] == profile_sha256(profile)
    assert receipt["case"] == case.key
    assert receipt["source_driver_sha256"] == manifest["source_driver_sha256"] == (
        case.source_sha256)
    assert receipt["source_header_sha256"] == case.header_sha256
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["native_dram_store_activation"] == "relu"
    assert receipt["mismatches"] == 0 and receipt["outputs_checked"] == 16384
    assert selected_native_dram(mlir, profile, manifest) == "ls2_relu"
    assert audit["source_driver_sha256"] == case.source_sha256
    assert audit["store_activation"] == "relu"
    assert audit["native_chunk_count"] == 2
    assert audit["native_command_functs"] == [9, 10, 11, 12, 13, 8] * 2
    assert audit["loop_scale_configurations"] == 4
    assert audit["scale_uploads"] == 0
    assert audit["explicit_operand_dma_commands"] == 0
    assert audit["explicit_output_dma_commands"] == 0
    assert audit["physical_program_sha256"] == _sha(
        directory / "object/physical_program.json")
    assert receipt["native_dram_equivalence_sha256"] == _sha(
        directory / "native_dram_equivalence.json")
    assert physical["plan"]["native_dram_store_activation"] == "relu"
    assert physical["source_golden_preserving"] is False
    assert physical["golden_derivation"] == "bf16_store_relu_sign_clear"
    store = [step["command"] for step in physical["steps"]
             if step["phase"] == "configure" and
             step["command"].get("funct") == 0 and
             step["command"]["rs1"].get("immediate") == 6]
    assert len(store) == 1 and store[0]["rs2"]["immediate"] == 256
    source = resources["golden_bf16"]
    assert len(source) == 32768
    expected = b"".join(b"\x00\x00" if source[i + 1] & 0x80 else source[i:i + 2]
                        for i in range(0, len(source), 2))
    assert sum(bool(source[i + 1] & 0x80) for i in range(0, len(source), 2)) == 8492
    program = lower_bound_source(mlir, profile, manifest, resources)
    assert program.derived_expected_bf16 == expected
    assert audit["derived_expected_bf16_sha256"] == hashlib.sha256(expected).hexdigest()
    assert physical["derived_expected_bf16_sha256"] == hashlib.sha256(expected).hexdigest()
    assert obj["allocated_data_section_bytes"] == 0
    assert receipt["object_sha256"] == _sha(directory / "object/mx_issue.o")
    assert receipt["spike_log_sha256"] == _sha(directory / "run/spike.log")
    assert "0 mismatches / 16384 BF16 values" in (
        directory / "run/spike.log").read_text()
    assert receipt["source_baseline"]["source_spike_exit_code"] == 0
    assert receipt["source_baseline"]["source_golden_bf16_values_checked"] == 16384
    assert receipt["source_baseline"]["source_elf_sha256"] == _sha(
        directory / "source_baseline/program.elf")
    assert receipt["source_baseline"]["source_spike_log_sha256"] == _sha(
        directory / "source_baseline/spike.log")
    assert "native multi-loop (2 chunks) test PASSED" in (
        directory / "source_baseline/spike.log").read_text()
