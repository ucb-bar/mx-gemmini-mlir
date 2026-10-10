"""Keep the direct Nicolas FP8 source-golden object replay auditable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_object import classify
from tools.qualify_nicolas_plain_fp8_object import (HEADER_SHA256, MODEL2MLIR_REVISION,
                                                    RTL_REVISION, SOURCE_SHA256)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_plain_fp8_typed_object_7d7a660_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nicolas_fp8_compiled_object_matches_all_source_bf16_outputs() -> None:
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    object_manifest = json.loads((EVIDENCE / "object/object_manifest.json").read_text())
    dispatch = json.loads((EVIDENCE / "object/compile_manifest.json").read_text())
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json")
    bundle, resources = load_bundle(EVIDENCE / "bundle")
    bound = (EVIDENCE / "payload_bound.mlir").read_text()
    family, report = classify(bound, profile)

    assert receipt["schema"] == "mx_gemmini.nicolas_plain_fp8_typed_object_spike.v1"
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["mismatches"] == 0 and receipt["outputs_checked"] == 128 * 128
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["source_driver_sha256"] == SOURCE_SHA256 == bundle["source_driver_sha256"]
    assert receipt["source_header_sha256"] == HEADER_SHA256 == bundle["source_header_sha256"]
    assert receipt["compiler_revision"] == dispatch["compiler_revision"] == (
        "7d7a660b8bfe6911904ae9585f47974bc83d814f")
    assert receipt["compiler_source_closure_sha256"] == dispatch["compiler_source_closure_sha256"]
    assert receipt["profile_sha256"] == object_manifest["profile_sha256"] == (
        profile_sha256(profile))
    assert bundle["origin"] == "nicolas_source_header_specialization"
    assert bundle["shape_mnk"] == [128, 128, 128]
    assert len(resources["golden_bf16"]) == 128 * 128 * 2
    assert family == dispatch["lowering_family"] == "source_contract"
    assert report["contracts"] == 1 and report["source_resources"] == 4

    for key, name in (
            ("source_mlir_sha256", "model2mlir.mlir"),
            ("handoff_mlir_sha256", "handoff.mlir"),
            ("payload_manifest_sha256", "bundle/manifest.json"),
            ("bound_mlir_sha256", "payload_bound.mlir"),
            ("object_dispatch_manifest_sha256", "object/compile_manifest.json"),
            ("object_manifest_sha256", "object/object_manifest.json"),
            ("object_sha256", "object/mx_issue.o"),
            ("elf_sha256", "run/mx_program.elf"),
            ("spike_log_sha256", "run/spike.log"),
            ("driver_sha256", "run/mx_driver.c")):
        assert receipt[key] == _sha(EVIDENCE / name)
    assert object_manifest["object_sha256"] == dispatch["object_sha256"]
    assert object_manifest["physical_program_sha256"] == _sha(
        EVIDENCE / "object/physical_program.json")
    assert object_manifest["allocated_data_section_bytes"] == 0
    assert object_manifest["embedded_operand_bytes"] == 0
    assert object_manifest["embedded_golden_bytes"] == 0
    assert [slot["name"] for slot in object_manifest["buffer_abi"]] == [
        "activation", "activation_scales", "output_bf16",
        "scratch_output_scales", "weight", "weight_scales"]
    assert len(json.loads((EVIDENCE / "object/physical_program.json").read_text())["steps"]) == 275
    assert "compiled Nicolas FP8 128 cubed: 0 mismatches / 16384 BF16 values" in (
        EVIDENCE / "run/spike.log").read_text()
    driver = (EVIDENCE / "run/mx_driver.c").read_text()
    assert "mx_issue(A_in, A_scales_row, C_hw" in driver
    assert "gemmini_loop_ws_spad" not in driver
