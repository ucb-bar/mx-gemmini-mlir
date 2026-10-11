"""Audit Nicolas's source-bound FP6 128³ object with repeated LUT codes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_plain_matrix_object import (
    CASES, MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (ROOT / "docs/evidence/"
            "nicolas_plain_fp6_128_aliased_lut_public_6c6b27a_266c593")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lut_lines(data: bytes) -> tuple[tuple[int, ...], ...]:
    assert len(data) == 64 * 12
    return tuple(tuple((int.from_bytes(data[12 * row:12 * (row + 1)], "little")
                        >> (6 * index)) & 63 for index in range(16))
                 for row in range(64))


def test_source_and_compiled_fp6_128_match_with_original_aliased_luts() -> None:
    case = CASES["fp6_128x128x128"]
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    manifest, resources = load_bundle(EVIDENCE / "bundle")
    physical = json.loads((EVIDENCE / "object/physical_program.json").read_text())
    obj = json.loads((EVIDENCE / "object/object_manifest.json").read_text())
    policy = yaml.safe_load((EVIDENCE / "source_line0_policy.yaml").read_text())
    profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/"
               "MxE3M2OnlyGemminiRocketConfig.json")

    assert receipt["compiler_revision"] == "6c6b27a9cfe7faf88326d7ab26eeb00c1e37d841"
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["mxq_revision"] == MXQ_REVISION
    assert receipt["profile_sha256"] == profile_sha256(profile)
    assert receipt["case"] == case.key
    assert receipt["source_driver_sha256"] == manifest["source_driver_sha256"] == (
        case.source_sha256)
    assert receipt["source_header_sha256"] == manifest["source_header_sha256"] == (
        case.header_sha256)
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["outputs_checked"] == 16384 and receipt["mismatches"] == 0
    assert receipt["fp6_capture_codebook_role"] == (
        "unique_structural_witness_not_source_lut")
    assert receipt["fp6_capture_policy_sha256"] == _sha(
        EVIDENCE / "source_line0_policy.yaml")
    assert len(resources["activation"]) == len(resources["weight"]) == 8192
    assert len(resources["golden_bf16"]) == 32768
    assert hashlib.sha256(resources["activation"]).hexdigest() == (
        "95f65cba82f9d965d2d5c759cb9abb49924806f4cd67624667ed8ffdb1aef590")
    assert hashlib.sha256(resources["weight"]).hexdigest() == (
        "53a0e2a7f91ed48378b353a5c345e5b8bca54b57b2ae807f4c0152eec4fd19fb")
    for name in ("activation_lut", "weight_lut", "output_lut"):
        assert len(resources[name]) == 768
    for name in ("activation_lut", "weight_lut"):
        lines = _lut_lines(resources[name])
        assert sum(len(set(line)) < 16 for line in lines) == 64
        capture = policy["fp6_codebooks"]["default"][name.removesuffix("_lut")]
        assert len(capture) == len(set(capture)) == 16
        assert tuple(capture) != lines[0]
    assert physical["shape_mnk"] == [128, 128, 128]
    assert physical["plan"]["format"] == "fp6_e3m2"
    assert physical["plan"]["lut_once"] is True
    assert physical.get("source_golden_preserving", True)
    assert obj["allocated_data_section_bytes"] == 0
    assert receipt["object_sha256"] == _sha(EVIDENCE / "object/mx_issue.o")
    assert receipt["spike_log_sha256"] == _sha(EVIDENCE / "run/spike.log")
    assert "0 mismatches / 16384 BF16 values" in (
        EVIDENCE / "run/spike.log").read_text()
    assert receipt["source_baseline"]["source_spike_exit_code"] == 0
    assert receipt["source_baseline"]["source_golden_bf16_values_checked"] == 16384
    assert receipt["source_baseline"]["source_elf_sha256"] == _sha(
        EVIDENCE / "source_baseline/program.elf")
    assert receipt["source_baseline"]["source_spike_log_sha256"] == _sha(
        EVIDENCE / "source_baseline/spike.log")
    assert "fp6 WS matmul test PASSED (no mismatches)." in (
        EVIDENCE / "source_baseline/spike.log").read_text()
