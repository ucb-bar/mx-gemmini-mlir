"""Audit the compiled Nicolas FP6 128x128x512 source-golden replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_object import classify
from tools.qualify_nicolas_plain_matrix_object import CASES, MODEL2MLIR_REVISION, RTL_REVISION


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_plain_fp6_typed_object_95fc6d5_266c593"
CASE = CASES["fp6_128x128x512"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line0(lut: bytes) -> list[int]:
    packed = int.from_bytes(lut[:12], "little")
    return [(packed >> (6 * index)) & 0x3f for index in range(16)]


def test_nicolas_fp6_compiled_object_matches_complete_source_golden() -> None:
    receipt = json.loads((EVIDENCE / "receipt.json").read_text())
    obj = json.loads((EVIDENCE / "object/object_manifest.json").read_text())
    dispatch = json.loads((EVIDENCE / "object/compile_manifest.json").read_text())
    profile = load_profile(ROOT / f"profiles/gemmini-mx-cleanup-266c593/{CASE.profile_name}.json")
    bundle, resources = load_bundle(EVIDENCE / "bundle")
    family, report = classify((EVIDENCE / "payload_bound.mlir").read_text(), profile)

    assert receipt["schema"] == "mx_gemmini.nicolas_plain_fp6_typed_object_spike.v1"
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["mismatches"] == 0 and receipt["outputs_checked"] == 16384
    assert receipt["case"] == CASE.key
    assert receipt["rtl_revision"] == RTL_REVISION
    assert receipt["model2mlir_revision"] == MODEL2MLIR_REVISION
    assert receipt["source_driver_sha256"] == bundle["source_driver_sha256"] == (
        CASE.source_sha256)
    assert receipt["source_header_sha256"] == bundle["source_header_sha256"] == (
        CASE.header_sha256)
    assert receipt["compiler_revision"] == dispatch["compiler_revision"] == (
        "95fc6d563b900356c72bd4215e847dbcd8395aeb")
    assert receipt["compiler_source_closure_sha256"] == dispatch["compiler_source_closure_sha256"]
    assert receipt["profile_name"] == CASE.profile_name
    assert receipt["profile_sha256"] == obj["profile_sha256"] == profile_sha256(profile)
    assert bundle["origin"] == "nicolas_source_header_specialization"
    assert bundle["shape_mnk"] == [128, 128, 512]
    assert family == dispatch["lowering_family"] == "source_contract"
    assert report["contracts"] == 1 and report["source_resources"] == 7
    assert report["lut_uploads"] == 3
    assert len(resources["golden_bf16"]) == 128 * 128 * 2
    for side in ("activation", "weight", "output"):
        lut = resources[f"{side}_lut"]
        assert len(lut) == 64 * 3 * 4
        assert len({lut[index:index + 12] for index in range(0, len(lut), 12)}) > 1
    policy = yaml.safe_load((EVIDENCE / "source_line0_policy.yaml").read_text())
    assert policy["default_format"] == "mxfp6"
    assert policy["fp6_codebooks"]["default"]["activation"] == _line0(resources["activation_lut"])
    assert policy["fp6_codebooks"]["default"]["weight"] == _line0(resources["weight_lut"])

    for key, path in (
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
        assert receipt[key] == _sha(EVIDENCE / path)
    assert obj["object_sha256"] == dispatch["object_sha256"]
    assert obj["physical_program_sha256"] == _sha(EVIDENCE / "object/physical_program.json")
    assert obj["mode"] == "spike_serial"
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert [slot["name"] for slot in obj["buffer_abi"]] == list(CASE.buffer_abi)
    assert "compiled Nicolas FP6 128x128x512: 0 mismatches / 16384 BF16 values" in (
        EVIDENCE / "run/spike.log").read_text()
    driver = (EVIDENCE / "run/mx_driver.c").read_text()
    assert "mx_issue(A_in_hw, A_lut, A_scales_row" in driver
    assert "gemmini_loop_ws_spad" not in driver
