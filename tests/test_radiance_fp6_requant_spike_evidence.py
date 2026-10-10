"""Recheck archived FP6 source-header parity and its typed LUT dependency."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.quant_reference import quantize_bf16_radiance_header_fp6
from mx_gemmini_support.source_payload import load_bundle, manifest_sha256
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_fp6_requant_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_radiance_fp6_source_requant_spike_evidence(tmp_path):
    index = json.loads((EVIDENCE / "qualification.json").read_text())
    assert index["schema"] == "mx_gemmini.radiance_fp6_requant_spike.v1"
    assert index["compiler_revision"].startswith("7224c9a")
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert {row["k"] for row in index["cases"]} == {128, 512}
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593" /
                           f"{index['profile']}.json")
    for row in index["cases"]:
        folder = EVIDENCE / row["case"]
        for filename, digest in row["files_sha256"].items():
            assert _sha(folder / filename) == digest
        receipt = json.loads((folder / "receipt.json").read_text())
        generation = json.loads((folder / "generation_receipt.json").read_text())
        manifest, resources = load_bundle(folder / "bundle")
        bound = (folder / "bound.mlir").read_text()
        m, n, k = manifest["shape_mnk"]
        assert (m, n, k) == (128, 128, row["k"])
        assert manifest_sha256(manifest) == row["payload_manifest_sha256"]
        assert receipt["compiler_revision"] == index["compiler_revision"]
        assert receipt["status"] == "radiance_header_matched_on_pinned_spike"
        assert receipt["spike_exit_code"] == 0
        assert receipt["elf_sha256"] == row["elf_sha256"]
        assert receipt["bound_mlir_sha256"] == _sha(folder / "bound.mlir")
        assert receipt["source_header_sha256"] == _sha(folder / "source_header.h")
        assert generation["generated_header_sha256"] == receipt["source_header_sha256"]
        assert generation["source_header_sha256"] == (
            "b57ef75fc634acd21750db45202bf12ced04569701f74e0607e0f06c974da93e")
        assert generation["bf16_sha256"] == _sha(folder / "bundle/golden_bf16.bin")
        assert generation["packed_output_sha256"] == _sha(
            folder / "bundle/source_fp6_packed.bin")
        assert generation["scale_sha256"] == _sha(folder / "bundle/output_scales.bin")
        assert receipt["source_driver_sha256"] == _sha(folder / "source_driver.cpp")
        assert receipt["compared_source_fp6_packed_bytes"] == m * n // 2
        assert receipt["compared_source_e8m0_scales"] == m * n // 32
        assert "0 Radiance FP6 packed-index mismatches, 0 E8M0 scale mismatches" in (
            folder / "spike.log").read_text()
        assert (resources["source_fp6_packed"], resources["golden_output_scales"]) == (
            quantize_bf16_radiance_header_fp6(
                resources["golden_bf16"], m, n, resources["output_lut"]))
        assert verify_ir(bound, profile)["source_resources"] == 7
        program = lower_bound_source(bound, profile, manifest, resources)
        assert program.output_format == "radiance_header_fp6"
        assert program.source_golden_preserving
        regenerated = write_standalone_sources(tmp_path / row["case"], program, resources)
        for name in ("mx_issue.c", "mx_driver.c", "physical_program.json"):
            assert regenerated["files_sha256"][name] == _sha(folder / name)

        with pytest.raises(ValueError, match="checked output LUT"):
            verify_ir(bound.replace('"mx_gemmini.host_requantize"(%out, %6)',
                                    '"mx_gemmini.host_requantize"(%out, %5)'), profile)
        bad = dict(resources)
        bad["output_lut"] = bytes(len(resources["output_lut"]))
        with pytest.raises(ValueError, match="golden differs"):
            lower_bound_source(bound, profile, manifest, bad)
