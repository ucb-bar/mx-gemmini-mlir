"""Recheck archived source-compatible FP8 output against bound bytes and Spike."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.quant_reference import quantize_bf16_radiance_header_fp8
from mx_gemmini_support.source_payload import load_bundle, manifest_sha256
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_header_requant_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_radiance_header_spike_matrix_is_source_bound_and_reproducible(tmp_path):
    index = json.loads((EVIDENCE / "qualification.json").read_text())
    assert index["schema"] == "mx_gemmini.radiance_header_requant_spike.v1"
    assert index["compiler_revision"].startswith("6a09dee")
    assert len(index["compiler_revisions"]) == 2
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["output_policy"] == "radiance_header_fp8_v1"
    assert {case["case"] for case in index["cases"]} == {
        "fp8_64x64x64", "fp4_64x64x64", "fp8_128x128x256",
        "fp4_128x128x128", "fp8_128x128x128", "fp4_128x128x512"}
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593" /
                           f"{index['profile']}.json")
    for row in index["cases"]:
        folder = EVIDENCE / row["case"]
        for filename, digest in row["files_sha256"].items():
            assert _sha(folder / filename) == digest
        receipt = json.loads((folder / "receipt.json").read_text())
        manifest, resources = load_bundle(folder / "bundle")
        bound = (folder / "bound.mlir").read_text()
        m, n, k = row["shape_mnk"]
        assert manifest_sha256(manifest) == row["payload_manifest_sha256"]
        assert receipt["compiler_revision"] == row["compiler_revision"]
        assert receipt["compiler_revision"] in index["compiler_revisions"]
        assert receipt["compiler_source_closure_sha256"] == (
            index["compiler_source_closure_sha256"])
        assert receipt["rtl_revision"] == index["rtl_revision"]
        assert receipt["status"] == "radiance_header_matched_on_pinned_spike"
        assert receipt["spike_exit_code"] == 0
        assert receipt["elf_sha256"] == row["elf_sha256"]
        assert receipt["spike_log_sha256"] == row["spike_log_sha256"]
        assert _sha(folder / "spike.log") == row["spike_log_sha256"]
        assert receipt["source_header_sha256"] == row["source_header_sha256"]
        assert receipt["source_driver_sha256"] == row["source_driver_sha256"]
        assert receipt["bound_mlir_sha256"] == _sha(folder / "bound.mlir")
        assert receipt["compared_source_fp8_codes"] == m * n
        assert receipt["compared_source_e8m0_scales"] == m * n // 32
        assert f"0 Radiance FP8 code mismatches, 0 E8M0 scale mismatches" in (
            folder / "spike.log").read_text()
        assert (resources["golden_fp8"], resources["golden_output_scales"]) == (
            quantize_bf16_radiance_header_fp8(resources["golden_bf16"], m, n))
        report = verify_ir(bound, profile)
        assert (report["contracts"], report["source_resources"]) == (1, 4)
        program = lower_bound_source(bound, profile, manifest, resources)
        assert program.output_format == "radiance_header_fp8"
        assert program.shape == (m, n, k)
        regenerated = write_standalone_sources(tmp_path / row["case"], program, resources)
        for name in ("mx_issue.c", "mx_driver.c", "physical_program.json"):
            assert regenerated["files_sha256"][name] == _sha(folder / name)
            assert receipt["files_sha256"][name] == _sha(folder / name)
