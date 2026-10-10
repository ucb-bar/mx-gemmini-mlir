"""Recheck two FP6 fullout source programs built from model2MLIR captures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle, manifest_sha256
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_fp6_fullout_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_radiance_fp6_fullout_spike_evidence(tmp_path):
    index = json.loads((EVIDENCE / "qualification.json").read_text())
    assert index["schema"] == "mx_gemmini.radiance_fp6_fullout_spike.v1"
    assert index["compiler_revision"].startswith("8fa6822")
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert {row["k"] for row in index["cases"]} == {128, 256, 512, 1024}
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593" /
                           f"{index['profile']}.json")
    for row in index["cases"]:
        folder = EVIDENCE / row["case"]
        for filename, digest in row["files_sha256"].items():
            assert _sha(folder / filename) == digest
        receipt = json.loads((folder / "receipt.json").read_text())
        manifest, resources = load_bundle(folder / "bundle")
        bound = (folder / "bound.mlir").read_text()
        m, n, k = manifest["shape_mnk"]
        assert (m, n, k) == (128, 128, row["k"])
        assert manifest_sha256(manifest) == row["payload_manifest_sha256"]
        assert receipt["compiler_revision"] == row["compiler_revision"]
        assert receipt["compiler_revision"] in index["compiler_revisions"]
        assert receipt["status"] == "source_golden_matched_on_pinned_spike"
        assert receipt["spike_exit_code"] == 0
        assert receipt["elf_sha256"] == row["elf_sha256"]
        assert receipt["source_header_sha256"] == _sha(
            ROOT / "docs/evidence" / row["source_header_path"])
        if (folder / "generation_receipt.json").is_file():
            generation = json.loads((folder / "generation_receipt.json").read_text())
            assert generation["generated_header_sha256"] == receipt["source_header_sha256"]
        assert receipt["source_driver_sha256"] == _sha(folder / "source_driver.cpp")
        assert receipt["bound_mlir_sha256"] == _sha(folder / "bound.mlir")
        assert receipt["compared_bf16_outputs"] == m * n
        assert "0 BF16 mismatches" in (folder / "spike.log").read_text()
        assert verify_ir(bound, profile)["source_resources"] == 7
        program = lower_bound_source(bound, profile, manifest, resources)
        assert program.output_format == "bf16"
        assert program.source_golden_preserving
        regenerated = write_standalone_sources(tmp_path / row["case"], program, resources)
        for name in ("mx_issue.c", "mx_driver.c", "physical_program.json"):
            assert regenerated["files_sha256"][name] == _sha(folder / name)

    pristine = EVIDENCE / "pristine_cli_1024"
    reproduction = index["pristine_cli_reproduction"]
    for filename, digest in reproduction["files_sha256"].items():
        assert _sha(pristine / filename) == digest
    receipt = json.loads((pristine / "receipt.json").read_text())
    case = next(row for row in index["cases"] if row["k"] == 1024)
    assert receipt["compiler_revision"] == reproduction["compiler_revision"]
    assert receipt["status"] == "source_golden_matched_on_pinned_spike"
    assert receipt["elf_sha256"] == case["elf_sha256"]
    assert receipt["spike_log_sha256"] == case["spike_log_sha256"]
