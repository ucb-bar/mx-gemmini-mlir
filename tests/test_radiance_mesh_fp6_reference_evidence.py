"""Check the two reproducible FP6 target mesh Spike qualifications."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import pytest

from mx_gemmini_support.bind_payload import bind_payload
from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.mesh_reference import derive_mesh_reference
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_mx_mesh_fp6_reference_266c593"
FRONTEND = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22/frontend"
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"
INDEX = json.loads((EVIDENCE / "index.json").read_text())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(root: Path, name: str) -> bytes:
    path = root / name
    return path.read_bytes() if path.exists() else gzip.decompress(
        (root / f"{name}.gz").read_bytes())


def test_fp6_mesh_matrix_is_pinned():
    assert INDEX["schema"] == "mx_gemmini.radiance_mesh_fp6_reference_spike_matrix.v1"
    assert INDEX["status"] == "two_fp6_target_mesh_reference_cases_matched_twice_on_pinned_spike"
    assert INDEX["compiler_revision"] == "7944c9eb42eab3d2897514e55f9f65b8567e926d"
    assert INDEX["source_revision"] == "80f84caedbabc663a7433c1da4455b936cca41f3"
    assert INDEX["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert INDEX["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert INDEX["profile_count"] == INDEX["case_count"] == 2
    assert INDEX["compared_bf16_outputs_per_run"] == 32768


@pytest.mark.parametrize("entry", INDEX["profiles"], ids=lambda row: f"dim{row['mesh_dim']}")
def test_fp6_mesh_case_rederives_commands_and_reference(entry: dict, tmp_path):
    root = EVIDENCE / entry["profile_name"]
    first = json.loads((root / "qualification.json").read_text())
    second = json.loads((root / "qualification_repro.json").read_text())
    assert _sha((root / "qualification.json").read_bytes()) == entry["qualification_sha256"]
    assert _sha((root / "qualification_repro.json").read_bytes()) == entry[
        "qualification_repro_sha256"]
    for summary in (first, second):
        assert summary["status"] == "target_mesh_reference_precisions_matched_on_pinned_spike"
        assert summary["compared_bf16_outputs"] == entry["compared_bf16_outputs"]
        assert summary["profile_sha256"] == entry["profile_sha256"]
    row, repro_row = first["rows"][0], second["rows"][0]
    for field in ("profile_bound_mlir_sha256", "payload_bound_mlir_sha256",
                  "bundle_manifest_sha256", "physical_program_sha256",
                  "generated_issue_sha256", "elf_sha256", "spike_log_sha256",
                  "source_golden_sha256", "target_golden_sha256"):
        assert row[field] == repro_row[field]
    root /= "fp6"
    profile = load_profile(PROFILES / f"{entry['profile_name']}.json")
    assert profile_sha256(profile) == entry["profile_sha256"]
    handoff = (FRONTEND / Path(row["driver"]).stem / "mx_gemm.handoff.mlir").read_text()
    selected = bind_handoff(handoff, profile)
    assert selected == _read(root, "profile_bound.mlir").decode()
    manifest, resources = load_bundle(root / "bundle")
    assert manifest["origin"] == "radiance_source_target_mesh_reference"
    assert manifest["target_mesh_reference"] == entry["target_mesh_reference"]
    assert manifest["target_mesh_reference"]["lut_granularity_shift"] == 1
    assert _sha(_read(root, "source_golden_bf16.bin")) == entry["source_golden_sha256"]
    assert _sha(resources["golden_bf16"]) == entry["target_golden_sha256"]
    assert entry["source_golden_sha256"] != entry["target_golden_sha256"]
    bound = bind_payload(selected, profile, manifest)
    assert bound == _read(root, "payload_bound.mlir").decode()
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.golden_origin == "target_mesh_reference"
    assert not program.source_golden_preserving
    assert program.receipt() == json.loads(_read(root, "build/physical_program.json"))
    emitted = write_standalone_sources(tmp_path / "emitted", program, resources)
    for name in ("mx_issue.c", "mx_driver.c", "physical_program.json"):
        assert emitted["files_sha256"][name] == _sha(_read(root, f"build/{name}"))
    assert _sha(_read(root, "build/mx_program.elf")) == entry["elf_sha256"]
    assert _sha(_read(root, "build/spike.log")) == entry["spike_log_sha256"]
    assert "0 BF16 mismatches" in _read(root, "build/spike.log").decode()
    a = json.loads(_read(root, "build/artifact_manifest.json"))
    b = json.loads(_read(root, "build/artifact_manifest_repro.json"))
    for receipt in (a, b):
        assert receipt["status"] == "target_mesh_reference_matched_on_pinned_spike"
        assert receipt["spike_extension_name"] == f"gemmini_dim{entry['mesh_dim']}"
        assert receipt["compared_bf16_outputs"] == 16384
        assert receipt["spike_exit_code"] == 0
    a.pop("build_log_sha256")
    b.pop("build_log_sha256")
    assert a == b


@pytest.mark.skipif(not os.getenv("RADIANCE_KERNELS_ROOT"),
                    reason="set RADIANCE_KERNELS_ROOT to recompile Radiance's host model")
@pytest.mark.parametrize("entry", INDEX["profiles"], ids=lambda row: f"dim{row['mesh_dim']}")
def test_fp6_mesh_reference_rederives_from_pinned_radiance_model(entry: dict):
    root = EVIDENCE / entry["profile_name"] / "fp6"
    manifest, resources = load_bundle(root / "bundle")
    resources["golden_bf16"] = _read(root, "source_golden_bf16.bin")
    target, policy = derive_mesh_reference(
        Path(os.environ["RADIANCE_KERNELS_ROOT"]), resources,
        tuple(manifest["shape_mnk"]), "FP6", entry["mesh_dim"])
    assert target == _read(root, "bundle/golden_bf16.bin")
    assert policy == entry["target_mesh_reference"]
