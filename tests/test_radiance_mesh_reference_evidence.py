"""Rebuild the DIM8/DIM32 physical streams from archived Radiance captures."""

from __future__ import annotations

from copy import deepcopy
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
from mx_gemmini_support.source_payload import (load_bundle,
                                               validate_target_mesh_reference)
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_mx_mesh_reference_266c593"
FRONTEND = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22/frontend"
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"
MATRIX = json.loads((EVIDENCE / "index.json").read_text())
CASES = [(entry["profile_name"], case)
         for entry in MATRIX["profiles"] for case in entry["cases"]]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(root: Path, name: str) -> bytes:
    path = root / name
    return path.read_bytes() if path.exists() else gzip.decompress(
        (root / f"{name}.gz").read_bytes())


def test_target_mesh_matrix_has_two_reproducible_profile_runs():
    assert MATRIX["schema"] == "mx_gemmini.radiance_mesh_reference_spike_matrix.v1"
    assert MATRIX["status"] == "four_target_mesh_reference_cases_matched_twice_on_pinned_spike"
    assert MATRIX["compiler_revision"] == "3a51a40e4b77297d4fcb1568743b249d2867dbd6"
    assert MATRIX["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert MATRIX["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert MATRIX["profile_count"] == 2
    assert MATRIX["case_count"] == len(CASES) == 4
    assert MATRIX["compared_bf16_outputs_per_run"] == 40960
    for entry in MATRIX["profiles"]:
        root = EVIDENCE / entry["profile_name"]
        first = (root / "qualification.json").read_bytes()
        second = (root / "qualification_repro.json").read_bytes()
        assert _sha(first) == entry["qualification_sha256"]
        assert _sha(second) == entry["qualification_repro_sha256"]
        a, b = json.loads(first), json.loads(second)
        assert a["profile_sha256"] == b["profile_sha256"] == entry["profile_sha256"]
        assert a["compared_bf16_outputs"] == b["compared_bf16_outputs"] == 20480
        assert a["status"] == b["status"] == (
            "target_mesh_reference_precisions_matched_on_pinned_spike")
        x, y = deepcopy(a), deepcopy(b)
        for summary in (x, y):
            for row in summary["rows"]:
                row.pop("artifact_manifest_sha256")
        assert x == y


@pytest.mark.parametrize("profile_name,precision", CASES)
def test_mesh_reference_case_rederives_bound_commands_and_spike_receipt(
        profile_name: str, precision: str, tmp_path):
    entry = next(item for item in MATRIX["profiles"]
                 if item["profile_name"] == profile_name)
    case = entry["cases"][precision]
    root = EVIDENCE / profile_name / precision
    profile = load_profile(PROFILES / f"{profile_name}.json")
    assert entry["profile_sha256"] == profile_sha256(profile)
    assert profile["geometry"]["mesh_columns"] == entry["mesh_dim"]
    summary = json.loads((EVIDENCE / profile_name / "qualification.json").read_text())
    row = next(item for item in summary["rows"]
               if item["precision"].lower() == precision)
    stem = Path(row["driver"]).stem
    handoff = (FRONTEND / stem / "mx_gemm.handoff.mlir").read_text()
    selected = bind_handoff(handoff, profile)
    assert selected == _read(root, "profile_bound.mlir").decode()
    manifest, resources = load_bundle(root / "bundle")
    assert manifest["origin"] == "radiance_source_target_mesh_reference"
    assert manifest["target_mesh_reference"] == case["target_mesh_reference"]
    assert manifest["target_mesh_reference"]["mesh_dim"] == entry["mesh_dim"]
    assert _sha(_read(root, "source_golden_bf16.bin")) == case["source_golden_sha256"]
    assert _sha(resources["golden_bf16"]) == case["target_golden_sha256"]
    assert case["source_golden_sha256"] != case["target_golden_sha256"]
    bound = bind_payload(selected, profile, manifest)
    assert bound == _read(root, "payload_bound.mlir").decode()
    program = lower_bound_source(bound, profile, manifest, resources)
    assert not program.source_golden_preserving
    assert program.golden_origin == "target_mesh_reference"
    assert program.receipt() == json.loads(_read(root, "build/physical_program.json"))
    emitted = write_standalone_sources(tmp_path / "emitted", program, resources)
    for name in ("mx_issue.c", "mx_driver.c", "physical_program.json"):
        assert emitted["files_sha256"][name] == _sha(_read(root, f"build/{name}"))
    assert _sha(_read(root, "build/mx_program.elf")) == case["elf_sha256"]
    assert _sha(_read(root, "build/spike.log")) == case["spike_log_sha256"]
    assert "0 BF16 mismatches" in _read(root, "build/spike.log").decode()
    assert f"dim = {entry['mesh_dim']}" in _read(root, "build/spike.log").decode()
    receipts = [json.loads(_read(root, name)) for name in (
        "build/artifact_manifest.json", "build/artifact_manifest_repro.json")]
    for receipt in receipts:
        assert receipt["status"] == "target_mesh_reference_matched_on_pinned_spike"
        assert receipt["spike_extension_name"] == f"gemmini_dim{entry['mesh_dim']}"
        assert receipt["spike_exit_code"] == 0
        assert receipt["compared_bf16_outputs"] == case["compared_bf16_outputs"]
        assert receipt["elf_sha256"] == case["elf_sha256"]
    receipts[0].pop("build_log_sha256")
    receipts[1].pop("build_log_sha256")
    assert receipts[0] == receipts[1]
    bad_manifest = deepcopy(manifest)
    bad_manifest["target_mesh_reference"]["mesh_dim"] = (
        32 if entry["mesh_dim"] == 8 else 8)
    with pytest.raises(ValueError, match="reference differs from selected profile"):
        bind_payload(selected, profile, bad_manifest)


@pytest.mark.skipif(not os.getenv("RADIANCE_KERNELS_ROOT"),
                    reason="set RADIANCE_KERNELS_ROOT to recompile the pinned source host model")
@pytest.mark.parametrize("profile_name,precision", CASES)
def test_target_mesh_reference_rederives_from_pinned_radiance_model(
        profile_name: str, precision: str):
    source = Path(os.environ["RADIANCE_KERNELS_ROOT"])
    root = EVIDENCE / profile_name / precision
    manifest, resources = load_bundle(root / "bundle")
    source_golden = _read(root, "source_golden_bf16.bin")
    resources["golden_bf16"] = source_golden
    target, policy = derive_mesh_reference(
        source, resources, tuple(manifest["shape_mnk"]), manifest["precision"],
        manifest["target_mesh_reference"]["mesh_dim"])
    assert target == (root / "bundle/golden_bf16.bin").read_bytes()
    assert policy == manifest["target_mesh_reference"]


@pytest.mark.skipif(not os.getenv("RADIANCE_KERNELS_ROOT"),
                    reason="set RADIANCE_KERNELS_ROOT to recompile the pinned source host model")
def test_rtl_product_floor_reference_is_pinned_and_fail_closed():
    root = EVIDENCE / MATRIX["profiles"][0]["profile_name"] / "fp8"
    manifest, resources = load_bundle(root / "bundle")
    resources["golden_bf16"] = _read(root, "source_golden_bf16.bin")
    target, policy = derive_mesh_reference(
        Path(os.environ["RADIANCE_KERNELS_ROOT"]), resources,
        tuple(manifest["shape_mnk"]), "FP8", 8, product_floor=True)
    assert policy["schema"] == "mx_gemmini.radiance_target_mesh_reference.v2"
    assert policy["product_floor_exponent"] == -16
    assert policy["target_golden_sha256"] == _sha(target)
    manifest["target_mesh_reference"] = policy
    manifest["resources"]["golden_bf16"]["sha256"] = _sha(target)
    validate_target_mesh_reference(manifest)
    manifest["target_mesh_reference"]["product_floor_exponent"] = -15
    with pytest.raises(ValueError, match="product-floor provenance"):
        validate_target_mesh_reference(manifest)
