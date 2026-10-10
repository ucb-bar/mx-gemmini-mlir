"""Recheck captured affine graphs and two-run Nicolas Spike evidence."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mx_gemmini_support.capture_epilogue import (
    append_captured_tilewise_vpu_muls, append_captured_tilewise_vpu_scalar)
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.quant_reference import bf16_add_scalar, bf16_mul_scalar
from mx_gemmini_support.source_payload import load_bundle, validate_derived_gemm_fixture
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_tilewise_vpu_affine_266c593"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


def _read(root: Path, name: str) -> bytes:
    path = root / name
    return path.read_bytes() if path.exists() else gzip.decompress(
        (root / f"{name}.gz").read_bytes())


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("precision", ("fp8", "fp4"))
def test_captured_affine_full_output_spike_evidence(precision: str, tmp_path):
    root = EVIDENCE / precision
    index = json.loads(_read(root, "index.json"))
    repro = json.loads(_read(root, "index_repro.json"))
    first, second = deepcopy(index), deepcopy(repro)
    first["files_sha256"].pop("build/artifact_manifest.json")
    second["files_sha256"].pop("build/artifact_manifest.json")
    assert first == second
    assert index["schema"] == (
        "mx_gemmini.radiance_tilewise_vpu_affine_spike.v1" if precision == "fp8"
        else "mx_gemmini.radiance_generated_fp4_tilewise_vpu_affine_spike.v1")
    assert index["compiler_revision"] == "3935032f37f192e3eb02a43bd981baa4c15f2d92"
    assert index["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["epilogue"] == "affine"
    assert index["scalar_bf16_bits"] == 0x4000
    assert index["second_scalar_bf16_bits"] == 0x3fc0
    assert index["output_tiles"] == 4
    assert index["vpu_commands"] == 8
    assert index["compared_bf16_outputs"] == 65536
    if precision == "fp4":
        assert "no committed Radiance driver or source ELF parity" in index["scope"]
    for name, digest in index["files_sha256"].items():
        assert _sha(_read(root, name)) == digest

    trace = json.loads(_read(root, "capture_trace.json"))
    calls = [node for node in trace["graphs"]["original"]["nodes"]
             if node["op"] == "call_function"]
    assert [node["target"] for node in calls] == [
        "aten.matmul.default", "aten.mul.Tensor", "aten.add.Tensor"]
    assert calls[1]["args"][1] == 2.0 and calls[1]["kwargs"] == {}
    assert calls[2]["args"][1] == 1.5 and calls[2]["kwargs"] == {}
    capture = SimpleNamespace(
        ok=True, capture_trace=trace,
        quantization_manifest=json.loads(_read(root, "quantization_manifest.json")),
        mlir_text=_read(root, "frontend.mlir").decode())
    profile = load_profile(PROFILE)
    manifest, resources = load_bundle(root / "bundle")
    if precision == "fp4":
        validate_derived_gemm_fixture(manifest)
    payload = _read(root, "payload_bound.mlir").decode()
    bound = _read(root, "tilewise_bound.mlir").decode()
    assert append_captured_tilewise_vpu_scalar(
        payload, profile, manifest, capture) == bound
    with pytest.raises(ValueError):
        append_captured_tilewise_vpu_muls(payload, profile, manifest, capture)
    changed = deepcopy(capture)
    changed.capture_trace["graphs"]["original"]["nodes"][4]["args"][1] = 1.0
    with pytest.raises(ValueError):
        append_captured_tilewise_vpu_scalar(payload, profile, manifest, changed)

    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.plan["vector_tile_policy"] == "bf16_scalar_chain_each_output_tile_v1"
    assert program.derived_vpu_scalar_chain == (("muls", 0x4000), ("adds", 0x3fc0))
    assert program.derived_expected_bf16 == bf16_add_scalar(
        bf16_mul_scalar(resources["golden_bf16"], 0x4000), 0x3fc0)
    assert _sha(program.derived_expected_bf16) == _sha(
        _read(root, "build/derived_expected_bf16.bin"))
    assert program.receipt() == json.loads(_read(root, "build/physical_program.json"))
    commands = [step.command for step in program.steps if step.phase == "vpu"]
    assert len(commands) == 8
    assert [command.rs2.immediate & 0x1f for command in commands] == [4, 3] * 4
    emitted = write_standalone_sources(tmp_path / precision, program, resources)
    for name in ("mx_issue.c", "mx_driver.c", "physical_program.json"):
        assert emitted["files_sha256"][name] == index["files_sha256"][f"build/{name}"]
    receipts = [json.loads(_read(root, name)) for name in (
        "build/artifact_manifest.json", "build/artifact_manifest_repro.json")]
    for receipt in receipts:
        assert receipt["status"] == "derived_vpu_golden_matched_on_pinned_spike"
        assert receipt["golden_basis"] == "derived_bf16_scalar_chain"
        assert receipt["spike_exit_code"] == 0
        assert receipt["compared_bf16_outputs"] == 65536
        assert receipt["elf_sha256"] == index["elf_sha256"]
    a, b = (dict(receipt) for receipt in receipts)
    a.pop("build_log_sha256")
    b.pop("build_log_sha256")
    assert a == b
    assert "0 BF16 mismatches" in _read(root, "build/spike.log").decode()


def test_capture_derived_source_cli_affine_reproduces_qualified_fp8_program():
    root = EVIDENCE / "fp8"
    index = json.loads(_read(root, "index.json"))
    cli = json.loads(_read(root, "cli_capture_index.json"))
    assert cli["schema"] == "mx_gemmini.capture_derived_source_cli_affine_spike.v1"
    assert cli["status"] == "two_capture_derived_affine_cli_runs_matched_on_pinned_spike"
    assert cli["source_index_sha256"] == _sha(_read(root, "index.json"))
    assert cli["compiler_revision"] == index["compiler_revision"]
    assert cli["compiler_source_closure_sha256"] == index[
        "compiler_source_closure_sha256"]
    assert cli["capture_scalar_chain"] == [
        {"kind": "muls", "immediate_bf16": 0x4000},
        {"kind": "adds", "immediate_bf16": 0x3fc0}]
    assert cli["compared_bf16_outputs"] == 65536
    for field, path in (("bound_mlir_sha256", "tilewise_bound.mlir"),
                        ("physical_program_sha256", "build/physical_program.json"),
                        ("generated_issue_sha256", "build/mx_issue.c"),
                        ("spike_log_sha256", "build/spike.log")):
        assert cli[field] == _sha(_read(root, path))
    assert cli["elf_sha256"] == index["elf_sha256"]
    receipts = [_read(root, name) for name in (
        "cli_capture_artifact_manifest.json",
        "cli_capture_artifact_manifest_repro.json")]
    for data, digest in zip(receipts, cli["artifact_manifest_sha256"]):
        assert _sha(data) == digest
    first, second = (json.loads(data) for data in receipts)
    for receipt in (first, second):
        assert receipt["golden_basis"] == "derived_bf16_scalar_chain"
        assert receipt["status"] == "derived_vpu_golden_matched_on_pinned_spike"
        assert receipt["compared_bf16_outputs"] == 65536
        assert receipt["elf_sha256"] == cli["elf_sha256"]
    first.pop("build_log_sha256")
    second.pop("build_log_sha256")
    assert first == second
