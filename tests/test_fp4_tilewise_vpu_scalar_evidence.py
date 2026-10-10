"""Recheck generated FP4 scalar MULS capture and two-run Spike evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.bind_payload import append_tilewise_vpu_muls
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.quant_reference import bf16_mul_scalar
from mx_gemmini_support.source_payload import load_bundle, validate_derived_gemm_fixture
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_fp4_generated_tilewise_vpu_scalar_266c593"
BASELINE = ROOT / "docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593"


def _archived(name: str) -> bytes:
    compressed = EVIDENCE / f"{name}.gz"
    return gzip.decompress(compressed.read_bytes()) if compressed.exists() else (
        EVIDENCE / name).read_bytes()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_generated_fp4_scalar_1p5_full_output_spike_evidence(tmp_path):
    index = json.loads(_archived("index.json"))
    repro = json.loads(_archived("index_repro.json"))
    a = json.loads(json.dumps(index))
    b = json.loads(json.dumps(repro))
    a["files_sha256"].pop("build/artifact_manifest.json")
    b["files_sha256"].pop("build/artifact_manifest.json")
    assert a == b
    assert index["schema"] == "mx_gemmini.radiance_generated_fp4_tilewise_vpu_scalar_spike.v1"
    assert index["status"] == "generated_fp4_fixture_vpu_scalar_matched_on_pinned_spike"
    assert "no committed Radiance driver or source ELF parity" in index["scope"]
    assert index["scalar_bf16_bits"] == 0x3fc0 and index["scalar_value"] == 1.5
    assert index["compiler_revision"] == "c6b41c0b94a7f9308ddc7a1a2d5ae187fd3e2a6c"
    assert index["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["output_tiles"] == index["vpu_commands"] == 4
    assert index["compared_bf16_outputs"] == 65536
    for name, digest in index["files_sha256"].items():
        assert _sha(_archived(name)) == digest

    baseline = json.loads((BASELINE / "index.json").read_text())
    assert index["source_derivation"] == baseline["source_derivation"]
    trace = json.loads(_archived("capture_trace.json"))
    assert trace["status"] == "complete" and not trace["blockers"]
    calls = [node for node in trace["graphs"]["original"]["nodes"]
             if node["op"] == "call_function"]
    assert [node["target"] for node in calls] == [
        "aten.matmul.default", "aten.mul.Tensor"]
    assert calls[1]["args"][1] == 1.5
    assert b"1.500000e+00" in _archived("frontend.mlir")

    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    manifest, resources = load_bundle(EVIDENCE / "bundle")
    validate_derived_gemm_fixture(manifest)
    payload = _archived("payload_bound.mlir").decode()
    bound = _archived("tilewise_bound.mlir").decode()
    assert append_tilewise_vpu_muls(payload, profile, manifest, 0x3fc0) == bound
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.derived_vpu_scalar_bf16 == 0x3fc0
    assert program.derived_expected_bf16 == bf16_mul_scalar(
        resources["golden_bf16"], 0x3fc0)
    assert _sha(program.derived_expected_bf16) == _sha(
        _archived("build/derived_expected_bf16.bin"))
    assert program.receipt() == json.loads(_archived("build/physical_program.json"))
    assert sum(step.phase == "vpu" for step in program.steps) == 4
    generated = write_standalone_sources(tmp_path / "generated", program, resources)
    for name in ("mx_issue.c", "mx_driver.c", "physical_program.json"):
        assert generated["files_sha256"][name] == index[
            "files_sha256"][f"build/{name}"]

    receipt = json.loads(_archived("build/artifact_manifest.json"))
    repro_receipt = json.loads(_archived("build/artifact_manifest_repro.json"))
    first, second = dict(receipt), dict(repro_receipt)
    first.pop("build_log_sha256")
    second.pop("build_log_sha256")
    assert first == second
    assert receipt["status"] == "derived_vpu_golden_matched_on_pinned_spike"
    assert receipt["golden_basis"] == "derived_bf16_muls"
    assert receipt["compared_bf16_outputs"] == 65536
    assert receipt["spike_exit_code"] == 0
    assert receipt["elf_sha256"] == index["elf_sha256"]
    assert "0 BF16 mismatches" in _archived("build/spike.log").decode()
