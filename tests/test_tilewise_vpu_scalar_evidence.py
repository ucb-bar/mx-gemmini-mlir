"""Recheck captured BF16 scalar epilogue and two-run Nicolas Spike evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.bind_payload import append_tilewise_vpu_muls
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.quant_reference import bf16_mul_scalar
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_tilewise_vpu_scalar_266c593"


def _archived(name: str) -> bytes:
    compressed = EVIDENCE / f"{name}.gz"
    return gzip.decompress(compressed.read_bytes()) if compressed.exists() else (
        EVIDENCE / name).read_bytes()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_captured_scalar_1p5_full_output_spike_evidence(tmp_path):
    index = json.loads(_archived("index.json"))
    repro = json.loads(_archived("index_repro.json"))
    first_stable = json.loads(json.dumps(index))
    repro_stable = json.loads(json.dumps(repro))
    first_stable["files_sha256"].pop("build/artifact_manifest.json")
    repro_stable["files_sha256"].pop("build/artifact_manifest.json")
    assert first_stable == repro_stable
    assert index["schema"] == "mx_gemmini.radiance_tilewise_vpu_scalar_spike.v1"
    assert index["status"] == "source_derived_vpu_scalar_matched_on_pinned_spike"
    assert index["scalar_bf16_bits"] == 0x3fc0
    assert index["scalar_value"] == 1.5
    assert index["compiler_revision"] == "082c47e1d5ffd60267af97459fe954c785a1bf8e"
    assert index["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["output_tiles"] == index["vpu_commands"] == 4
    assert index["compared_bf16_outputs"] == 65536
    for name, digest in index["files_sha256"].items():
        assert _sha(_archived(name)) == digest

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
    original = _archived("payload_bound.mlir").decode()
    bound = _archived("tilewise_bound.mlir").decode()
    assert append_tilewise_vpu_muls(original, profile, manifest, 0x3fc0) == bound
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.derived_vpu_scalar_bf16 == 0x3fc0
    assert program.derived_expected_bf16 == bf16_mul_scalar(
        resources["golden_bf16"], 0x3fc0)
    assert _sha(program.derived_expected_bf16) == _sha(
        _archived("build/derived_expected_bf16.bin"))
    assert program.receipt() == json.loads(_archived("build/physical_program.json"))
    vpu = [step for step in program.steps if step.phase == "vpu"]
    assert len(vpu) == 4
    generated = write_standalone_sources(tmp_path / "generated", program, resources)
    for name in ("mx_issue.c", "mx_driver.c", "physical_program.json"):
        assert generated["files_sha256"][name] == index[
            "files_sha256"][f"build/{name}"]

    receipt = json.loads(_archived("build/artifact_manifest.json"))
    repro_receipt = json.loads(_archived("build/artifact_manifest_repro.json"))
    a, b = dict(receipt), dict(repro_receipt)
    a.pop("build_log_sha256")
    b.pop("build_log_sha256")
    assert a == b
    assert receipt["status"] == "derived_vpu_golden_matched_on_pinned_spike"
    assert receipt["golden_basis"] == "derived_bf16_muls"
    assert receipt["compared_bf16_outputs"] == 65536
    assert receipt["spike_exit_code"] == 0
    assert receipt["elf_sha256"] == index["elf_sha256"]
    assert "0 BF16 mismatches" in _archived("build/spike.log").decode()
