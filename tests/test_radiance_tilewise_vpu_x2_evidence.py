"""Audit source-bound four-tile MX GEMM followed by one VPU x2 per tile."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.bind_payload import append_tilewise_vpu_x2
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_tilewise_vpu_x2_266c593"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archived(name: str) -> bytes:
    compressed = EVIDENCE / f"{name}.gz"
    if compressed.exists():
        return gzip.decompress(compressed.read_bytes())
    return (EVIDENCE / name).read_bytes()


def test_radiance_tilewise_vpu_x2_spike_evidence(tmp_path):
    index_bytes = (EVIDENCE / "index.json").read_bytes()
    index = json.loads(index_bytes)
    repro = json.loads((EVIDENCE / "index_repro.json").read_text())
    # The linker warning embeds the chosen build directory. All other
    # captured, emitted, and executed bytes must reproduce exactly.
    first_stable = json.loads(index_bytes)
    repro_stable = json.loads(json.dumps(repro))
    first_stable["files_sha256"].pop("build/artifact_manifest.json")
    repro_stable["files_sha256"].pop("build/artifact_manifest.json")
    assert first_stable == repro_stable
    assert index["schema"] == "mx_gemmini.radiance_tilewise_vpu_x2_spike.v1"
    assert index["status"] == "source_derived_vpu_x2_matched_on_pinned_spike"
    assert index["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["compared_bf16_outputs"] == 256 * 256
    assert index["output_tiles"] == index["vpu_commands"] == 4
    for name, digest in index["files_sha256"].items():
        assert _sha(_archived(name)) == digest

    trace = json.loads(_archived("capture_trace.json"))
    assert trace["status"] == "complete" and not trace["blockers"]
    calls = [node for node in trace["graphs"]["original"]["nodes"]
             if node["op"] == "call_function"]
    assert [node["target"] for node in calls] == ["aten.matmul.default", "aten.mul.Tensor"]
    assert calls[1]["args"][1] == 2.0
    assert b"linalg.matmul" in _archived("frontend.mlir")

    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    manifest, resources = load_bundle(EVIDENCE / "bundle")
    original = _archived("payload_bound.mlir").decode()
    bound = _archived("tilewise_bound.mlir").decode()
    assert append_tilewise_vpu_x2(original, profile, manifest) == bound
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.plan["vector_tile_policy"] == "bf16_muls_x2_each_output_tile_v1"
    assert program.shape == (256, 256, 256)
    assert _sha(program.derived_expected_bf16) == json.loads(
        _archived("build/physical_program.json"))["derived_expected_bf16_sha256"]
    assert program.receipt() == json.loads(_archived("build/physical_program.json"))

    vpu_positions = [i for i, step in enumerate(program.steps) if step.phase == "vpu"]
    assert len(vpu_positions) == 4
    for i, position in enumerate(vpu_positions):
        prior = vpu_positions[i - 1] if i else -1
        following = vpu_positions[i + 1] if i < 3 else len(program.steps)
        assert any(step.phase == "compute" for step in program.steps[prior + 1:position])
        readouts = [step for step in program.steps[position + 1:following]
                    if step.phase == "readout" and
                    getattr(getattr(step.command, "rs1", None), "buffer", None) == "output_bf16"]
        assert len(readouts) == 128
        assert readouts[0].command.rs1.byte_offset == i * 128 * 128 * 2
    emitted = write_standalone_sources(tmp_path / "emitted", program, resources)
    for name in ("mx_issue.c", "mx_driver.c", "physical_program.json"):
        assert emitted["files_sha256"][name] == index["files_sha256"][f"build/{name}"]

    for changed in (
        bound.replace("mx.vector_tile_policy = \"bf16_muls_x2_each_output_tile_v1\"",
                      "mx.vector_tile_policy = \"unknown\""),
        bound.replace("immediate_bf16 = 16384 : i32", "immediate_bf16 = 16256 : i32"),
        bound.replace("src1_row = 2048 : i32", "src1_row = 2049 : i32"),
        bound.replace(", mx.vector_tile_policy = \"bf16_muls_x2_each_output_tile_v1\"", ""),
    ):
        assert changed != bound
        with pytest.raises(ValueError):
            lower_bound_source(changed, profile, manifest, resources)

    receipt = json.loads(_archived("build/artifact_manifest.json"))
    repro_receipt = json.loads(_archived("build/artifact_manifest_repro.json"))
    assert _sha(_archived("build/artifact_manifest_repro.json")) == repro[
        "files_sha256"]["build/artifact_manifest.json"]
    first_receipt_stable, repro_receipt_stable = dict(receipt), dict(repro_receipt)
    first_receipt_stable.pop("build_log_sha256")
    repro_receipt_stable.pop("build_log_sha256")
    assert first_receipt_stable == repro_receipt_stable
    assert receipt["status"] == "derived_vpu_golden_matched_on_pinned_spike"
    assert receipt["golden_basis"] == "derived_bf16_x2"
    assert receipt["spike_exit_code"] == 0
    assert receipt["elf_sha256"] == index["elf_sha256"]
    assert receipt["extension_sha256"] == index["extension_sha256"]
    assert receipt["files_sha256"]["derived_expected_bf16.bin"] == _sha(
        program.derived_expected_bf16)
    assert "0 BF16 mismatches" in _archived("build/spike.log").decode()
