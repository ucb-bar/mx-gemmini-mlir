"""Audit the generated FP4 four-tile MX+VPU case without claiming source parity."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.bind_payload import append_tilewise_vpu_x2, bind_payload
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import (
    DERIVED_GEMM_FIXTURE_ORIGIN, load_bundle, validate_derived_gemm_fixture)
from mx_gemmini_support.standalone import write_standalone_sources
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593"
DRIVER = ("fixture/kernels/gemm_mxgemmini/"
          "mxgemm.fp4.m256n256k256.tm128tn128tk128.fullout.cpp")
HEADER = "fixture/kernels/gemm_mxgemmini/mxgemm.data.fp4.m256n256k256.h"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archived(name: str) -> bytes:
    compressed = EVIDENCE / f"{name}.gz"
    return gzip.decompress(compressed.read_bytes()) if compressed.exists() else (
        EVIDENCE / name).read_bytes()


def test_generated_fp4_tilewise_vpu_spike_receipts_and_refusals(tmp_path):
    index = json.loads((EVIDENCE / "index.json").read_text())
    repro = json.loads((EVIDENCE / "index_repro.json").read_text())
    first_stable, repro_stable = deepcopy(index), deepcopy(repro)
    first_stable["files_sha256"].pop("build/artifact_manifest.json")
    repro_stable["files_sha256"].pop("build/artifact_manifest.json")
    assert first_stable == repro_stable
    assert index["schema"] == "mx_gemmini.radiance_generated_fp4_tilewise_vpu_spike.v1"
    assert index["status"] == "generated_fp4_fixture_vpu_x2_matched_on_pinned_spike"
    assert "no committed Radiance driver or source ELF parity" in index["scope"]
    assert index["source_revision"] == "80f84caedbabc663a7433c1da4455b936cca41f3"
    assert index["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["output_tiles"] == index["vpu_commands"] == 4
    assert index["compared_bf16_outputs"] == 256 * 256
    for name, digest in index["files_sha256"].items():
        assert _sha(_archived(name)) == digest

    trace = json.loads(_archived("capture_trace.json"))
    assert trace["status"] == "complete" and not trace["blockers"]
    calls = [node for node in trace["graphs"]["original"]["nodes"]
             if node["op"] == "call_function"]
    assert [node["target"] for node in calls] == ["aten.matmul.default", "aten.mul.Tensor"]
    assert calls[1]["args"][1] == 2.0
    quant = json.loads(_archived("quantization_manifest.json"))
    assert [(site["format"], site["status"], site["shape"])
            for site in quant["sites"]] == [("mxfp4", "quantized", [256, 256, 256])]

    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    manifest, resources = load_bundle(EVIDENCE / "bundle")
    validate_derived_gemm_fixture(manifest)
    assert manifest["origin"] == DERIVED_GEMM_FIXTURE_ORIGIN
    assert manifest["source_derivation"] == index["source_derivation"]
    assert manifest["source_derivation"]["transformation"] == (
        "fp8_m256n256k256_tk256_to_fp4_tk128_with_activation_alias_v2")
    assert manifest["source_driver_sha256"] == _sha(_archived(DRIVER))
    assert manifest["source_header_sha256"] == _sha(_archived(HEADER))
    assert _archived(DRIVER).decode().count(
        "static const uint8_t *A_in = &A_in_hw[0][0];") == 1
    assert len(resources["golden_bf16"]) == 256 * 256 * 2
    payload = _archived("payload_bound.mlir").decode()
    bound = _archived("tilewise_bound.mlir").decode()
    assert append_tilewise_vpu_x2(payload, profile, manifest) == bound
    program = lower_bound_source(bound, profile, manifest, resources)
    assert program.shape == (256, 256, 256)
    assert program.plan["vector_tile_policy"] == "bf16_muls_x2_each_output_tile_v1"
    assert program.receipt() == json.loads(_archived("build/physical_program.json"))
    positions = [i for i, step in enumerate(program.steps) if step.phase == "vpu"]
    assert len(positions) == 4
    for i, position in enumerate(positions):
        following = positions[i + 1] if i < 3 else len(program.steps)
        readouts = [step for step in program.steps[position + 1:following]
                    if step.phase == "readout" and
                    getattr(getattr(step.command, "rs1", None), "buffer", None) == "output_bf16"]
        assert len(readouts) == 128
        assert readouts[0].command.rs1.byte_offset == i * 128 * 128 * 2
    emitted = write_standalone_sources(tmp_path / "emitted", program, resources)
    for name in ("mx_issue.c", "mx_driver.c", "physical_program.json"):
        assert emitted["files_sha256"][name] == index["files_sha256"][f"build/{name}"]

    for mutate in (
        lambda m: m["source_derivation"].pop("base_driver_sha256"),
        lambda m: m["source_derivation"].update({"generated_header_sha256": "0" * 64}),
        lambda m: m.update({"origin": "radiance_source_header_specialization"}),
    ):
        changed = deepcopy(manifest)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_derived_gemm_fixture(changed)
    disguised = deepcopy(manifest)
    disguised["origin"] = "radiance_source_header_specialization"
    with pytest.raises(ValueError):
        bind_payload(_archived("profile_bound.mlir").decode(), profile, disguised)

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
    assert receipt["compared_bf16_outputs"] == 65536
    assert receipt["elf_sha256"] == index["elf_sha256"]
    assert "0 BF16 mismatches" in _archived("build/spike.log").decode()

    source_build = json.loads(_archived("source_build/receipt.json"))
    assert source_build["schema"] == "mx_gemmini.radiance_generated_fp4_source_build.v1"
    assert source_build["status"] == "derived_muon_soc_elf_built_not_executed"
    assert source_build["source_revision"] == index["source_revision"]
    assert source_build["driver_sha256"] == _sha(_archived(DRIVER))
    assert source_build["header_sha256"] == _sha(_archived(HEADER))
    source_elf = _archived("source_build/source.soc.elf")
    assert source_elf[:4] == b"\x7fELF" and source_elf[4] == 2
    assert int.from_bytes(source_elf[18:20], "little") == 243
    assert _sha(source_elf) == source_build["source_soc_elf_sha256"]
    build_log = _archived("source_build/build.log")
    assert _sha(build_log) == source_build["build_log_sha256"]
    assert b"wrote mxgemm.fp4.m256n256k256.tm128tn128tk128.fullout.soc.elf" in build_log
