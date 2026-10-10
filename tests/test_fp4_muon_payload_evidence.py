"""Audit the generated FP4+VPU Muon command stream and executed payload."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.quant_reference import exact_bf16_x2
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile
from tools.link_mx_muon_payload import _tile_major_reference


ROOT = Path(__file__).resolve().parents[1]
OBJECT = ROOT / "docs/evidence/fp4_vpu_muon_mmio_object_266c593"
PAYLOAD = ROOT / "docs/evidence/fp4_vpu_muon_payload_266c593"
PATCH = ROOT / "docs/evidence/fp8_vpu_muon_payload_266c593/cyclotron_compiler_stream.patch"
PROFILE = ROOT / ("profiles/gemmini-mx-cleanup-266c593/"
                  "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archive_index(directory: Path, schema: str) -> dict:
    index = json.loads((directory / "index.json").read_text())
    assert index["schema"] == schema
    for name, digest in index["files_sha256"].items():
        assert _sha((directory / name).read_bytes()) == digest
    return index


def test_fp4_muon_object_matches_typed_mlir_and_physical_schedule():
    index = _archive_index(OBJECT, "mx_gemmini.fp4_vpu_muon_mmio_object_archive.v1")
    mlir = (ROOT / index["source_mlir"]).read_text()
    manifest, resources = load_bundle(ROOT / index["source_bundle"])
    program = lower_bound_source(mlir, load_profile(PROFILE), manifest, resources,
                                 mode="rtl_alternating")
    physical = gzip.decompress((OBJECT / "physical_program.json.gz").read_bytes())
    assert program.receipt() == json.loads(physical)
    receipt = json.loads((OBJECT / "object_manifest.json").read_text())
    assert receipt == json.loads((OBJECT / "object_manifest_repro.json").read_text())
    assert receipt["bound_mlir_sha256"] == _sha(mlir.encode())
    assert receipt["physical_program_sha256"] == _sha(physical)
    assert receipt["physical_mode"] == "rtl_alternating"
    assert receipt["transport"] == "muon_mmio"
    assert receipt["buffer_abi"][2]["layout"] == "output_tile_major_bf16"
    assert receipt["command_count"] == len(program.steps) == 1134
    assert receipt["completion_fences"] == receipt["gateway_busy_waits"] == 33
    assert sum(step.phase == "vpu" for step in program.steps) == 4
    assert len(program.plan["output_tiles"]) == 4
    disassembly = gzip.decompress((OBJECT / "disassembly.txt.gz").read_bytes())
    assert receipt["shared_gateway_stores"] == disassembly.count(b"sw.shared") == 5505
    assert receipt["shared_gateway_loads"] == disassembly.count(b"lw.shared") == 66
    assert receipt["object_sha256"] == _sha((OBJECT / "mx_issue.o").read_bytes())


def test_fp4_muon_payload_matches_source_golden_in_two_clean_cyclotron_builds():
    index = _archive_index(PAYLOAD, "mx_gemmini.fp4_vpu_muon_payload_archive.v1")
    receipt = json.loads((PAYLOAD / "payload_manifest.json").read_text())
    assert receipt == json.loads((PAYLOAD / "payload_manifest_repro.json").read_text())
    assert receipt["precision"] == "FP4"
    assert receipt["shape_mnk"] == [256, 256, 256]
    assert receipt["bf16_output_layout"] == "output_tile_major_bf16"
    assert receipt["object_manifest_sha256"] == _sha((OBJECT / "object_manifest.json").read_bytes())
    assert receipt["payload_linker_sha256"] == _sha((ROOT / "tools/link_mx_muon_payload.py").read_bytes())
    bundle = ROOT / index["source_bundle"]
    _, resources = load_bundle(bundle)
    expected = _tile_major_reference(exact_bf16_x2(resources["golden_bf16"]),
                                     receipt["shape_mnk"], receipt["output_plan"])
    assert receipt["expected_bf16_sha256"] == _sha(expected)
    assert receipt["files_sha256"]["mx_expected_bf16.bin"] == _sha(expected)
    for name in ("activation", "activation_scales", "weight", "weight_scales"):
        assert receipt["files_sha256"][f"{name}.bin"] == _sha(resources[name])
    for name in ("mx_kernel.cpp", "mx_payload.S", "mx_kernel.elf"):
        assert receipt["files_sha256"][name] == _sha((PAYLOAD / name).read_bytes())
    elf = (PAYLOAD / "mx_kernel.elf").read_bytes()
    assert elf[:4] == b"\x7fELF" and elf[4] == 1
    assert int.from_bytes(elf[18:20], "little") == 243

    first = json.loads((PAYLOAD / "patched_cyclotron_qualification.json").read_text())
    repro = json.loads((PAYLOAD / "patched_cyclotron_qualification_repro.json").read_text())
    for qualification in (first, repro):
        assert qualification["status"] == "all_65536_bf16_outputs_matched_on_isolated_patched_cyclotron"
        assert qualification["qualification"] == "experimental_functional_model_only"
        assert qualification["precision"] == "FP4"
        assert qualification["bf16_output_layout"] == "output_tile_major_bf16"
        assert qualification["patch_sha256"] == _sha(PATCH.read_bytes())
        assert qualification["qualifier_sha256"] == _sha((
            ROOT / "tools/qualify_mx_muon_cyclotron.py").read_bytes())
        assert qualification["payload_manifest_sha256"] == _sha((
            PAYLOAD / "payload_manifest.json").read_bytes())
        assert qualification["compiler_elf_sha256"] == _sha(elf)
        assert qualification["expected_bf16_sha256"] == _sha(expected)
        assert qualification["cyclotron_mx_tests_passed"] == 35
        assert len(qualification["runs"]) == 2
    assert first["cyclotron_mx_tests_sha256"] == _sha((
        PAYLOAD / "mxgemmini_tests.log").read_bytes())
    for key in ("cyclotron_binary_sha256", "patched_model_sha256", "patch_sha256"):
        assert first[key] == repro[key]
    dump = gzip.decompress((PAYLOAD / "patched_cyclotron_gmem.bin.gz").read_bytes())
    base = min(first["output_address"], first["mismatch_address"], first["completed_address"])
    offset = first["output_address"] - base
    assert dump[offset:offset + len(expected)] == expected
    assert int.from_bytes(dump[first["mismatch_address"] - base:
                               first["mismatch_address"] - base + 4], "little") == 0
    assert int.from_bytes(dump[first["completed_address"] - base:
                               first["completed_address"] - base + 4], "little") == 1
    for qualification in (first, repro):
        for run in qualification["runs"]:
            assert run["gmem_sha256"] == _sha(dump)
            assert run["output_sha256"] == _sha(expected)
            assert run["completed"] == 1
            assert run["reported_bf16_mismatches"] == 0
            assert run["independent_bf16_mismatches"] == 0
    assert first["runs"][0]["cyclotron_log_sha256"] == _sha((
        PAYLOAD / "first.cyclotron.log").read_bytes())
    assert first["runs"][1]["cyclotron_log_sha256"] == _sha((
        PAYLOAD / "repro.cyclotron.log").read_bytes())
