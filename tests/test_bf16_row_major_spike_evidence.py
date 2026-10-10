"""Audit source-bound row-major BF16 output on Nicolas's pinned Spike."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.command_ir import emit_c
from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile
from tools.emit_mx_object import _referenced_buffers


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/bf16_row_major_readout_266c593"
PROFILE = ROOT / ("profiles/gemmini-mx-cleanup-266c593/"
                  "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(case: Path, name: str) -> bytes:
    path = case / name
    return path.read_bytes() if path.is_file() else gzip.decompress(
        (case / f"{name}.gz").read_bytes())


def test_three_row_major_spike_cases_regenerate_from_typed_mlir():
    index = json.loads((EVIDENCE / "index.json").read_text())
    assert index["schema"] == "mx_gemmini.bf16_row_major_spike.v1"
    assert index["status"] == "three_multi_tile_row_major_cases_matched_on_pinned_spike"
    assert index["compiler_revision"] == "5b542807fd510250313e9cf9671794e5c21cfb75"
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert set(index["cases"]) == {
        "fp4_source_cli", "fp8_tilewise_vpu", "fp8_retile64"}
    profile = load_profile(PROFILE)

    for name, case_index in index["cases"].items():
        case = EVIDENCE / name
        for filename, digest in case_index["files_sha256"].items():
            assert _sha((case / filename).read_bytes()) == digest
        manifest = json.loads(_read(case, "bundle_manifest.json"))
        _, resources = load_bundle(ROOT / case_index["source_bundle"])
        assert manifest["origin"] == case_index["bundle_origin"]
        for key in ("source_driver_sha256", "source_header_sha256"):
            assert manifest[key] == case_index[key]
        bound = _read(case, "bound.mlir").decode()
        receipt = json.loads(_read(case, "artifact_manifest.json"))
        repro = json.loads(_read(case, "artifact_manifest_repro.json"))
        assert receipt["compiler_revision"] == index["compiler_revision"]
        assert receipt["rtl_revision"] == index["rtl_revision"]
        assert receipt["spike_exit_code"] == 0
        assert receipt["status"] in {
            "source_golden_matched_on_pinned_spike",
            "derived_vpu_golden_matched_on_pinned_spike"}
        assert receipt["status"] == case_index["spike_status"]
        if "golden_basis" in case_index:
            assert receipt["golden_basis"] == case_index["golden_basis"]
        assert receipt["compared_bf16_outputs"] == case_index["compared_bf16_outputs"]
        assert receipt["bound_mlir_sha256"] == _sha(bound.encode())
        assert receipt["shape_mnk"] == case_index["shape_mnk"]
        assert _sha(_read(case, "mx_program.elf")) == receipt["elf_sha256"]
        assert _sha(_read(case, "spike.log")) == receipt["spike_log_sha256"]
        assert b"0 BF16 mismatches" in _read(case, "spike.log")

        first_stable, repro_stable = deepcopy(receipt), deepcopy(repro)
        assert first_stable["build_log_sha256"].pop("link.log") != (
            repro_stable["build_log_sha256"].pop("link.log"))
        assert first_stable == repro_stable

        assert 'memory_layout = "row_major_bf16"' in bound
        program = lower_bound_source(bound, profile, manifest, resources)
        physical = _read(case, "physical_program.json")
        assert program.receipt() == json.loads(physical)
        assert _sha(physical) == receipt["files_sha256"]["physical_program.json"]
        assert program.plan["bf16_output_layout"] == "row_major_bf16"
        assert len(program.plan["output_tiles"]) == 4
        assert sum(step.phase == "vpu" for step in program.steps) == (
            case_index["vpu_commands"])
        assert emit_c([step.command for step in program.steps],
                      transport="rocket_rocc",
                      buffers=tuple(entry["name"] for entry in
                                    _referenced_buffers(program, manifest))) == (
                                        _read(case, "mx_issue.c").decode())
        driver = _read(case, "mx_driver.c")
        assert b"if (got[i] != expected[i])" in driver
        assert b"uint32_t tile =" not in driver

        # Account for every output byte exactly once, including the strips
        # written by separate 64- or 128-column output tiles.
        coverage: Counter[int] = Counter()
        for step in program.steps:
            command = step.command
            if (step.phase != "readout" or
                    getattr(getattr(command, "rs1", None), "buffer", None)
                    != "output_bf16"):
                continue
            rows = (command.rs2.immediate >> 48) & 0xffff
            cols = (command.rs2.immediate >> 32) & 0xffff
            assert 1 <= rows <= 16 and cols == 16
            for offset in range(rows * cols):
                coverage[command.rs1.byte_offset + offset] += 1
        m, n, _ = case_index["shape_mnk"]
        assert len(coverage) == m * n * 2
        assert set(coverage) == set(range(m * n * 2))
        assert set(coverage.values()) == {1}

        if name == "fp8_retile64":
            assert case_index["tile_mnk"] == [64, 64, 256]
            assert "retiling" in case_index["scope"]
        else:
            obj = json.loads(_read(case, "object_manifest.json"))
            assert obj["schema"] == "mx_gemmini.linkable_object.v1"
            assert obj["allocated_data_section_bytes"] == 0
            assert obj["embedded_operand_bytes"] == 0
            assert obj["embedded_golden_bytes"] == 0
            assert obj["object_sha256"] == _sha(_read(case, "mx_issue.o"))
            assert obj["issuer_c_sha256"] == _sha(_read(case, "mx_issue.c"))
            assert obj["physical_program_sha256"] == _sha(physical)
            assert obj["buffer_abi"] == _referenced_buffers(program, manifest)
            assert obj["buffer_abi"][2]["layout"] == "row_major_bf16"
            assert obj["buffer_abi"][2]["minimum_bytes"] == m * n * 2
            assert "source ELF parity" in case_index["scope"]
