"""Audit clean-checkout public compilation of Nicolas's full MX/VPU branch."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.full_chain_pipelined import (
    FULL_INPUTS, FULL_OUTPUTS, lower_full_chain_pipelined)
from mx_gemmini_support.target_profile import load_profile
from tools.emit_full_vpu_branch_object import _buffer_abi


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/mx_full_vpu_branch_public_3619043"
SOURCE = ROOT / "docs/evidence/nicolas_chain_pipelined_full_266c593"
ONLY = ROOT / "docs/evidence/nicolas_chain_pipelined_e4m3_only_266c593"
PROFILE_DIR = ROOT / "profiles/gemmini-mx-cleanup-266c593"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _unpack(path: Path) -> bytes:
    return gzip.decompress(path.read_bytes())


def test_four_public_branch_objects_replay_source_goldens():
    index = _read(EVIDENCE / "index.json")
    source = _read(SOURCE / "object_manifest.json")
    abi = _read(EVIDENCE / "abi.json")
    assert index["schema"] == "mx_gemmini.public_full_vpu_branch_replay.v1"
    assert index["status"] == "four_public_branch_objects_matched_source_on_pinned_spike"
    assert index["compiler_revision"].startswith("3619043")
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert [row["name"] for row in index["rows"]] == [
        "fp4_vpu_serial", "fp4_vpu_pipelined",
        "e4m3_vpu_serial", "e4m3_vpu_pipelined"]
    assert (index["compared_c1_bf16_values"], index["compared_fp8_codes"],
            index["compared_e8m0_scales"]) == (16384, 65536, 2048)
    assert abi == {
        "schema": "mx_gemmini.full_vpu_branch_buffer_map.v1",
        "inputs": {name: name for name in FULL_INPUTS},
        "outputs": {name: name for name in FULL_OUTPUTS},
        "source_reference": "c1_bf16"}
    for name, digest in source["source_resource_sha256"].items():
        assert _sha(_unpack(EVIDENCE / f"{name}.bin.gz")) == digest
        if name in index["input_sha256"]:
            assert index["input_sha256"][name] == digest

    for row in index["rows"]:
        name = row["name"]
        case = EVIDENCE / name
        old = (SOURCE / ("pipelined" if name.endswith("pipelined") else "")
               if name.startswith("fp4_") else
               ONLY / ("compiled_pipelined" if name.endswith("pipelined") else
                       "compiled"))
        obj = _read(case / "object_manifest.json")
        dispatch = _read(case / "compile_manifest.json")
        qualified = _read(case / "spike_qualification.json")
        physical = json.loads(_unpack(case / "physical_program.json.gz"))
        source_physical = _read(old / "physical_program.json")
        assert dispatch["lowering_family"] == "full_vpu_branch"
        assert dispatch["verified_graph_counts"] == {
            "contracts": 1, "resident_contracts": 2,
            "vpu_commands": 2, "spad_requants": 2,
            "source_resources": 0, "lut_uploads": 0, "runtime_luts": 0}
        assert dispatch["native_verifier_sha256"] is not None
        assert obj["schema"] == "mx_gemmini.full_vpu_branch_linkable_object.v1"
        assert obj["transport"] == "rocket_rocc"
        assert obj["issue_schedule"] == row["schedule"]
        assert obj["profile_sha256"] == row["profile_sha256"]
        assert obj["source_reference_sha256"] == index["input_sha256"]["c1_bf16"]
        assert obj["input_sha256"] == {
            key: index["input_sha256"][key] for key in FULL_INPUTS}
        assert (obj["allocated_data_section_bytes"], obj["embedded_operand_bytes"],
                obj["embedded_golden_bytes"]) == (0, 0, 0)
        assert {entry["slot"] for entry in obj["buffer_abi"]} == (
            set(FULL_INPUTS) | set(FULL_OUTPUTS))
        assert row["object_manifest_sha256"] == _sha(
            (case / "object_manifest.json").read_bytes())
        assert row["compile_manifest_sha256"] == _sha(
            (case / "compile_manifest.json").read_bytes())
        assert row["spike_qualification_sha256"] == _sha(
            (case / "spike_qualification.json").read_bytes())
        assert row["object_sha256"] == row["baseline_object_sha256"] == (
            obj["object_sha256"]) == _sha(_unpack(case / "mx_issue.o.gz"))
        assert row["object_sha256"] == _sha((old / "mx_issue.o").read_bytes())
        assert obj["issuer_c_sha256"] == _sha(_unpack(case / "mx_issue.c.gz")) == _sha(
            (old / "mx_issue.c").read_bytes())
        assert obj["physical_program_sha256"] == _sha(
            _unpack(case / "physical_program.json.gz"))
        assert physical["commands"] == source_physical["commands"]
        assert row["elf_sha256"] == qualified["elf_sha256"] == _sha(
            _unpack(case / "mx_program.elf.gz"))
        assert row["spike_log_sha256"] == qualified["spike_log_sha256"] == _sha(
            (case / "spike.log").read_bytes())
        assert qualified["status"] == "full_three_site_chain_matched_on_pinned_spike"
        assert qualified["spike_exit_code"] == 0
        assert (qualified["compared_c1_bf16_values"],
                qualified["compared_fp8_codes"],
                qualified["compared_e8m0_scales"]) == (4096, 16384, 512)
        assert "lowered full two-tile MX/VPU: C1 BF16 0, C1 0 codes 0 scales, " \
               "C2 0 codes 0 scales" in (case / "spike.log").read_text()


def test_full_branch_precursor_and_abi_fail_closed():
    resources = {name: _unpack(EVIDENCE / f"{name}.bin.gz")
                 for name in (*FULL_INPUTS, "c1_bf16")}
    profile = load_profile(PROFILE_DIR / "MxE4M3Fp4VpuGemminiRocketConfig.json")
    full = (SOURCE / "connected.mlir").read_text()
    precursor = (SOURCE / "preloaded.mlir").read_text()
    chain = lower_full_chain_pipelined(full, precursor, profile, resources)
    assert {entry["slot"] for entry in _buffer_abi(chain.commands)} == (
        set(FULL_INPUTS) | set(FULL_OUTPUTS))
    with pytest.raises(ValueError, match="differs from captured MM1 binding"):
        lower_full_chain_pipelined(
            full, precursor.replace("immediate_bf16 = 16384",
                                    "immediate_bf16 = 16385", 1),
            profile, resources)
    with pytest.raises(ValueError, match="runtime resources|captured MM1 binding"):
        lower_full_chain_pipelined(
            full, precursor, profile,
            {**resources, "b2_weight": bytes(len(resources["b2_weight"]))})
