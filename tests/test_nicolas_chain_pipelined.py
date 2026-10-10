"""Source and frontend gates for the two-tile MX+VPU chain."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from mx_gemmini_support.chain_pipelined_source import audit_chain_pipelined
from mx_gemmini_support.chain_pipelined_graph import (
    lower_chain_pipelined, render_chain_pipelined,
    validate_original_branch_trace)
from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
RTL = Path(os.environ.get("MX_GEMMINI_RTL_ROOT", "/nonexistent"))
EVIDENCE = ROOT / "docs/evidence/nicolas_chain_pipelined_266c593"
COMPILED = ROOT / "docs/evidence/nicolas_chain_pipelined_compiled_266c593"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_graph_has_shared_c1_and_b2_with_two_scalars():
    graph = json.loads((EVIDENCE / "original_graph.json").read_text())
    validate_original_branch_trace({"graphs": {"original": graph}})
    changed = json.loads(json.dumps(graph))
    changed["nodes"][6]["args"][1] = 3.0
    with pytest.raises(ValueError, match="lost the shared two-branch chain"):
        validate_original_branch_trace({"graphs": {"original": changed}})
    changed = json.loads(json.dumps(graph))
    changed["nodes"][7]["args"][1]["node_id"] = changed["nodes"][1]["id"]
    with pytest.raises(ValueError, match="lost the shared two-branch chain"):
        validate_original_branch_trace({"graphs": {"original": changed}})


def test_archived_capture_and_handwritten_spike_are_distinct_gates():
    capture = json.loads((EVIDENCE / "receipt.json").read_text())
    source = json.loads((EVIDENCE / "source_spike_receipt.json").read_text())
    assert capture["schema"] == "mx_gemmini.nicolas_chain_pipelined_model2mlir_capture.v1"
    assert capture["status"] == "three_site_frontend_handoff_only"
    assert not capture["opaque_calls"]
    assert capture["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    for key, name in (("original_graph_sha256", "original_graph.json"),
                      ("bound_mlir_sha256", "chain_pipelined.profile_bound.mlir"),
                      ("manifest_sha256", "quantization_manifest.json")):
        assert capture[key] == _sha(EVIDENCE / name)
    assert source["schema"] == "mx_gemmini.nicolas_chain_pipelined_source_spike.v1"
    assert source["status"] == "four_source_checks_matched_on_pinned_spike"
    assert source["spike_log_sha256"] == _sha(EVIDENCE / "spike.log")
    assert [check["schedule"] for check in source["checks"]] == [
        "warmup", "fenced", "program", "pipelined"]
    assert all(check["clamped_scales_skipped"] == 0 for check in source["checks"])
    assert "chain_pipelined PASSED" in (EVIDENCE / "spike.log").read_text()


def test_source_audit_derives_both_complete_tile_goldens(tmp_path):
    source = RTL / "software/gemmini-rocc-tests/bareMetalC/chain_pipelined.c"
    header = RTL / "software/gemmini-rocc-tests/include/matmul_fp8_64x64_chain.h"
    if not source.is_file() or not header.is_file():
        pytest.skip("requires Nicolas's pinned chain source and header")
    profile = load_profile(PROFILE, rtl_root=RTL)
    resources, facts = audit_chain_pipelined(source, header, profile)
    assert facts["profile_sha256"] == profile_sha256(profile)
    assert facts["tile_factors_bf16"] == [0x4000, 0x4080]
    assert len(resources["c1_bf16"]) == 8192
    assert len(resources["b2_weight"]) == 4096
    assert len(resources["b2_scales"]) == 128
    for tile in (0, 1):
        assert len(resources[f"c1_codes_ref_{tile}"]) == 4096
        assert len(resources[f"c1_scales_ref_{tile}"]) == 128
        assert len(resources[f"c2_codes_ref_{tile}"]) == 4096
        assert len(resources[f"c2_scales_ref_{tile}"]) == 128
    assert resources["c1_codes_ref_0"] == resources["c1_codes_ref_1"]
    assert resources["c2_codes_ref_0"] == resources["c2_codes_ref_1"]
    assert all(b == a + 1 for a, b in zip(resources["c1_scales_ref_0"],
                                             resources["c1_scales_ref_1"]))
    changed = tmp_path / source.name
    changed.write_text(source.read_text().replace(
        "vpu_tile(1);          // runs while tile 0's matmul computes",
        "vpu_tile(0);"))
    with pytest.raises(ValueError, match="issue order changed"):
        audit_chain_pipelined(changed, header, profile)


def test_typed_two_branch_graph_and_shared_b2_physical_commands():
    source = RTL / "software/gemmini-rocc-tests/bareMetalC/chain_pipelined.c"
    header = RTL / "software/gemmini-rocc-tests/include/matmul_fp8_64x64_chain.h"
    if not source.is_file() or not header.is_file():
        pytest.skip("requires Nicolas's pinned chain source and header")
    profile = load_profile(PROFILE, rtl_root=RTL)
    resources, facts = audit_chain_pipelined(source, header, profile)
    frontend = (EVIDENCE / "chain_pipelined.profile_bound.mlir").read_text()
    trace = {"graphs": {"original": json.loads(
        (EVIDENCE / "original_graph.json").read_text())}}
    manifest = json.loads((EVIDENCE / "quantization_manifest.json").read_text())
    bound = render_chain_pipelined(frontend, trace, manifest, profile, resources, facts)
    assert bound == (COMPILED / "connected.mlir").read_text()
    chain = lower_chain_pipelined(bound, profile, resources)
    commands = [item for item in chain.commands if isinstance(item, Command)]
    assert [command.funct for command in commands].count(33) == 2
    assert [command.funct for command in commands].count(34) == 2
    assert [command.funct for command in commands].count(8) == 2
    assert sum(command.rs1.buffer == "b2_weight" for command in commands) == 16
    assert sum(command.rs1.buffer == "b2_scales" for command in commands) == 1
    changed = bound.replace("immediate_bf16 = 16512", "immediate_bf16 = 16384")
    with pytest.raises(ValueError, match="vector placement differs"):
        lower_chain_pipelined(changed, profile, resources)
    changed = bound.replace('weight_buffer = "b2_weight"',
                            'weight_buffer = "wrong_weight"', 1)
    with pytest.raises(ValueError, match="shared B2 differs"):
        lower_chain_pipelined(changed, profile, resources)


def test_archived_compiler_object_and_spike_full_outputs():
    manifest = json.loads((COMPILED / "object_manifest.json").read_text())
    assert manifest["schema"] == "mx_gemmini.chain_pipelined_linkable_object.v1"
    assert manifest["status"] == "rv64_rocc_two_tile_object_built"
    assert manifest["allocated_data_section_bytes"] == 0
    assert manifest["embedded_operand_bytes"] == 0
    assert manifest["embedded_golden_bytes"] == 0
    assert manifest["bound_mlir_sha256"] == _sha(COMPILED / "connected.mlir")
    assert manifest["physical_program_sha256"] == _sha(COMPILED / "physical_program.json")
    assert manifest["issuer_c_sha256"] == _sha(COMPILED / "mx_issue.c")
    assert manifest["object_sha256"] == _sha(COMPILED / "mx_issue.o")
    assert manifest["spike_qualification"]["status"] == "both_source_tiles_matched_on_pinned_spike"
    assert manifest["spike_qualification"]["compared_fp8_codes"] == 16384
    assert manifest["spike_qualification"]["compared_e8m0_scales"] == 512
    assert manifest["spike_qualification"]["elf_sha256"] == _sha(COMPILED / "mx_program.elf")
    assert manifest["spike_qualification"]["spike_log_sha256"] == _sha(COMPILED / "spike.log")
    assert "C1 0 codes 0 scales, C2 0 codes 0 scales" in (COMPILED / "spike.log").read_text()
