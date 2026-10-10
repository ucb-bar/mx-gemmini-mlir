"""Check fresh 16- and 64-row model2MLIR → object → Nicolas Spike replays."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.plain_chain_128 import lower_plain_chain
from mx_gemmini_support.resident_pair_graph import INPUTS, lower_connected_fp8_pair
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = {
    16: ROOT / "docs/evidence/nicolas_plain_pair_16x96_derived_f460d91",
    64: ROOT / "docs/evidence/nicolas_plain_pair_64x96_derived_4dd6108",
}
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"


def _read(m: int, name: str) -> bytes:
    path = EVIDENCE[m] / name
    data = path.read_bytes()
    return gzip.decompress(data) if name.endswith(".gz") else data


@pytest.mark.parametrize("m", (16, 64))
def test_fresh_pair_archive_has_complete_numerical_and_object_provenance(m: int) -> None:
    index = json.loads(_read(m, "index.json"))
    assert index["schema"] == f"mx_gemmini.derived_plain_pair_{m}x96_fresh_spike.v1"
    assert index["shape_mnk"] == [m, 96, 96]
    assert index["reference_kind"] == "source_wire_slice_with_pinned_mesh_model_outputs"
    assert index["status"] == "derived_source_connected_object_matched_on_pinned_spike"
    assert index["compared"] == {
        "c1_fp8_codes": m * 96, "c1_e8m0_scales": m * 3,
        "c2_fp8_codes": m * 96, "c2_e8m0_scales": m * 3}
    for name, descriptor in index["files"].items():
        data = _read(m, name)
        assert len(data) == descriptor["bytes"], name
        assert hashlib.sha256(data).hexdigest() == descriptor["sha256"], name

    capture = json.loads(_read(m, "capture/receipt.json"))
    reference = json.loads(_read(m, "reference/artifact_manifest.json"))
    obj = json.loads(_read(m, "object/object_manifest.json"))
    linked = json.loads(_read(m, "linked/qualification_manifest.json"))
    assert capture["model2mlir_revision"] == index["model2mlir_revision"]
    assert capture["matrix_dim"] == 96 and capture["output_rows"] == m
    assert [site["shape"] for site in capture["sites"]] == [
        [m, 96, 96], [m, 96, 96]]
    assert reference["model_sha256"] == linked["model_sha256"] == index["model_sha256"]
    assert reference["spike_exit_code"] == linked["spike_exit_code"] == 0
    assert obj["shape_mnk"] == linked["shape_mnk"] == [m, 96, 96]
    assert (obj["allocated_data_section_bytes"], obj["embedded_operand_bytes"],
            obj["embedded_golden_bytes"]) == (0, 0, 0)
    assert obj["object_sha256"] == linked["object_sha256"]
    assert obj["issuer_c_sha256"] == linked["source_issuer_sha256"]
    assert _read(m, "reference/spike.log") == _read(m, "linked/spike.log")
    assert (f"lowered connected {m}x96: C1 0 codes 0 scales; "
            "C2 0 codes 0 scales").encode() in _read(m, "linked/spike.log")


@pytest.mark.parametrize("m", (16, 64))
def test_derived_pair_still_lowers_from_its_typed_edges_and_checked_bytes(m: int) -> None:
    frontend = _read(m, "capture/nicolas_chain.profile_bound.mlir.gz").decode()
    manifest = json.loads(_read(m, "capture/quantization_manifest.json.gz"))
    bound = _read(m, "reference/connected_chain.mlir.gz").decode()
    receipt = json.loads(_read(m, "reference/artifact_manifest.json"))
    resources = {name: _read(m, f"reference/{name}.bin.gz") for name in INPUTS}
    profile = load_profile(PROFILE)
    commands = lower_plain_chain(
        bound, frontend, manifest, profile, resources,
        source_sha256=receipt["source_sha256"],
        header_sha256=receipt["header_sha256"], width=96)
    names = tuple(sorted({operand.buffer for command in commands
                          if isinstance(command, Command)
                          for operand in (command.rs1, command.rs2)
                          if operand.buffer is not None}))
    assert emit_c(commands, transport="rocket_rocc", buffers=names).encode() == (
        _read(m, "object/mx_issue.c.gz"))
    assert _read(m, "reference/mx_issue.c.gz") == _read(m, "object/mx_issue.c.gz")
    pair = lower_connected_fp8_pair(
        bound, profile, resources, buffers={name: name for name in INPUTS},
        c1_scales="c1_scales", c1_tiled_observed="c1_tiled_observed",
        c2_tiled="c2_tiled")
    assert pair.plan.m_tiles == m // 16
    assert pair.plan.n_tiles == pair.plan.k_tiles == 6
    assert pair.plan.b_row == pair.plan.rows - 576
    changed = dict(resources)
    changed["b2_weight"] = bytes([resources["b2_weight"][0] ^ 1]) + resources["b2_weight"][1:]
    with pytest.raises(ValueError, match="runtime payload digest"):
        lower_connected_fp8_pair(
            bound, profile, changed, buffers={name: name for name in INPUTS},
            c1_scales="c1_scales", c1_tiled_observed="c1_tiled_observed",
            c2_tiled="c2_tiled")
