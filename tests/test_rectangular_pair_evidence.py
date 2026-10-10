"""Fresh rectangular model2MLIR → typed object → Nicolas Spike evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.resident_pair_graph import INPUTS, lower_connected_fp8_pair
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_rectangular_pair_64x96x64_64x64x96_c0be9b9"
NARROW = ROOT / "docs/evidence/nicolas_rectangular_pair_64x96x64_64x32x96_ea746a2"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"


def _read(name: str) -> bytes:
    data = (EVIDENCE / name).read_bytes()
    return gzip.decompress(data) if name.endswith(".gz") else data


def test_rectangular_pair_fresh_spike_and_object_provenance() -> None:
    index = json.loads(_read("index.json"))
    assert index["schema"] == "mx_gemmini.nicolas_rectangular_pair_fresh_spike.v1"
    assert index["status"] == "source_c1_and_model_c2_matched_on_pinned_spike"
    assert index["first_shape_mnk"] == [64, 96, 64]
    assert index["second_shape_mnk"] == [64, 64, 96]
    assert index["compared"] == {
        "c1_fp8_codes": 6144, "c1_e8m0_scales": 192,
        "c2_fp8_codes": 4096, "c2_e8m0_scales": 128}
    for name, descriptor in index["files"].items():
        data = _read(name)
        assert len(data) == descriptor["bytes"], name
        assert hashlib.sha256(data).hexdigest() == descriptor["sha256"], name
    capture = json.loads(_read("capture/receipt.json"))
    binding = json.loads(_read("bound/binding_manifest.json"))
    obj = json.loads(_read("object/object_manifest.json"))
    spike = json.loads(_read("spike/qualification_manifest.json"))
    assert capture["model2mlir_revision"] == index["model2mlir_revision"]
    assert [site["shape"] for site in capture["sites"]] == [
        [64, 96, 64], [64, 64, 96]]
    assert binding["reference_kind"] == spike["reference_kind"] == (
        "unchanged_nicolas_c1_and_pinned_model_derived_c2")
    assert binding["model_sha256"] == spike["model_sha256"] == index["model_sha256"]
    assert binding["bound_mlir_sha256"] == obj["bound_mlir_sha256"] == (
        spike["bound_mlir_sha256"])
    assert obj["first_shape_mnk"] == [64, 96, 64]
    assert obj["shape_mnk"] == [64, 64, 96]
    assert (obj["allocated_data_section_bytes"], obj["embedded_operand_bytes"],
            obj["embedded_golden_bytes"]) == (0, 0, 0)
    assert spike["object_sha256"] == obj["object_sha256"]
    assert spike["spike_exit_code"] == 0
    assert (b"lowered rectangular 64x96x64 -> 64x64x96: "
            b"C1 0 codes 0 scales; C2 0 codes 0 scales") in _read("spike/spike.log")
    assert b"C2 3992 codes 34 scales" in _read("diagnostic/spike.log")


def test_rectangular_pair_relowers_its_typed_edges_and_rejects_changed_payload() -> None:
    mlir = _read("bound/connected.mlir.gz").decode()
    resources = {slot: _read(f"bound/{slot}.bin.gz") for slot in INPUTS}
    pair = lower_connected_fp8_pair(
        mlir, load_profile(PROFILE), resources,
        buffers={name: name for name in INPUTS}, c1_scales="c1_scales",
        c1_tiled_observed="c1_tiled_observed", c2_tiled="c2_tiled")
    plan = pair.plan
    assert (plan.m, plan.n, plan.k, plan.first_k) == (64, 64, 96, 64)
    assert (plan.b1_row, plan.b_row, plan.c1_rows, plan.c_rows) == (
        16000, 16000, 384, 256)
    names = tuple(sorted({operand.buffer for command in pair.commands
                          if isinstance(command, Command)
                          for operand in (command.rs1, command.rs2)
                          if operand.buffer is not None}))
    assert emit_c(pair.commands, transport="rocket_rocc", buffers=names).encode() == (
        _read("object/mx_issue.c.gz"))
    changed = dict(resources)
    changed["b2_weight"] = bytes([changed["b2_weight"][0] ^ 1]) + changed["b2_weight"][1:]
    with pytest.raises(ValueError, match="runtime payload digest"):
        lower_connected_fp8_pair(
            mlir, load_profile(PROFILE), changed,
            buffers={name: name for name in INPUTS}, c1_scales="c1_scales",
            c1_tiled_observed="c1_tiled_observed", c2_tiled="c2_tiled")


def test_narrow_rectangular_pair_fresh_spike_and_distinct_weight_footprints() -> None:
    def read(name: str) -> bytes:
        data = (NARROW / name).read_bytes()
        return gzip.decompress(data) if name.endswith(".gz") else data

    index = json.loads(read("index.json"))
    assert index["schema"] == "mx_gemmini.nicolas_rectangular_pair_fresh_spike.v1"
    assert index["first_shape_mnk"] == [64, 96, 64]
    assert index["second_shape_mnk"] == [64, 32, 96]
    assert index["compared"] == {
        "c1_fp8_codes": 6144, "c1_e8m0_scales": 192,
        "c2_fp8_codes": 2048, "c2_e8m0_scales": 64}
    for name, descriptor in index["files"].items():
        data = read(name)
        assert len(data) == descriptor["bytes"], name
        assert hashlib.sha256(data).hexdigest() == descriptor["sha256"], name
    capture = json.loads(read("capture/receipt.json"))
    binding = json.loads(read("bound/binding_manifest.json"))
    obj = json.loads(read("object/object_manifest.json"))
    spike = json.loads(read("spike/qualification_manifest.json"))
    assert [site["shape"] for site in capture["sites"]] == [
        [64, 96, 64], [64, 32, 96]]
    assert binding["bound_mlir_sha256"] == obj["bound_mlir_sha256"] == (
        spike["bound_mlir_sha256"])
    assert binding["model_sha256"] == spike["model_sha256"] == index["model_sha256"]
    assert spike["object_sha256"] == obj["object_sha256"]
    assert spike["spike_exit_code"] == 0
    assert (obj["allocated_data_section_bytes"], obj["embedded_operand_bytes"],
            obj["embedded_golden_bytes"]) == (0, 0, 0)
    abi = {entry["slot"]: entry["minimum_bytes"] for entry in obj["buffer_abi"]}
    assert (abi["b1_weight"], abi["b2_weight"]) == (6144, 3072)
    assert (abi["c1_tiled_observed"], abi["c2_tiled"]) == (6144, 2048)
    assert (b"lowered rectangular 64x96x64 -> 64x32x96: "
            b"C1 0 codes 0 scales; C2 0 codes 0 scales") in read("spike/spike.log")

    mlir = read("bound/connected.mlir.gz").decode()
    resources = {slot: read(f"bound/{slot}.bin.gz") for slot in INPUTS}
    pair = lower_connected_fp8_pair(
        mlir, load_profile(PROFILE), resources,
        buffers={name: name for name in INPUTS}, c1_scales="c1_scales",
        c1_tiled_observed="c1_tiled_observed", c2_tiled="c2_tiled")
    assert (pair.plan.b1_row, pair.plan.b_row, pair.plan.c1_rows,
            pair.plan.c_rows) == (16000, 16192, 384, 128)
    names = tuple(sorted({operand.buffer for command in pair.commands
                          if isinstance(command, Command)
                          for operand in (command.rs1, command.rs2)
                          if operand.buffer is not None}))
    assert emit_c(pair.commands, transport="rocket_rocc", buffers=names).encode() == (
        read("object/mx_issue.c.gz"))
