"""Check source provenance for the pinned Nicolas MX Makefile roster."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.qualify_nicolas_plain_matrix_object import CASES


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/evidence/nicolas_mx_source_inventory_266c593/index.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nicolas_mx_inventory_references_real_source_receipts() -> None:
    inventory = json.loads(INVENTORY.read_text())
    assert inventory["schema"] == "mx_gemmini.nicolas_mx_source_inventory.v1"
    assert inventory["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert inventory["makefile_sha256"] == "fbf15aa6938b7da6b701a628292123ec652c31e7841f692005871e9a13c3c76d"
    assert inventory["programs"] == len(inventory["entries"]) == 171
    assert inventory["family_counts"] == {
        "asymmetric_matrix": 74, "other_tiled_matrix": 88,
        "chain": 2, "vpu": 2, "spad_requant": 2, "other_mx": 3,
    }
    names = [entry["name"] for entry in inventory["entries"]]
    assert len(names) == len(set(names))
    with_refs = sum(bool(entry["evidence_references"])
                    for entry in inventory["entries"])
    assert inventory["with_direct_hash_reference"] == with_refs == 150
    assert inventory["without_direct_hash_reference"] == 171 - with_refs == 21
    by_name = {entry["name"]: entry for entry in inventory["entries"]}
    assert not by_name["matmul_tiled_fp8_128x128_dramloop"]["evidence_references"]
    assert any("nicolas_plain_fp8_typed_object_7d7a660" in ref["path"] for ref in
               by_name["matmul_tiled_fp8_128x128"]["evidence_references"])
    assert any("nicolas_plain_fp8_256_two_wave_object_9250250" in ref["path"] for ref in
               by_name["matmul_tiled_fp8_128x128x256"]["evidence_references"])
    assert any("nicolas_plain_fp8_96x96x64_object_52dcc4d" in ref["path"] for ref in
               by_name["matmul_tiled_fp8_96x96x64"]["evidence_references"])
    assert any("nicolas_fp8_requant_typed_object_fa5ec73" in ref["path"] for ref in
               by_name["matmul_tiled_fp8_128x128_requant"]["evidence_references"])
    assert any("nicolas_fp8_dim32_requant_typed_object_835e5ba" in ref["path"] for ref in
               by_name["matmul_tiled_fp8_64x64_requant_dim32"]["evidence_references"])
    assert any("nicolas_plain_fp4_typed_object_97b0913" in ref["path"] for ref in
               by_name["matmul_tiled_fp4_64x64"]["evidence_references"])
    assert any("nicolas_fp4_requant_typed_object_5657fb6" in ref["path"] for ref in
               by_name["matmul_tiled_fp4_64x64_requant"]["evidence_references"])
    assert any("nicolas_fp4_large_requant_typed_object_03490a5" in ref["path"] for ref in
               by_name["matmul_tiled_fp4_128x128x512_requant"]["evidence_references"])
    assert any("nicolas_fp4_dim32_requant_typed_object_f475d05" in ref["path"] for ref in
               by_name["matmul_tiled_fp4_128x128_requant_dim32"]["evidence_references"])
    assert any("nicolas_plain_fp6_typed_object_95fc6d5" in ref["path"] for ref in
               by_name["matmul_tiled_fp6_128x128x512"]["evidence_references"])
    assert any("nicolas_fp6_requant_typed_object_b25fa48" in ref["path"] for ref in
               by_name["matmul_tiled_fp6_128x128x512_requant"]["evidence_references"])
    for source in (
            "matmul_tiled_fp8_32x32x32",
            "matmul_tiled_fp8_64x256x64_requant_dim32",
            "matmul_tiled_fp8_128x128x256_requant_dim32",
            "matmul_tiled_fp4_128x128x512"):
        assert any("nicolas_direct_matrix_suite_9df3384" in ref["path"] for ref in
                   by_name[source]["evidence_references"])
    assert any("nicolas_asym_public_suite_e6923e8" in ref["path"] for ref in
               by_name["matmul_tiled_asym_e4m3s_fp4_16x32"]["evidence_references"])
    for source in ("matmul_tiled_fp8_64x64", "matmul_tiled_fp8_64x64_requant",
                   "matmul_tiled_fp8_96x32x32"):
        assert any("nicolas_direct_fp8_three_789d192" in ref["path"] for ref in
                   by_name[source]["evidence_references"])
    expanded = json.loads((ROOT / "docs/evidence/nicolas_direct_sixteen_28936e2_266c593/index.json").read_text())
    for key in expanded["selected_cases"]:
        source = CASES[key].source_name.removesuffix(".c")
        assert any("nicolas_direct_sixteen_28936e2" in ref["path"] for ref in
                   by_name[source]["evidence_references"])
    for source in ("matmul_tiled_fp6_e2m3_lut_64x64",
                   "matmul_tiled_fp8_e4m3_lut_64x64",
                   "matmul_tiled_fp8_e5m2_64x64"):
        assert any("nicolas_symmetric_lut_public_4e30dcf" in ref["path"] for ref in
                   by_name[source]["evidence_references"])
    for source in ("matmul_tiled_fp8_e4m3_lut_64x64_dim32",
                   "matmul_tiled_fp8_e4m3_lut_64x64_nonrequant_dim8",
                   "matmul_tiled_fp8_e4m3_lut_128x128_nonrequant_dim8"):
        assert any("nicolas_e4m3_lut_shapes_public_4f316ac" in ref["path"] for ref in
                   by_name[source]["evidence_references"])
    assert any("mx_public_vpu_softmax_3f9af55" in ref["path"] for ref in
               by_name["vpu_softmax"]["evidence_references"])
    for entry in inventory["entries"]:
        assert len(entry["source_sha256"]) == 64
        for ref in entry["evidence_references"]:
            path = ROOT / ref["path"]
            assert path.is_relative_to(ROOT / "docs/evidence")
            assert ref["sha256"] == _sha(path)
            record = json.loads(path.read_text())
            assert record[ref["field"]] == entry["source_sha256"]
            assert ref["schema"] == record.get("schema")
            assert ref["status"] == record.get("status")
