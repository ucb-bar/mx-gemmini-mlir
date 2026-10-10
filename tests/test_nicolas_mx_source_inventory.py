"""Check source provenance for the pinned Nicolas MX Makefile roster."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


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
    assert inventory["with_direct_hash_reference"] == with_refs == 95
    assert inventory["without_direct_hash_reference"] == 171 - with_refs == 76
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
    assert any("nicolas_plain_fp4_typed_object_97b0913" in ref["path"] for ref in
               by_name["matmul_tiled_fp4_64x64"]["evidence_references"])
    assert any("nicolas_fp4_requant_typed_object_5657fb6" in ref["path"] for ref in
               by_name["matmul_tiled_fp4_64x64_requant"]["evidence_references"])
    assert any("nicolas_plain_fp6_typed_object_95fc6d5" in ref["path"] for ref in
               by_name["matmul_tiled_fp6_128x128x512"]["evidence_references"])
    assert any("nicolas_fp6_requant_typed_object_b25fa48" in ref["path"] for ref in
               by_name["matmul_tiled_fp6_128x128x512_requant"]["evidence_references"])
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
