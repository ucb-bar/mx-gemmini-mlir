"""Audit the reusable four-tile FP4 object's direct row-major Spike output."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.quant_reference import exact_bf16_x2
from mx_gemmini_support.source_payload import load_bundle
from tools.qualify_runtime_fp4_tilewise_object import derive_second_payload


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/fp4_row_major_runtime_object_266c593"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_data_free_object_rebinds_two_fp4_payloads_without_host_untiling():
    archive = json.loads((EVIDENCE / "archive_manifest.json").read_text())
    assert archive["schema"] == "mx_gemmini.runtime_fp4_row_major_archive.v1"
    for name, digest in archive["files_sha256"].items():
        assert _sha((EVIDENCE / name).read_bytes()) == digest
    index = json.loads((EVIDENCE / "index.json").read_text())
    assert index == json.loads((EVIDENCE / "index_repro.json").read_text())
    assert index["schema"] == "mx_gemmini.runtime_fp4_row_major_object_spike.v1"
    assert index["status"] == "two_runtime_fp4_row_major_payloads_matched_on_pinned_spike"
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert index["spike_exit_code"] == 0
    assert index["compared_bf16_outputs"] == 2 * 256 * 256
    assert index["permutation"] == "swap_M_128_rows_and_N_128_columns_v1"
    assert index["qualifier_sha256"] == _sha((
        ROOT / "tools/qualify_runtime_fp4_row_major_object.py").read_bytes())
    assert index["spike_log_sha256"] == _sha((EVIDENCE / "spike.log").read_bytes())
    assert b"runtime FP4 row-major: 0/131072 BF16 mismatches" in (
        EVIDENCE / "spike.log").read_bytes()
    elf = gzip.decompress((EVIDENCE / "mx_runtime_fp4_row_major.elf.gz").read_bytes())
    assert _sha(elf) == index["elf_sha256"]
    assert elf[:4] == b"\x7fELF" and int.from_bytes(elf[18:20], "little") == 243

    object_dir = ROOT / archive["object_evidence"]
    obj = json.loads((object_dir / "object_manifest.json").read_text())
    assert index["object_sha256"] == obj["object_sha256"] == _sha((
        object_dir / "mx_issue.o").read_bytes())
    assert index["object_manifest_sha256"] == _sha((
        object_dir / "object_manifest.json").read_bytes())
    assert obj["allocated_data_section_bytes"] == 0
    assert obj["embedded_operand_bytes"] == obj["embedded_golden_bytes"] == 0
    assert obj["buffer_abi"][2]["layout"] == "row_major_bf16"
    assert obj["buffer_abi"][2]["minimum_bytes"] == 256 * 256 * 2
    assert index["source_bundle_manifest_sha256"] == _sha((
        object_dir / "bundle_manifest.json").read_bytes())

    _, first = load_bundle(ROOT / archive["source_bundle"])
    second = derive_second_payload(first)
    assert [case["name"] for case in index["payloads"]] == ["first", "second"]
    for case, resource in zip(index["payloads"], (first, second)):
        for name in ("activation", "activation_scales", "weight", "weight_scales",
                     "golden_bf16"):
            assert case[f"{name}_sha256"] == _sha(resource[name])
        assert case["expected_x2_bf16_sha256"] == _sha(
            exact_bf16_x2(resource["golden_bf16"]))
    assert all(first[name] != second[name] for name in (
        "activation", "activation_scales", "weight", "weight_scales", "golden_bf16"))

    driver = (EVIDENCE / "mx_runtime_driver.c").read_text()
    assert driver.count("mx_issue(") == 2
    assert "first_after_second" in driver
    assert "got[i] != want[i]" in driver
    assert "tile *" not in driver
    assembly = (EVIDENCE / "mx_runtime_data.S").read_text()
    assert assembly.count(".incbin") == 10
