"""Check source-bound 128³ resident MM2 lowering and direct Spike parity."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.resident_lowering import (lower_single_resident_contract,
                                                  validate_resident_contract)
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_resident_mm2_128_266c593"
FRONTEND = ROOT / "docs/evidence/nicolas_plain_chain_128_model2mlir_e9ded36"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(name: str) -> bytes:
    return gzip.decompress((EVIDENCE / f"{name}.gz").read_bytes())


def test_archived_typed_mm2_and_spike_result_match_source() -> None:
    index = json.loads((EVIDENCE / "index.json").read_text())
    first = json.loads((EVIDENCE / "artifact_manifest.json").read_text())
    reproduced = json.loads((EVIDENCE / "reproduction_manifest.json").read_text())
    profile = load_profile(PROFILE)
    assert index["schema"] == "mx_gemmini.nicolas_resident_mm2_128_archive.v1"
    assert index["profile_sha256"] == profile_sha256(profile)
    assert index["rtl_revision"] == "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
    assert profile["resources"]["requantizer"] is True
    assert profile["resources"]["spad_requant"] is False
    assert index["compared_fp8_codes"] == 16384
    assert index["compared_e8m0_scales"] == 512
    assert first["status"] == reproduced["status"] == (
        "source_resident_mm2_matched_on_pinned_spike")
    for key in ("source_sha256", "header_sha256", "profile_sha256",
                "bound_mlir_sha256", "elf_sha256", "extension_sha256",
                "spike_log_sha256", "files_sha256", "object_sha256"):
        assert first[key] == reproduced[key], key
    assert first["spike_exit_code"] == reproduced["spike_exit_code"] == 0
    for name, digest in index["files_sha256"].items():
        assert _sha(_read(name)) == digest
    assert _sha(_read("resident_mm2.mlir")) == first["bound_mlir_sha256"]
    assert _sha(_read("mx_program.elf")) == first["elf_sha256"]
    assert _sha(_read("spike.log")) == first["spike_log_sha256"]
    assert (b"lowered resident MM2 128x128: 0 FP8 code mismatches, "
            b"0 E8M0 scale mismatches") in _read("spike.log")


def test_latest_model2mlir_captures_both_plain_chain_sites() -> None:
    index = json.loads((FRONTEND / "index.json").read_text())
    first = json.loads((FRONTEND / "receipt.json").read_text())
    second = json.loads((FRONTEND / "reproduction_receipt.json").read_text())
    assert first == second
    assert first["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert first["matrix_dim"] == 128
    assert first["profile_sha256"] == index["profile_sha256"]
    assert first["source_sha256"] == json.loads(
        (EVIDENCE / "index.json").read_text())["source_sha256"]
    assert first["opaque_calls"] == {}
    assert [(site["site_id"], site["status"], site["shape"])
            for site in first["sites"]] == [
                ("functional:matmul", "quantized", [128, 128, 128]),
                ("functional:matmul_1", "quantized", [128, 128, 128]),
            ]
    for name, digest in index["files_sha256"].items():
        assert _sha(gzip.decompress((FRONTEND / f"{name}.gz").read_bytes())) == digest
    bound = gzip.decompress((FRONTEND / "nicolas_chain.profile_bound.mlir.gz").read_bytes())
    assert bound.count(b'"mx_gemmini.contract"') == 2


def test_typed_mm2_uses_resident_c1_and_source_placement() -> None:
    profile = load_profile(PROFILE)
    mlir = _read("resident_mm2.mlir").decode()
    commands = lower_single_resident_contract(mlir, profile)
    transfers = [item for item in commands if isinstance(item, Command) and
                 item.funct == 2]
    assert len(transfers) == 64
    assert all(item.rs1.buffer == "b2_weight" for item in transfers)
    assert transfers[0].rs2.immediate & 0x3fff == 15360
    assert transfers[-1].rs2.immediate & 0x3fff == 16368
    execute = next(item for item in commands if isinstance(item, Command) and
                   item.funct == 24)
    assert (execute.rs1.immediate, execute.rs2.immediate) == (2048, 16384)
    compute = next(item for item in commands if isinstance(item, Command) and
                   item.funct == 8)
    assert compute.rs2.immediate >> 32 == 4096
    assert (b'"mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s)'
            in _read("resident_mm2.mlir"))


def test_wrong_lifetime_or_ssa_handoff_fails_closed() -> None:
    profile = load_profile(PROFILE)
    mlir = _read("resident_mm2.mlir").decode()
    with pytest.raises(ValueError, match="scratchpad tile placement"):
        lower_single_resident_contract(
            mlir.replace("output_row = 4096 : i32", "output_row = 15300 : i32"),
            profile)
    with pytest.raises(ValueError, match="SSA tensor edges"):
        lower_single_resident_contract(
            mlir.replace('"mx_gemmini.resident_contract"(%c1, %c1s, %b2, %b2s)',
                         '"mx_gemmini.resident_contract"(%b2, %c1s, %c1, %b2s)'),
            profile)
    attrs = {"activation_row": 2048, "weight_row": 15360, "output_row": 4096,
             "m": 128, "n": 128, "k": 128,
             "activation_format": "fp8_e4m3", "weight_format": "fp8_e4m3",
             "output_format": "fp8_e4m3", "weight_buffer": "b2_weight",
             "weight_scales_buffer": "b2_scales", "output_scales_buffer": "c2_scales"}
    profile["resources"]["requantizer"] = False
    with pytest.raises(ValueError, match="requantizer profile"):
        validate_resident_contract(profile, attrs)
    profile = load_profile(PROFILE)
    with pytest.raises(ValueError, match="SPAD_REQUANT profile"):
        validate_resident_contract(profile, attrs | {"m": 64, "n": 64, "k": 64})
    vpu_profile = load_profile(
        ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json")
    with pytest.raises(ValueError, match="plain MX profile"):
        validate_resident_contract(vpu_profile, attrs)
