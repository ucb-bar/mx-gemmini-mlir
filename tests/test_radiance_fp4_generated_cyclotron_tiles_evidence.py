"""Audit executed FP4 Muon source tiles separately from the raw 256x256 driver."""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.quant_reference import exact_bf16_x2
from mx_gemmini_support.source_payload import load_bundle
from tools.qualify_radiance_fp4_derived_cyclotron_tiles import (
    CYCLOTRON_BINARY_SHA256, CYCLOTRON_MODEL_SHA256,
    CYCLOTRON_REVISION, EVIDENCE, MX_SOFTWARE_REVISION,
    SOURCE_FILES, SOURCE_REVISION, _tile_header)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/radiance_fp4_generated_cyclotron_tiles_266c593"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archived(name: str) -> bytes:
    return gzip.decompress((ARCHIVE / f"{name}.gz").read_bytes())


def _stable(receipt: dict) -> dict:
    result = deepcopy(receipt)
    for case in result["cases"]:
        case.pop("build_log_sha256")
        case.pop("cyclotron_log_sha256")
    result["raw_driver"].pop("build_log_sha256")
    result["raw_driver"].pop("cyclotron_log_sha256")
    return result


def test_four_executed_fp4_source_tiles_and_raw_driver_limit():
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    repro = json.loads((ARCHIVE / "receipt_repro.json").read_text())
    assert _stable(receipt) == _stable(repro)
    assert receipt["schema"] == "mx_gemmini.radiance_generated_fp4_four_source_tiles_cyclotron.v1"
    assert receipt["status"] == "four_derived_muon_mx_tiles_match_compiler_vpu_reference"
    assert "not the unmodified 256x256 Radiance driver" in receipt["scope"]
    assert "RTL, or FPGA" in receipt["scope"]
    assert receipt["source_revision"] == SOURCE_REVISION
    assert receipt["mx_software_revision"] == MX_SOFTWARE_REVISION
    assert receipt["cyclotron_revision"] == CYCLOTRON_REVISION
    assert receipt["cyclotron_model_sha256"] == CYCLOTRON_MODEL_SHA256
    assert receipt["cyclotron_binary_sha256"] == CYCLOTRON_BINARY_SHA256
    assert receipt["source_files_sha256"] == SOURCE_FILES
    assert receipt["compiler_revision"] == "44d4cc996eb432b327755f6d0273d432c70ce995"

    manifest, resources = load_bundle(EVIDENCE / "bundle")
    compiler = json.loads((EVIDENCE / "build/artifact_manifest.json").read_text())
    assert receipt["source_payload_manifest_sha256"] == _sha(
        (EVIDENCE / "bundle/manifest.json").read_bytes())
    assert receipt["source_header_sha256"] == manifest["source_header_sha256"]
    assert receipt["source_golden_bf16_sha256"] == _sha(resources["golden_bf16"])
    assert receipt["compared_bf16"] == 65536
    assembled = bytearray(256 * 256 * 2)
    assert len(receipt["cases"]) == 4
    for case in receipt["cases"]:
        tile_m, tile_n = case["tile"]
        name = f"tile_{tile_m}{tile_n}"
        header, golden = _tile_header(resources, tile_m, tile_n)
        driver = _archived(f"kernels/gemm_mxgemmini/{name}.cpp")
        source_elf = _archived(f"kernels/gemm_mxgemmini/{name}.radiance.elf")
        actual = _archived(f"{name}.bf16.bin")
        build_log = _archived(f"{name}.build.log")
        sim_log = _archived(f"{name}.cyclotron.log")
        assert _archived(f"kernels/gemm_mxgemmini/{name}.h") == header.encode()
        assert b"C_out_bf16" not in header.encode()
        assert case["header_sha256"] == _sha(header.encode())
        assert case["driver_sha256"] == _sha(driver)
        assert f'#include "{name}.h"'.encode() in driver
        assert b"static const uint8_t *A_in = &A_in_hw[0][0];" in driver
        assert source_elf[:4] == b"\x7fELF" and source_elf[4] == 1
        assert int.from_bytes(source_elf[18:20], "little") == 243
        assert case["elf_sha256"] == _sha(source_elf)
        assert case["build_log_sha256"] == _sha(build_log)
        assert case["cyclotron_log_sha256"] == _sha(sim_log)
        assert b"simulation finished" in sim_log
        assert actual == golden and case["matched_bf16"] == 16384
        assert case["output_sha256"] == _sha(actual)
        for row in range(128):
            dst = ((tile_m * 128 + row) * 256 + tile_n * 128) * 2
            src = row * 128 * 2
            assembled[dst:dst + 256] = actual[src:src + 256]
    full = bytes(assembled)
    scaled = exact_bf16_x2(full)
    assert full == resources["golden_bf16"] == _archived("assembled_bf16.bin")
    assert scaled == _archived("assembled_x2_bf16.bin")
    assert receipt["assembled_bf16_sha256"] == _sha(full)
    assert receipt["assembled_x2_bf16_sha256"] == _sha(scaled)
    assert receipt["compiler_derived_x2_bf16_sha256"] == _sha(scaled)
    assert compiler["files_sha256"]["derived_expected_bf16.bin"] == _sha(scaled)

    raw = receipt["raw_driver"]
    assert raw["status"] == "builds_but_does_not_compute_four_output_tiles"
    assert raw["driver_sha256"] == manifest["source_driver_sha256"]
    assert raw["header_sha256"] == manifest["source_header_sha256"]
    assert raw["elf_sha256"] == _sha(_archived(
        "kernels/gemm_mxgemmini/raw_256x256.radiance.elf"))
    assert raw["build_log_sha256"] == _sha(_archived("raw_256x256.build.log"))
    assert raw["cyclotron_log_sha256"] == _sha(_archived("raw_256x256.cyclotron.log"))
    assert b"simulation finished" in _archived("raw_256x256.cyclotron.log")
    raw_output = _archived("raw_256x256.bf16.bin")
    assert raw["output_sha256"] == _sha(raw_output)
    assert raw["mismatched_bf16"] == 65490
    assert sum(raw_output[i:i + 2] != full[i:i + 2]
               for i in range(0, len(full), 2)) == raw["mismatched_bf16"]


def test_fp4_tile_source_adapter_rejects_invalid_coordinates():
    _, resources = load_bundle(EVIDENCE / "bundle")
    with pytest.raises(ValueError):
        _tile_header(resources, 2, 0)
