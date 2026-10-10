"""Check fresh MX+VPU Spike runs against the archived physical programs."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_mx_vpu_80f84ca_upstream_repro_20261010"
BASELINES = {
    "fp8": ROOT / "docs/evidence/radiance_tilewise_vpu_x2_266c593",
    "fp4": ROOT / "docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archived(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(path.with_suffix(path.suffix + ".gz").read_bytes())


@pytest.mark.parametrize("precision", ("fp8", "fp4"))
def test_fresh_pinned_spike_vpu_reproduction(precision: str) -> None:
    latest = EVIDENCE / precision
    baseline = BASELINES[precision]
    index = json.loads((latest / "index.json").read_text())
    old_index = json.loads((baseline / "index.json").read_text())
    manifest_bytes = (latest / "artifact_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    old_manifest = json.loads(_archived(baseline / "build/artifact_manifest.json"))
    spike = (latest / "spike.log").read_bytes()

    assert index["compiler_revision"] == manifest["compiler_revision"] == (
        "d548fd7f3beb2dfbda3465fca9eb79ba381ea93c")
    assert index["compiler_source_closure_sha256"] == (
        manifest["compiler_source_closure_sha256"])
    assert index["source_revision"] == "80f84caedbabc663a7433c1da4455b936cca41f3"
    assert index["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert index["rtl_revision"] == manifest["rtl_revision"] == (
        "266c593f2cb51d7e3fe83fc0317072b585ac3c52")
    assert index["output_tiles"] == index["vpu_commands"] == 4
    assert index["compared_bf16_outputs"] == manifest["compared_bf16_outputs"] == 65536
    assert manifest["status"] == "derived_vpu_golden_matched_on_pinned_spike"
    assert manifest["golden_basis"] == "derived_bf16_x2"
    assert manifest["spike_exit_code"] == 0
    assert b"lowered MX 256x256x256: 0 BF16 mismatches" in spike

    assert _sha(manifest_bytes) == index["files_sha256"]["build/artifact_manifest.json"]
    assert _sha(spike) == index["files_sha256"]["build/spike.log"]
    assert _sha(spike) == manifest["spike_log_sha256"]
    assert index["elf_sha256"] == manifest["elf_sha256"] == old_index["elf_sha256"]
    assert index["extension_sha256"] == manifest["extension_sha256"] == (
        old_index["extension_sha256"])
    if precision == "fp8":
        assert index["source_driver_sha256"] == manifest["source_driver_sha256"]
        assert index["source_header_sha256"] == manifest["source_header_sha256"]
        assert index["source_driver_sha256"] == old_index["source_driver_sha256"]
        assert index["source_header_sha256"] == old_index["source_header_sha256"]
    else:
        derivation = index["source_derivation"]
        assert derivation == old_index["source_derivation"]
        assert derivation["derived_driver_sha256"] == manifest["source_driver_sha256"]
        assert derivation["generated_header_sha256"] == manifest["source_header_sha256"]
    for name, digest in index["files_sha256"].items():
        if name != "build/artifact_manifest.json":
            assert digest == old_index["files_sha256"][name], name
    for key, value in manifest.items():
        if key not in {"build_log_sha256", "compiler_revision",
                       "compiler_source_closure_sha256"}:
            assert value == old_manifest[key], key
    if precision == "fp4":
        assert "no committed Radiance driver or source ELF parity" in index["scope"]
