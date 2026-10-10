"""Audit the current model2MLIR capture through Nicolas's executed MX/VPU chain."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_connected_chain_upstream_e9ded36_20261010"
OLD = ROOT / "docs/evidence/compiled_nicolas_connected_chain_266c593.json"
PROFILE = (ROOT / "profiles/gemmini-mx-cleanup-266c593/"
           "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_two_site_capture_executes_connected_chain_on_pinned_spike() -> None:
    capture = json.loads((EVIDENCE / "capture_receipt.json").read_text())
    quant = json.loads((EVIDENCE / "quantization_manifest.json").read_text())
    built = json.loads((EVIDENCE / "artifact_manifest.json").read_text())
    old = json.loads(OLD.read_text())
    physical = json.loads((EVIDENCE / "physical_program.json").read_text())
    source_mlir = gzip.decompress((EVIDENCE / "source.mlir.gz").read_bytes())

    assert capture["status"] == "two_site_frontend_handoff_only"
    assert capture["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert capture["rtl_revision"] == built["rtl_revision"] == (
        "266c593f2cb51d7e3fe83fc0317072b585ac3c52")
    assert capture["compiler_revision"] == built["compiler_revision"] == (
        "321831d9a88a32f775615b84fc56c90d690a0b00")
    assert capture["opaque_calls"] == {}
    assert [(site["site_id"], site["format"], site["status"], site["shape"])
            for site in capture["sites"]] == [
                ("functional:matmul", "mxfp8", "quantized", [64, 64, 64]),
                ("functional:matmul_1", "mxfp8", "quantized", [64, 64, 64])]
    assert quant["sites"] == capture["sites"]
    assert hashlib.sha256(source_mlir).hexdigest() == capture["source_mlir_sha256"]
    for name, field in (("handoff.mlir", "handoff_mlir_sha256"),
                        ("frontend_bound.mlir", "bound_mlir_sha256"),
                        ("quantization_manifest.json", "manifest_sha256")):
        assert _sha(EVIDENCE / name) == capture[field]
    profile = load_profile(PROFILE)
    checked = verify_ir((EVIDENCE / "frontend_bound.mlir").read_text(), profile)
    assert checked["contracts"] == 2

    assert built["status"] == "source_connected_full_chain_matched_on_pinned_spike"
    assert built["spike_exit_code"] == 0
    assert built["compared_bf16_values"] == 4096
    assert built["compared_fp8_codes"] == 8192
    assert built["compared_e8m0_scales"] == 256
    assert built["frontend_capture_receipt_sha256"] == _sha(EVIDENCE / "capture_receipt.json")
    assert built["frontend_bound_mlir_sha256"] == _sha(EVIDENCE / "frontend_bound.mlir")
    assert built["connected_bound_mlir_sha256"] == _sha(EVIDENCE / "connected_bound.mlir")
    assert physical["frontend_mlir_sha256"] == built["frontend_bound_mlir_sha256"]
    assert physical["connected_mlir_sha256"] == built["connected_bound_mlir_sha256"]
    assert physical["command_count"] == 134 and physical["fence_count"] == 11
    assert built["files_sha256"]["physical_program.json"] == _sha(
        EVIDENCE / "physical_program.json")
    assert built["spike_log_sha256"] == _sha(EVIDENCE / "spike.log")
    assert "C1 BF16 0, C1 0 codes 0 scales, C2 0 codes 0 scales mismatches" in (
        EVIDENCE / "spike.log").read_text()

    for field in ("object_sha256", "elf_sha256", "extension_sha256",
                  "spike_log_sha256"):
        assert built[field] == old[field], field
    for name, digest in built["files_sha256"].items():
        if name != "physical_program.json":
            assert digest == old["files_sha256"][name], name
