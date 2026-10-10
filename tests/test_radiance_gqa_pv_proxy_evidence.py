"""Audit the typed PV proxy, source bytes, and Nicolas Spike result."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from mx_gemmini_support.source_payload import (
    ATTENTION_PV_PROXY_ORIGIN, load_bundle, validate_attention_pv_proxy)
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.verify_profile_ir import verify_ir


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_gqa_pv_proxy_80f84ca"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pv_proxy_capture_binding_and_full_spike_output() -> None:
    index = json.loads((EVIDENCE / "index.json").read_text())
    assert index == json.loads((EVIDENCE / "index_repro.json").read_text())
    assert index["schema"] == "mx_gemmini.gqa_pv_proxy_spike.v1"
    assert index["status"] == "source_derived_pv_proxy_matched_on_pinned_spike"
    assert "no Muon execution" in index["scope"]
    assert index["spike_exit_code"] == 0
    assert index["compared_bf16_outputs"] == 4096
    manifest, resources = load_bundle(EVIDENCE / "bundle")
    assert manifest["origin"] == ATTENTION_PV_PROXY_ORIGIN
    assert manifest["source_derivation"]["exp_policy"] == (
        "torch_exp_bf16_proxy_for_mu_fexp")
    assert manifest["source_derivation"]["qk_golden_bf16_sha256"] == (
        index["qk_golden_bf16_sha256"])
    assert resources["golden_bf16"] == (EVIDENCE / "bundle/golden_bf16.bin").read_bytes()
    for key, filename in (("frontend_mlir_sha256", "model2mlir.mlir"),
                          ("payload_bound_mlir_sha256", "payload_bound.mlir"),
                          ("bundle_manifest_sha256", "bundle/manifest.json"),
                          ("physical_program_sha256", "build/physical_program.json"),
                          ("issuer_c_sha256", "build/mx_issue.c"),
                          ("elf_sha256", "build/mx_program.elf"),
                          ("spike_log_sha256", "build/spike.log"),
                          ("object_sha256", "linkable_object/mx_issue.o")):
        assert index[key] == _sha(EVIDENCE / filename)
    for name in ("activation", "activation_scales", "weight", "weight_scales",
                 "golden_bf16"):
        assert index[f"{name}_sha256"] == _sha(EVIDENCE / f"bundle/{name}.bin")
    assert "0 BF16 mismatches" in (EVIDENCE / "build/spike.log").read_text()
    object_receipt = json.loads((EVIDENCE / "linkable_object/object_manifest.json").read_text())
    assert object_receipt["embedded_operand_bytes"] == 0
    assert object_receipt["allocated_data_section_bytes"] == 0
    assert object_receipt["object_sha256"] == index["object_sha256"]
    qk_object = ROOT / "docs/evidence/radiance_gqa_runtime_object_80f84ca/head0_object/mx_issue.o"
    assert _sha(qk_object) == index["object_sha256"]
    profile = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593" /
                           "MxE4M3Fp4VpuGemminiRocketConfig.json")
    report = verify_ir((EVIDENCE / "payload_bound.mlir").read_text(), profile)
    assert report["contracts"] == 1
    assert report["source_resources"] == 4


def test_pv_proxy_rejects_policy_or_operand_relabeling() -> None:
    manifest, _ = load_bundle(EVIDENCE / "bundle")
    changed = copy.deepcopy(manifest)
    changed["source_derivation"]["exp_policy"] = "actual_muon_execution"
    with pytest.raises(ValueError, match="explicit source derivation"):
        validate_attention_pv_proxy(changed)
    changed = copy.deepcopy(manifest)
    changed["source_derivation"]["source_arrays_sha256"]["activation"] = "0" * 64
    with pytest.raises(ValueError, match="operand hashes"):
        validate_attention_pv_proxy(changed)
