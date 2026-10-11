"""Validate VPU object portability across Nicolas's two Rocket VPU profiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "docs/evidence/nicolas_vpu_public_objects_266c593"
FP4 = ROOT / "docs/evidence/nicolas_vpu_public_fp4_profile_266c593"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rebound_vpu_objects_match_both_profiles_numerically() -> None:
    original = json.loads((ORIGINAL / "index.json").read_text())
    rebound = json.loads((FP4 / "index.json").read_text())
    assert rebound["schema"] == original["schema"]
    assert rebound["profile_name"] == "MxE4M3Fp4VpuGemminiRocketConfig"
    assert rebound["profile_sha256"] != original["profile_sha256"]
    assert rebound["compiler_revision"] == "ba671ec2eeca6a35a5974f7369536d8d23f3db56"
    assert rebound["total_output_bf16_compared"] == 6016
    assert rebound["total_sum_bf16_compared"] == 128
    assert len(rebound["rows"]) == len(original["rows"]) == 14
    assert {row["kind"] for row in rebound["rows"]} == {
        row["kind"] for row in original["rows"]}
    by_original = {row["kind"]: row for row in original["rows"]}
    for row in rebound["rows"]:
        case = FP4 / row["kind"]
        baseline = by_original[row["kind"]]
        manifest = json.loads((case / "object_manifest.json").read_text())
        dispatch = json.loads((case / "compile_manifest.json").read_text())
        assert row["baseline_bound_mlir_sha256"] == baseline["bound_mlir_sha256"]
        assert row["bound_mlir_sha256"] == _sha(case / "rebound.mlir")
        assert row["bound_mlir_sha256"] != baseline["bound_mlir_sha256"]
        assert row["rebound_binding_sha256"] == _sha(case / "rebound_binding.json")
        assert row["spike_log_sha256"] == _sha(case / "spike.log")
        assert row["object_manifest_sha256"] == _sha(case / "object_manifest.json")
        assert row["physical_program_sha256"] == _sha(case / "physical_program.json")
        assert manifest["profile_sha256"] == rebound["profile_sha256"]
        assert dispatch["profile_sha256"] == rebound["profile_sha256"]
        assert dispatch["lowering_family"] == "vpu_elementwise"
        assert len(dispatch["native_verifier_sha256"]) == 64
        assert row["mismatches"] == 0
        for field in ("issuer_c_sha256", "object_sha256", "elf_sha256",
                      "spike_log_sha256", "compared_output_bf16",
                      "compared_sum_bf16"):
            assert row[field] == baseline[field]
        assert row["object_sha256"] == _sha(ORIGINAL / row["kind"] / "mx_issue.o")
        assert row["elf_sha256"] == _sha(ORIGINAL / row["kind"] / "vpu.elf")
