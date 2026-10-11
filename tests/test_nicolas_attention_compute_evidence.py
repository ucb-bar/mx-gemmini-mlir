"""Archived generated attention compute must retain its source parity proof."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_attn_vpu_generated_compute_e9ded36_0344bcb"
SOURCE = ROOT / "docs/evidence/nicolas_attn_vpu_source_baseline_e9ded36_0344bcb"
VECTOR = ROOT / "docs/evidence/nicolas_attn_vpu_generated_middle_e9ded36_0344bcb"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generated_attention_compute_matches_source_spike_oracle() -> None:
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    source = json.loads((SOURCE / "receipt.json").read_text())
    vector = json.loads((VECTOR / "receipt.json").read_text())
    compiled = json.loads((ARCHIVE / "vector_compile_manifest.json").read_text())
    obj = json.loads((ARCHIVE / "vector_object_manifest.json").read_text())
    assert receipt["stage_hashes"] == source["source_stage_hashes"]
    assert receipt["source_sha256"] == source["npu_source_sha256"]
    assert receipt["frontend_mlir_sha256"] == source["frontend_mlir_sha256"]
    assert receipt["header_sha256"] == source["generated_header_sha256"]
    assert receipt["profile_sha256"] == source["profile_sha256"]
    assert receipt["vector_object_sha256"] == vector["object_sha256"]
    assert receipt["vector_object_sha256"] == compiled["object_sha256"] == obj["object_sha256"]
    assert compiled["lowering_family"] == "attention_vector"
    assert receipt["driver_sha256"] == _sha(ARCHIVE / "attn_vpu_generated_compute.c")
    assert receipt["elf_sha256"] == _sha(ARCHIVE / "attn_vpu_generated_compute.elf")
    assert receipt["spike_log_sha256"] == _sha(ARCHIVE / "spike.log")
    assert receipt["spike_reproduced_runs"] == 2
    for stage in ("qk", "pv"):
        assert receipt["matrix_object_sha256"][stage] == _sha(ARCHIVE / f"{stage}_issue.o")
        assert f"void mx_issue_{stage}(" in (ARCHIVE / f"{stage}_issue.c").read_text()
    driver = (ARCHIVE / "attn_vpu_generated_compute.c").read_text()
    assert "static void mm(" not in driver
    assert "gemmini_vpu_" not in driver and "gemmini_spad_requant(" not in driver
    assert "mvin_A((const uint8_t *)Q_IN" in driver
    assert "gemmini_mx_load_scales((uint64_t)Q_SCALES" in driver
    log = (ARCHIVE / "spike.log").read_text()
    assert "check S vs mesh golden: 0/512 4-element words differ" in log
    assert "check hashes vs Spike: match (0 of 4 differ)" in log
    assert "attn_vpu PASSED" in log
