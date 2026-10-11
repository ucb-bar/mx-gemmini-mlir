"""The one-object attention replay must preserve its source and Spike proof."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_attn_vpu_single_object_e9ded36_0344bcb"
SOURCE = ROOT / "docs/evidence/nicolas_attn_vpu_source_baseline_e9ded36_0344bcb"
VECTOR = ROOT / "docs/evidence/nicolas_attn_vpu_generated_middle_e9ded36_0344bcb"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_one_object_matches_pinned_attention_source_on_spike() -> None:
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    source = json.loads((SOURCE / "receipt.json").read_text())
    vector = json.loads((VECTOR / "receipt.json").read_text())
    program = json.loads((ARCHIVE / "physical_program.json").read_text())
    assert receipt["stage_hashes"] == source["source_stage_hashes"]
    assert receipt["source_sha256"] == source["npu_source_sha256"]
    assert receipt["frontend_mlir_sha256"] == source["frontend_mlir_sha256"]
    assert receipt["header_sha256"] == source["generated_header_sha256"]
    assert receipt["vector_object_sha256"] == vector["object_sha256"]
    assert receipt["single_object_sha256"] == _sha(ARCHIVE / "mx_issue.o")
    assert receipt["single_physical_program_sha256"] == _sha(ARCHIVE / "physical_program.json")
    assert receipt["driver_sha256"] == _sha(ARCHIVE / "attn_vpu_single_object.c")
    assert receipt["elf_sha256"] == _sha(ARCHIVE / "attn_vpu_single_object.elf")
    assert receipt["spike_log_sha256"] == _sha(ARCHIVE / "spike.log")
    assert receipt["spike_reproduced_runs"] == 2
    assert len(program["commands"]) == 127
    assert sum(item.get("funct") == 33 for item in program["commands"]) == 7
    assert sum(item.get("funct") == 34 for item in program["commands"]) == 1
    assert sum(item.get("kind") == "fence" for item in program["commands"]) == 1
    issuer = (ARCHIVE / "mx_issue.c").read_text()
    assert issuer.count(".insn r 0x7b") == 126
    driver = (ARCHIVE / "attn_vpu_single_object.c").read_text()
    assert driver.count("  mx_issue(Q_SCALES, KT_SCALES, V_SCALES") == 1
    assert "gemmini_flush(0);" in driver
    assert "gemmini_fence();" in driver
    log = (ARCHIVE / "spike.log").read_text()
    assert "check S vs mesh golden: 0/512 4-element words differ" in log
    assert "check hashes vs Spike: match (0 of 4 differ)" in log
    assert "attn_vpu PASSED" in log
