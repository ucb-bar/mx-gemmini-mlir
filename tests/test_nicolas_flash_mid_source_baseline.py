"""Pin the original four-block flash source's output and frontend identity."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_attn_flash_mid_source_baseline_a042643_0344bcb"
MID = ROOT / "docs/evidence/nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_flash_source_oracle_and_reused_clean_frontend() -> None:
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    mid = json.loads((MID / "receipt.json").read_text())
    replay = json.loads((ARCHIVE / "replay_receipt.json").read_text())
    log = (ARCHIVE / "source_spike.log").read_text()
    expect = (ARCHIVE / "attn_flash_mid_expect.h").read_text()
    assert receipt["source_shape_sq_sk_d"] == [64, 128, 64]
    assert receipt["block_k"] == 32 and receipt["key_blocks"] == 4
    assert receipt["model2mlir_revision"] == mid["model2mlir_revision"]
    assert receipt["frontend_mlir_sha256"] == mid["frontend_mlir_sha256"]
    assert receipt["source_header_archive_sha256"] == _sha(MID / "attn_vpu_mid.h.gz")
    assert receipt["source_header_sha256"] == mid["source_header_sha256"]
    assert receipt["source_spike_reproduced_runs"] == 2
    assert receipt["source_elf_sha256"] == _sha(ARCHIVE / "source.elf")
    assert receipt["source_spike_log_sha256"] == _sha(ARCHIVE / "source_spike.log")
    assert replay["source_baseline_sha256"] == _sha(ARCHIVE / "receipt.json")
    assert replay["source_elf_sha256"] == receipt["source_elf_sha256"]
    assert replay["source_spike_log_sha256"] == receipt["source_spike_log_sha256"]
    assert receipt["expect_header_sha256"] == _sha(ARCHIVE / "attn_flash_mid_expect.h")
    assert receipt["source_output_hash"] == "cd7aef4d68d1e0d9"
    assert "#define EXP_FLASH_HASH_O 0xcd7aef4d68d1e0d9ULL" in expect
    assert "check O hash vs Spike: match (cd7aef4d68d1e0d9)" in log
    assert "attn_flash PASSED" in log
    assert "Bk=32 (4 key blocks)" in log
