"""Verify the second Nicolas dense attention shape's source and frontend oracle."""

import gzip
import hashlib
import json
from pathlib import Path
import re

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mid_source_and_clean_frontend_oracles() -> None:
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    header = gzip.decompress((ARCHIVE / "attn_vpu_mid.h.gz").read_bytes())
    log = (ARCHIVE / "source_spike.log").read_text()
    mlir = (ARCHIVE / "attention.model2mlir.mlir").read_text()
    worklist = build_model2mlir_worklist(mlir, PROFILE)
    assert receipt["source_spike_reproduced_runs"] == 2
    assert receipt["model2mlir_revision"] == "a042643e31366724ca0482389c4ee85a9f88e343"
    assert receipt["generator_args"] == ["--sq", "64", "--sk", "128", "--d",
                                         "64", "--seed", "0", "--tag", "_mid"]
    assert receipt["source_header_sha256"] == hashlib.sha256(header).hexdigest()
    assert receipt["source_header_gzip_sha256"] == _sha(ARCHIVE / "attn_vpu_mid.h.gz")
    assert receipt["source_elf_sha256"] == _sha(ARCHIVE / "source.elf")
    assert receipt["source_spike_log_sha256"] == _sha(ARCHIVE / "source_spike.log")
    assert receipt["frontend_mlir_sha256"] == _sha(ARCHIVE / "attention.model2mlir.mlir")
    assert receipt["worklist_sha256"] == _sha(ARCHIVE / "attention.worklist.json")
    assert worklist == json.loads((ARCHIVE / "attention.worklist.json").read_text())
    assert [(site["region_id"], site["shape_bmnk"])
            for site in worklist["rank3_batch_matmuls"]] == [
                ("matmul_0", [1, 64, 128, 64]),
                ("matmul_1", [1, 64, 64, 128])]
    assert b"#define ATTN_SQ 64" in header and b"#define ATTN_SK 128" in header
    assert b"#define ATTN_D  64" in header
    assert "check S vs mesh golden: 0/2048 4-element words differ" in log
    assert "check hashes vs Spike: match (0 of 4 differ)" in log
    assert "attn_vpu PASSED" in log
    hashes = re.search(r"check hashes: S ([0-9a-f]{16})  P ([0-9a-f]{16})  "
                       r"Pq ([0-9a-f]{16})  O ([0-9a-f]{16})", log)
    assert hashes is not None
    assert dict(zip(("S", "P", "Pq", "O"), hashes.groups())) == receipt[
        "source_stage_hashes"]
