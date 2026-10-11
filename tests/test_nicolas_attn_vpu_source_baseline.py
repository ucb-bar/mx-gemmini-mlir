"""Pin Nicolas's dense attention source oracle and the current frontend graph."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_attn_vpu_source_baseline_e9ded36_0344bcb"
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_source_spike_oracle_uses_exact_generated_mesh_golden() -> None:
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    header = gzip.decompress((ARCHIVE / "attn_vpu.h.gz").read_bytes()).decode()
    expect = (ARCHIVE / "attn_vpu_expect.h").read_text()
    log = (ARCHIVE / "source_spike.log").read_text()
    assert receipt["status"] == "original_source_matched_on_pinned_spike"
    assert receipt["source_spike_reproduced_runs"] == 2
    assert receipt["generator_args"] == ["--sq", "32", "--sk", "64", "--d", "64",
                                         "--seed", "0"]
    assert _sha(header.encode()) == receipt["generated_header_sha256"]
    assert _sha((ARCHIVE / "attn_vpu.h.gz").read_bytes()) == receipt[
        "source_header_gzip_sha256"]
    assert _sha(log.encode()) == receipt["source_spike_log_sha256"]
    assert "check S vs mesh golden: 0/512 4-element words differ" in log
    assert "check hashes vs Spike: match (0 of 4 differ)" in log
    assert "attn_vpu PASSED" in log
    assert ("#define ATTN_SQ 32" in header and "#define ATTN_SK 64" in header and
            "#define ATTN_D  64" in header)
    for label, digest in receipt["source_stage_hashes"].items():
        assert f"#define EXP_HASH_{label.upper()} 0x{digest}ULL" in expect
    assert re.search(r"check hashes: S ([0-9a-f]{16})  P ([0-9a-f]{16})  "
                     r"Pq ([0-9a-f]{16})  O ([0-9a-f]{16})", log).groups() == tuple(
                         receipt["source_stage_hashes"][key] for key in ("S", "P", "Pq", "O"))
    block = re.search(r"static const uint16_t S_GOLDEN\[ATTN_SQ\]\[ATTN_SK\].*?=\s*"
                      r"\{(.*?)\};", header, re.DOTALL)
    assert block is not None
    words = np.asarray([int(item, 16) for item in re.findall(r"0x([0-9a-fA-F]{4})",
                                                           block.group(1))], dtype="<u2")
    assert words.size == 32 * 64
    h = 1469598103934665603
    for offset in range(0, words.nbytes, 8):
        h = ((h ^ int.from_bytes(words.tobytes()[offset:offset + 8], "little")) *
             1099511628211) & ((1 << 64) - 1)
    assert f"{h:016x}" == receipt["source_stage_hashes"]["S"]


def test_latest_model2mlir_frontend_identifies_both_attention_contractions() -> None:
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    frontend = (ARCHIVE / "attention.model2mlir.mlir").read_text()
    saved = json.loads((ARCHIVE / "attention.worklist.json").read_text())
    actual = build_model2mlir_worklist(frontend, PROFILE)
    assert _sha(frontend.encode()) == receipt["frontend_mlir_sha256"]
    assert _sha((ARCHIVE / "attention.worklist.json").read_bytes()) == receipt[
        "worklist_sha256"]
    assert actual == saved
    assert receipt["profile_sha256"] == profile_sha256(PROFILE)
    assert receipt["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"
    assert [site["region_id"] for site in saved["rank3_batch_matmuls"]] == [
        "matmul_0", "matmul_1"]
    assert [site["shape_bmnk"] for site in saved["rank3_batch_matmuls"]] == [
        [1, 32, 64, 64], [1, 32, 64, 64]]
    assert 'prov.aten = "aten._softmax.default"' in frontend
    assert 'prov.region_id = "softmax_0"' in frontend
    assert "torch.operator" not in frontend
