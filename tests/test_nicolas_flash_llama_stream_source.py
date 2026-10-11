"""The streamed flash source oracle and model2MLIR capture are pinned."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.nicolas_attention_shape import DenseAttentionShape
from mx_gemmini_support.nicolas_attention_vector import _audit_frontend_edges
from mx_gemmini_support.target_profile import load_profile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_flash_llama_stream_source_baseline_e9ded36_0344bcb"
PROFILE = load_profile(ROOT / ("profiles/gemmini-mx-cleanup-266c593/"
                               "MxE4M3Fp4VpuGemminiRocketConfig.json"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_streamed_flash_frontend_and_source_geometry() -> None:
    header = gzip.decompress((ARCHIVE / "attn_vpu_llama.h.gz").read_bytes())
    frontend = (ARCHIVE / "attention.model2mlir.mlir").read_text()
    shape = DenseAttentionShape.from_header(header.decode())
    assert shape == DenseAttentionShape(64, 2048, 64)
    _audit_frontend_edges(frontend, shape)
    worklist = build_model2mlir_worklist(frontend, PROFILE)
    assert [site["shape_bmnk"] for site in worklist["rank3_batch_matmuls"]] == [
        [1, 64, 2048, 64], [1, 64, 64, 2048]]
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    assert _sha(ARCHIVE / "attention.model2mlir.mlir") == receipt["frontend_mlir_sha256"]
    assert hashlib.sha256(header).hexdigest() == receipt["source_header_sha256"]
    assert _sha(ARCHIVE / "attn_vpu_llama.h.gz") == receipt["source_header_gzip_sha256"]
    assert receipt["model2mlir_revision"] == "e9ded36eb85abf2d9097ac4dc11457c825853388"


def test_streamed_flash_source_matches_every_bf16_output_twice() -> None:
    receipt = json.loads((ARCHIVE / "receipt.json").read_text())
    assert receipt["status"] == "canonical_streamed_source_full_bf16_spike_parity"
    assert receipt["shape_sq_sk_d_bk"] == [64, 2048, 64, 256]
    assert receipt["source_runs"] == 2 and receipt["compared_bf16_words"] == 4096
    assert receipt["source_output_hash"] == "dc33d078bc537b19"
    assert _sha(ARCHIVE / "source.elf") == receipt["source_elf_sha256"]
    assert _sha(ARCHIVE / "source_spike_1.log") == receipt["source_spike_log_sha256"]
    assert _sha(ARCHIVE / "driver.c") == receipt["source_driver_sha256"]
    outputs = []
    for run in (1, 2):
        log = (ARCHIVE / f"source_spike_{run}.log").read_text()
        assert "K/V streamed (K x2, V x3 block buffers)" in log
        assert "check O hash vs Spike: match (dc33d078bc537b19)" in log
        match = re.search(r"^O hash ([0-9a-f]{16})$\n^O BF16 words ([0-9a-f]+)$",
                          log, re.MULTILINE)
        assert match and match.group(1) == "dc33d078bc537b19"
        assert len(match.group(2)) == 4096 * 4
        outputs.append(match.group(2))
    assert outputs[0] == outputs[1]
