"""A source performance ratio must include the same token lookup work."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tools.compare_matched_tinyllama_forward import read_equal_boundary

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/nicolas_tinyllama_onepass160_same_input_20261014"
BOUNDARY = ROOT / "docs/evidence/nicolas_tinyllama_lookup_equal_boundary_20261015"


def test_equal_boundary_reopens_raw_source_cycles_including_embedding_lookup():
    result = read_equal_boundary(ARCHIVE, BOUNDARY)
    source = json.loads((BOUNDARY / "run_receipt.json").read_text())
    old = json.loads((ARCHIVE / "matched_onepass_comparison.json").read_text())
    assert result["cycles"] == source["stage_cycles"]
    assert result["cycles"] > old["source_stage_cycles"]
    assert result["boundary"] == "token_ids_to_logits_including_embedding_lookup"


@pytest.mark.parametrize("field,value", [
    ("source_stage_cycles", 1), ("same_logical_input_output_boundary", False),
    ("source_log_sha256", "0" * 64), ("source_build_receipt_sha256", "0" * 64),
])
def test_changed_boundary_or_run_refuses(tmp_path, field, value):
    for path in BOUNDARY.iterdir():
        if path.is_file():
            shutil.copyfile(path, tmp_path / path.name)
    path = tmp_path / "fair_comparison.json"
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        read_equal_boundary(ARCHIVE, tmp_path)


def test_receipt_bound_source_timing_model_deadlock_refuses(tmp_path):
    for path in BOUNDARY.iterdir():
        if path.is_file():
            shutil.copyfile(path, tmp_path / path.name)
    log = tmp_path / "source_spike.log.gz"
    console = gzip.decompress(log.read_bytes()) + b"gemmini perf: MODEL DEADLOCK at fence\n"
    log.write_bytes(gzip.compress(console, mtime=0))
    run_path = tmp_path / "run_receipt.json"
    run = json.loads(run_path.read_text())
    run["log_sha256"] = hashlib.sha256(console).hexdigest()
    run_path.write_text(json.dumps(run))
    comparison_path = tmp_path / "fair_comparison.json"
    comparison = json.loads(comparison_path.read_text())
    comparison["source_log_sha256"] = run["log_sha256"]
    comparison["source_run_receipt_sha256"] = hashlib.sha256(run_path.read_bytes()).hexdigest()
    comparison_path.write_text(json.dumps(comparison))
    with pytest.raises(ValueError, match="timing model deadlocked"):
        read_equal_boundary(ARCHIVE, tmp_path)
