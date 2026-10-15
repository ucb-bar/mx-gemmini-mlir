"""Verify the archived all-FP8 TinyLlama session without checkpoint files."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(actual: np.ndarray, expected: np.ndarray) -> float:
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    return float(np.dot(a.ravel(), b.ravel()) /
                 (np.linalg.norm(a) * np.linalg.norm(b)))


def verify() -> None:
    archive = json.loads((HERE / "archive_manifest.json").read_text())
    assert archive["schema"] == "mx_gemmini.tinyllama_all_fp8_full_archive.v1"
    for name, digest in archive["files"].items():
        assert sha(HERE / name) == digest, name
    run = json.loads((HERE / "run.json").read_text())
    link = json.loads((HERE / "link.json").read_text())
    result = json.loads((HERE / "against_fp32.json").read_text())
    assert run["complete"] and run["exit_code"] == 0
    assert run["link_receipt_sha256"] == sha(HERE / "link.json")
    assert run["log_sha256"] == hashlib.sha256(
        gzip.decompress((HERE / "spike.log.gz").read_bytes())).hexdigest()
    assert result["run_receipt_sha256"] == sha(HERE / "run.json")
    assert result["link_receipt_sha256"] == sha(HERE / "link.json")
    assert result["prefill_build_receipt_sha256"] == sha(HERE / "prefill_build.json")
    assert result["decode_build_receipt_sha256"] == sha(HERE / "decode_build.json")
    assert result["prefill_reference_sha256"] == sha(HERE / "prefill_golden.npy")
    assert result["decode_reference_sha256"] == sha(HERE / "decode_fp32.npz")
    assert run["timing_config"] == "e4m3_vpu"
    assert run["timing_overrides"] == "mem.host_tracking=0"
    assert len(run["step_performance"]) == 33
    assert sum(row["cycles"] for row in run["step_performance"]) == result[
        "total_modeled_cycles"] == run["total_performance"]["cycles"]
    assert [row["steps"] for row in link["programs"]] == [1, 32]
    for stage in ("prefill", "decode"):
        build = json.loads((HERE / f"{stage}_build.json").read_text())
        catalog = json.loads((HERE / f"{stage}_catalog.json").read_text())
        assert build["profile_sha256"] == link["profile_sha256"]
        assert build["precision_overrides_sha256"] == sha(
            HERE / f"{stage}_overrides.json")
        assert build["selected_operations"] == 200
        assert len(catalog["bindings"]) == 200
        assert all(row["target_dtypes"][:2] == ["f8E4M3FN", "f8E4M3FN"]
                   for row in catalog["bindings"])

    golden = np.load(HERE / "prefill_golden.npy", allow_pickle=False)
    with np.load(HERE / "decode_fp32.npz", allow_pickle=False) as data:
        decode = data["logits"]
    assert golden.shape == (1, 32000) and decode.shape == (32, 1, 32000)
    dumps = {(row["program"], row["step"], row["output"]): row
             for row in run["output_dumps"]}
    digests = {(row["program"], row["step"], row["output"]): row
               for row in run["output_digests"]}
    assert len(dumps) == 33 and len(digests) == 99
    matches = 0
    for key, row in dumps.items():
        path = HERE / row["path"]
        assert path.stat().st_size == 128000
        assert sha(path) == row["sha256"] == digests[key]["sha256"]
        actual = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(1, 32000)
        reference = golden if key[0] == 0 else decode[key[1]]
        recorded = (result["prefill_metrics"] if key[0] == 0 else
                    result["decode_step_metrics"][key[1]])
        assert abs(cosine(actual, reference) - recorded["cosine"]) < 1e-12
        same_top = bool(np.argmax(actual) == np.argmax(reference))
        assert same_top == recorded["top_token_equal"]
        if key[0] == 1:
            matches += same_top
    assert matches == result["decode_summary"]["top_token_matches"] == 2
    print("all 33 outputs, FP8 bindings, receipts, and cycles verified")


if __name__ == "__main__":
    verify()
