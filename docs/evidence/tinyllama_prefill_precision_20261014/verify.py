"""Verify archived TinyLlama prefill precision outputs and identities."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from tools.compare_tinyllama_session import metrics  # noqa: E402
from docs.evidence.tinyllama_prefill_precision_20261014.archive import parse_fp8_log  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify() -> None:
    receipt = json.loads((HERE / "receipt.json").read_text())
    assert receipt["schema"] == "mx_gemmini.tinyllama_prefill_precision.v1"
    assert receipt["numerical_qualification"] == "not_established"
    assert receipt["reference_kind"] == "captured_unquantized_fp32"
    for name, expected in receipt["artifact_sha256"].items():
        assert sha((HERE / name).read_bytes()) == expected, name

    source = ROOT / "docs/evidence/nicolas_tinyllama_prefill128_source_20261014"
    fast_path = ROOT / ("docs/evidence/nicolas_matrix_substitution_"
                        "tinyllama_full_20261014/compiler/mx_session_run.json")
    catalog_path = source / "catalog_summary.json"
    fast = json.loads(fast_path.read_text())
    catalog = json.loads(catalog_path.read_text())
    assert sha(fast_path.read_bytes()) == receipt["mixed_fast_run_sha256"]
    assert fast["complete"] is True and fast["exit_code"] == 0
    assert sha(catalog_path.read_bytes()) == receipt["catalog_sha256"]
    assert catalog["covered_contractions"] == 200
    assert catalog["profile_sha256"] == receipt["profile_sha256"]
    assert catalog["checkpoint_weights_sha256"] == receipt["checkpoint_weights_sha256"]
    assert sha((source / "compiler_inputs.npz").read_bytes()) == receipt["capture_inputs_sha256"]

    reference = np.load(HERE / "reference.npy", allow_pickle=False)
    assert np.array_equal(reference, np.load(source / "pytorch_last_logits.npy",
                                            allow_pickle=False))
    assert reference.shape == (1, 32000) and reference.dtype == np.float32
    mixed_bytes = (HERE / "mixed_prefill.bin").read_bytes()
    assert len(mixed_bytes) == 128000
    assert sha(mixed_bytes) == fast["output_digests"][0]["sha256"]
    mixed = np.frombuffer(mixed_bytes, dtype="<f4").reshape(1, 32000)
    fp8, cycles = parse_fp8_log(gzip.decompress(
        (HERE / "fp8_spike.log.gz").read_bytes()).decode())
    assert np.array_equal(fp8, np.frombuffer(
        (HERE / "fp8_prefill.bin").read_bytes(), dtype="<f4").reshape(1, 32000))
    assert cycles == receipt["all_fp8_compiler_cycles"]
    assert receipt["mixed_prefill_matches_completed_fast_run"] is True

    head = json.loads((HERE / "head_only_overrides.json").read_text())
    assert head["regions"] == {"matmul_199": "mxfp8"}
    assert head["source_mlir_sha256"] == receipt["capture_model_sha256"]
    assert head["profile_sha256"] == receipt["profile_sha256"]
    arrays = {"mixed_fp4_fp8_spike": mixed, "all_fp8_spike": fp8}
    counts = {"default": (155, 0), "head_only": (154, 1),
              "all_fp8": (0, 155)}
    for name, (fp4_count, fp8_count) in counts.items():
        result = json.loads((HERE / f"{name}_result.json").read_text())
        proxy = np.load(HERE / f"{name}_proxy.npy", allow_pickle=False)
        assert result["status"] == "diagnostic_not_mesh_oracle"
        assert result["model2mlir_revision"] == receipt["model2mlir_revision"]
        assert result["capture_model_sha256"] == receipt["capture_model_sha256"]
        assert result["capture_weights_sha256"] == receipt["checkpoint_weights_sha256"]
        assert result["capture_inputs_sha256"] == receipt["capture_inputs_sha256"]
        assert result["baseline_logits_sha256"] == sha(reference.tobytes())
        assert result["proxy_logits_sha256"] == sha(proxy.tobytes())
        assert result["fp4_linear_sites"] == fp4_count
        assert result["fp8_linear_sites"] == fp8_count
        assert result["fp8_attention_calls"] == 44
        assert result["fp8_rotary_bmm_calls"] == 1
        arrays[f"{name}_operand_proxy"] = proxy
    assert json.loads((HERE / "default_result.json").read_text())["precision_overrides"] == {}
    assert json.loads((HERE / "head_only_result.json").read_text())["precision_overrides"] == head["regions"]
    assert (json.loads((HERE / "all_fp8_result.json").read_text())["precision_overrides"]
            == catalog["precision_overrides"])
    for name, values in arrays.items():
        observed = metrics(values, reference)
        stored = receipt["metrics"][name]
        for key, value in observed.items():
            assert abs(stored[key] - value) < 1e-12, (name, key)
    print("TinyLlama prefill precision archive verified")


if __name__ == "__main__":
    verify()
