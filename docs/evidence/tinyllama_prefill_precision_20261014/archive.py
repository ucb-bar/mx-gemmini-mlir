"""Archive a completed prefill output and controlled precision diagnostics."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from tools.compare_tinyllama_session import metrics  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_fp8_log(log: str) -> tuple[np.ndarray, int]:
    chunks = []
    for line in log.splitlines():
        if not line.startswith("OUT_CHUNK f32 "):
            continue
        words = line.split()
        assert int(words[2]) == sum(part.size for part in chunks)
        chunk = np.array([int(word) for word in words[4:]], dtype="<u4")
        assert chunk.size == int(words[3])
        chunks.append(chunk)
    bits = np.concatenate(chunks)
    assert bits.size == 32000
    expected = re.search(r"^OUT_SHA256 f32le 32000 128000 ([0-9a-f]+)$", log, re.M)
    cycles = re.search(r"^METRIC cycles (\d+)$", log, re.M)
    assert expected and sha(bits.tobytes()) == expected.group(1)
    assert cycles and re.search(r"^DONE$", log, re.M)
    return bits.view("<f4").reshape(1, 32000), int(cycles.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-log", type=Path, required=True)
    parser.add_argument("--fp8-log", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--head-manifest", type=Path, required=True)
    args = parser.parse_args()
    if (HERE / "receipt.json").exists():
        parser.error("receipt already exists")

    fast_path = ROOT / ("docs/evidence/nicolas_matrix_substitution_"
                        "tinyllama_full_20261014/compiler/mx_session_run.json")
    fast = json.loads(fast_path.read_text())
    assert fast["complete"] is True and fast["exit_code"] == 0
    reference_source = ROOT / ("docs/evidence/nicolas_tinyllama_"
                               "prefill128_source_20261014/pytorch_last_logits.npy")
    catalog_path = reference_source.with_name("catalog_summary.json")
    catalog = json.loads(catalog_path.read_text())
    shutil.copyfile(reference_source, HERE / "reference.npy")
    reference = np.load(HERE / "reference.npy", allow_pickle=False)
    assert reference.shape == (1, 32000) and reference.dtype == np.float32

    mixed_lines = [line for line in args.mixed_log.read_text().splitlines()
                   if line.startswith("SESSION_HEX 0 0 0 ")]
    assert len(mixed_lines) == 1
    words = mixed_lines[0].split(maxsplit=5)
    assert words[:5] == ["SESSION_HEX", "0", "0", "0", "128000"]
    mixed_bytes = bytes.fromhex(words[5])
    assert len(mixed_bytes) == 128000
    assert sha(mixed_bytes) == fast["output_digests"][0]["sha256"]
    (HERE / "mixed_prefill.bin").write_bytes(mixed_bytes)
    mixed = np.frombuffer(mixed_bytes, dtype="<f4").reshape(1, 32000)

    fp8_log = args.fp8_log.read_text()
    fp8, fp8_cycles = parse_fp8_log(fp8_log)
    (HERE / "fp8_spike.log.gz").write_bytes(
        gzip.compress(fp8_log.encode(), mtime=0))
    (HERE / "fp8_prefill.bin").write_bytes(fp8.tobytes())

    names = {"default": "tiny_precision_default_20261014",
             "head_only": "tiny_precision_head_only_20261014",
             "all_fp8": "tiny_precision_all_fp8_20261014"}
    proxies = {}
    results = {}
    for name, directory in names.items():
        source = args.proxy_root / directory
        shutil.copyfile(source / "proxy.npy", HERE / f"{name}_proxy.npy")
        shutil.copyfile(source / "result.json", HERE / f"{name}_result.json")
        proxy = np.load(HERE / f"{name}_proxy.npy", allow_pickle=False)
        result = json.loads((HERE / f"{name}_result.json").read_text())
        assert result["baseline_logits_sha256"] == sha(reference.tobytes())
        assert result["proxy_logits_sha256"] == sha(proxy.tobytes())
        assert result["capture_weights_sha256"] == catalog["checkpoint_weights_sha256"]
        assert result["capture_inputs_sha256"] == sha(
            reference_source.with_name("compiler_inputs.npz").read_bytes())
        proxies[name] = metrics(proxy, reference)
        results[name] = result
    head = json.loads(args.head_manifest.read_text())
    assert results["head_only"]["precision_overrides"] == head["regions"]
    assert all(result["capture_model_sha256"] == head["source_mlir_sha256"]
               for result in results.values())
    shutil.copyfile(args.head_manifest, HERE / "head_only_overrides.json")
    assert results["default"]["precision_overrides"] == {}
    assert (results["all_fp8"]["precision_overrides"] ==
            catalog["precision_overrides"])
    assert [results[name]["fp4_linear_sites"] for name in names] == [155, 154, 0]
    assert [results[name]["fp8_linear_sites"] for name in names] == [0, 1, 155]

    receipt = {
        "schema": "mx_gemmini.tinyllama_prefill_precision.v1",
        "scope": "one captured TinyLlama 22-layer prefill and final LM head",
        "reference_kind": "captured_unquantized_fp32",
        "numerical_qualification": "not_established",
        "mixed_prefill_matches_completed_fast_run": True,
        "mixed_fast_run_sha256": sha(fast_path.read_bytes()),
        "catalog_sha256": sha(catalog_path.read_bytes()),
        "model2mlir_revision": results["default"]["model2mlir_revision"],
        "capture_model_sha256": results["default"]["capture_model_sha256"],
        "checkpoint_weights_sha256": results["default"]["capture_weights_sha256"],
        "capture_inputs_sha256": results["default"]["capture_inputs_sha256"],
        "profile_sha256": catalog["profile_sha256"],
        "all_fp8_compiler_cycles": fp8_cycles,
        "metrics": {
            "mixed_fp4_fp8_spike": metrics(mixed, reference),
            "all_fp8_spike": metrics(fp8, reference),
            **{f"{name}_operand_proxy": value for name, value in proxies.items()},
        },
    }
    receipt["artifact_sha256"] = {
        path.name: sha(path.read_bytes()) for path in HERE.iterdir()
        if path.name in {"reference.npy", "mixed_prefill.bin", "fp8_prefill.bin",
                         "fp8_spike.log.gz", "head_only_overrides.json"}
        or path.name.endswith(("_proxy.npy", "_result.json"))
    }
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
