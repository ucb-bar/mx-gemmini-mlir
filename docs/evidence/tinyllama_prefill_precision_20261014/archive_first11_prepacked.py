"""Archive a prepacked first-11-layer FP8 TinyLlama prefill on Spike."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from docs.evidence.tinyllama_prefill_precision_20261014.archive import parse_fp8_log  # noqa: E402
from tools.compare_tinyllama_session import metrics  # noqa: E402


def sha_file(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()


def read_json(name: str) -> dict:
    return json.loads((HERE / name).read_text())


def observed() -> dict:
    log = gzip.decompress((HERE / "first11_prepacked_spike.log.gz").read_bytes()).decode()
    logits, cycles = parse_fp8_log(log)
    assert logits.tobytes() == (HERE / "first11_prepacked_prefill.bin").read_bytes()
    fp32 = np.load(HERE / "reference.npy", allow_pickle=False)
    all_fp8 = np.frombuffer((HERE / "fp8_prefill.bin").read_bytes(),
                            dtype="<f4").reshape(1, 32000)
    mixed = np.frombuffer((HERE / "mixed_prefill.bin").read_bytes(),
                          dtype="<f4").reshape(1, 32000)
    runtime = np.frombuffer((HERE / "first11_prefill.bin").read_bytes(),
                            dtype="<f4").reshape(1, 32000)
    return {
        "cycles": cycles,
        "vs_captured_fp32": metrics(logits, fp32),
        "vs_all_fp8_spike": metrics(logits, all_fp8),
        "vs_mixed_spike": metrics(logits, mixed),
        "vs_runtime_packed_first11": metrics(logits, runtime),
    }


def check_build(probe: dict, catalog: dict, runtime: dict) -> None:
    manifest = read_json("first11_overrides.json")
    assert probe["status"] == "linked_experimental_numeric_qualification_pending"
    assert probe["prepack_checkpoint_weights"] is True
    assert probe["prepacked_checkpoint_kernels"] == 155
    assert runtime["prepack_checkpoint_weights"] is False
    assert runtime["prepacked_checkpoint_kernels"] == 0
    assert probe["capture_model_sha256"] == runtime["capture_model_sha256"] == manifest["source_mlir_sha256"]
    assert probe["profile_sha256"] == runtime["profile_sha256"] == manifest["profile_sha256"]
    assert probe["precision_overrides"] == runtime["precision_overrides"] == manifest["regions"]
    assert probe["precision_overrides_sha256"] == runtime["precision_overrides_sha256"]
    assert probe["selected_operations"] == runtime["selected_operations"] == 200
    for key in ("vpu_add_lowered_regions", "vpu_elementwise_lowered_regions",
                "vpu_reduce_lowered_regions", "vpu_softmax_lowered_regions",
                "vpu_unary_lowered_regions"):
        assert probe[key] == runtime[key]
    assert catalog["profile_sha256"] == probe["profile_sha256"]
    assert catalog["precision_overrides"] == manifest["regions"]
    assert catalog["covered_contractions"] == 200
    assert catalog["prepacked_checkpoint_kernels"] == 155
    assert catalog["checkpoint_weights_sha256"] == probe["checkpoint_weights_sha256"]
    bindings = catalog["bindings"]
    assert len(bindings) == len({row["region"] for row in bindings}) == 200
    assert [{"region": row["region"], "target_dtypes": row["target_dtypes"]}
            for row in bindings] == read_json("first11_catalog_summary.json")["bindings"]
    pairs = [row["target_dtypes"][:2] for row in bindings]
    assert pairs.count(["f8E4M3FN"] * 2) == 122
    assert pairs.count(["f4E2M1FN"] * 2) == 78


def verify() -> None:
    receipt = read_json("first11_prepacked_spike_receipt.json")
    assert receipt["schema"] == "mx_gemmini.tinyllama_first11_prepacked_spike.v1"
    assert receipt["scope"] == "one captured 128-token prefill, 22 layers and head"
    assert receipt["numerical_qualification"] == "not_established"
    assert receipt["spike_isa"] == "rv64gc_zicntr"
    assert receipt["perf_config"] == "e4m3_vpu"
    assert receipt["perf_set"] == "mem.host_tracking=0"
    assert receipt["matrix_sites"] == {"fp8": 122, "fp4": 78}
    assert receipt["prepacked_checkpoint_kernels"] == 155
    runtime_receipt = read_json("first11_spike_receipt.json")
    assert receipt["capture_model_sha256"] == runtime_receipt["capture_model_sha256"]
    assert receipt["profile_sha256"] == runtime_receipt["profile_sha256"]
    assert receipt["runtime_control_build_hash"] == runtime_receipt["build_hash"]
    assert receipt["runtime_control_cycles"] == runtime_receipt["observed"]["cycles"]
    artifacts = receipt["artifact_sha256"]
    for name, expected in artifacts.items():
        assert sha_file(HERE / name) == expected, name
    probe = read_json("first11_prepacked_probe.json")
    catalog = read_json("first11_prepacked_catalog.json")
    runtime = read_json("first11_runtime_probe.json")
    check_build(probe, catalog, runtime)
    assert receipt["build_hash"] == probe["build_hash"]
    assert receipt["observed"] == observed()
    assert (HERE / "first11_prepacked_perf_report.txt").read_bytes().startswith(
        b"gemmini perf [e4m3_vpu]:")
    print("first-11-layer prepacked FP8 Spike result verified")


def archive(args: argparse.Namespace) -> None:
    target = HERE / "first11_prepacked_spike_receipt.json"
    if target.exists():
        raise ValueError("prepacked first11 receipt already exists")
    assert args.exit_code.read_text().strip() == "0"
    log = args.run_log.read_text()
    assert "FATAL:" not in log and "*** FAILED ***" not in log
    logits, _ = parse_fp8_log(log)
    assert len(logits.tobytes()) == 128000
    probe = json.loads(args.stage_probe.read_text())
    catalog = json.loads(args.catalog.read_text())
    runtime = json.loads(args.runtime_stage_probe.read_text())
    check_build(probe, catalog, runtime)
    for name, value in (("first11_prepacked_probe.json", probe),
                        ("first11_prepacked_catalog.json", catalog),
                        ("first11_runtime_probe.json", runtime)):
        (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    (HERE / "first11_prepacked_spike.log.gz").write_bytes(
        gzip.compress(log.encode(), mtime=0))
    (HERE / "first11_prepacked_prefill.bin").write_bytes(logits.tobytes())
    (HERE / "first11_prepacked_perf_report.txt").write_bytes(
        args.perf_report.read_bytes())
    artifact_names = ("first11_prepacked_probe.json", "first11_prepacked_catalog.json",
                      "first11_runtime_probe.json", "first11_prepacked_spike.log.gz",
                      "first11_prepacked_prefill.bin", "first11_prepacked_perf_report.txt",
                      "first11_spike_receipt.json", "first11_prefill.bin",
                      "first11_overrides.json", "reference.npy", "fp8_prefill.bin",
                      "mixed_prefill.bin")
    runtime_receipt = read_json("first11_spike_receipt.json")
    receipt = {
        "schema": "mx_gemmini.tinyllama_first11_prepacked_spike.v1",
        "scope": "one captured 128-token prefill, 22 layers and head",
        "numerical_qualification": "not_established",
        "spike_isa": "rv64gc_zicntr",
        "perf_config": "e4m3_vpu",
        "perf_set": "mem.host_tracking=0",
        "matrix_sites": {"fp8": 122, "fp4": 78},
        "prepacked_checkpoint_kernels": 155,
        "capture_model_sha256": probe["capture_model_sha256"],
        "profile_sha256": probe["profile_sha256"],
        "build_hash": probe["build_hash"],
        "runtime_control_build_hash": runtime_receipt["build_hash"],
        "runtime_control_cycles": runtime_receipt["observed"]["cycles"],
        "elf_bytes": args.elf.stat().st_size,
        "elf_sha256": sha_file(args.elf),
        "artifact_sha256": {name: sha_file(HERE / name) for name in artifact_names},
        "observed": observed(),
    }
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    verify()
    print(json.dumps(receipt["observed"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    for name in ("run_log", "perf_report", "exit_code", "stage_probe", "catalog",
                 "runtime_stage_probe", "elf"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        for name in ("run_log", "perf_report", "exit_code", "stage_probe",
                     "catalog", "runtime_stage_probe", "elf"):
            if getattr(args, name) is None:
                parser.error("--" + name.replace("_", "-") + " required")
        archive(args)


if __name__ == "__main__":
    main()
