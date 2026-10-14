"""Archive and verify a controlled first-11-layer FP8 prefill Spike run."""

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


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            state.update(chunk)
    return state.hexdigest()


def observed() -> dict:
    log = gzip.decompress((HERE / "first11_spike.log.gz").read_bytes()).decode()
    logits, cycles = parse_fp8_log(log)
    assert logits.tobytes() == (HERE / "first11_prefill.bin").read_bytes()
    reference = np.load(HERE / "reference.npy", allow_pickle=False)
    all_fp8 = np.frombuffer((HERE / "fp8_prefill.bin").read_bytes(),
                            dtype="<f4").reshape(1, 32000)
    mixed = np.frombuffer((HERE / "mixed_prefill.bin").read_bytes(),
                          dtype="<f4").reshape(1, 32000)
    return {
        "cycles": cycles,
        "vs_captured_fp32": metrics(logits, reference),
        "vs_all_fp8_spike": metrics(logits, all_fp8),
        "vs_mixed_spike": metrics(logits, mixed),
    }


def verify() -> None:
    receipt = json.loads((HERE / "first11_spike_receipt.json").read_text())
    assert receipt["schema"] == "mx_gemmini.tinyllama_first11_spike.v1"
    assert receipt["scope"] == "one captured 128-token prefill, 22 layers and head"
    assert receipt["numerical_qualification"] == "not_established"
    assert receipt["spike_isa"] == "rv64gc_zicntr"
    assert receipt["perf_config"] == "e4m3_vpu"
    assert receipt["perf_set"] == "mem.host_tracking=0"
    manifest = json.loads((HERE / "first11_overrides.json").read_text())
    earlier = json.loads((HERE / "receipt.json").read_text())
    assert len(manifest["regions"]) == 77
    assert set(manifest["regions"].values()) == {"mxfp8"}
    assert manifest["profile_sha256"] == receipt["profile_sha256"]
    assert manifest["source_mlir_sha256"] == receipt["capture_model_sha256"]
    assert receipt["capture_model_sha256"] == earlier["capture_model_sha256"]
    assert receipt["profile_sha256"] == earlier["profile_sha256"]
    assert receipt["matrix_sites"] == {"fp8": 122, "fp4": 78}
    catalog = json.loads((HERE / "first11_catalog_summary.json").read_text())
    assert catalog["profile_sha256"] == receipt["profile_sha256"]
    assert catalog["capture_model_sha256"] == receipt["capture_model_sha256"]
    assert catalog["precision_overrides"] == manifest["regions"]
    bindings = catalog["bindings"]
    assert len(bindings) == len({row["region"] for row in bindings}) == 200
    assert {name: sum(row["target_dtypes"][:2] == pair for row in bindings)
            for name, pair in {"fp8": ["f8E4M3FN"] * 2,
                               "fp4": ["f4E2M1FN"] * 2}.items()} == receipt["matrix_sites"]
    assert (HERE / "first11_perf_report.txt").read_bytes().startswith(
        b"gemmini perf [e4m3_vpu]:")
    for name, expected in receipt["artifact_sha256"].items():
        assert digest((HERE / name).read_bytes()) == expected, name
    assert observed() == receipt["observed"]
    print("first-11-layer FP8 Spike result verified")


def archive(args: argparse.Namespace) -> None:
    target = HERE / "first11_spike_receipt.json"
    if target.exists():
        raise ValueError("first11 Spike receipt already exists")
    assert args.exit_code.read_text().strip() == "0"
    log_text = args.run_log.read_text()
    if ("DONE\n" not in log_text or "FATAL:" in log_text
            or "*** FAILED ***" in log_text):
        raise ValueError("TinyLlama guest did not finish successfully")
    logits, _ = parse_fp8_log(log_text)
    assert len(logits.tobytes()) == 128000

    manifest = json.loads((HERE / "first11_overrides.json").read_text())
    probe = json.loads(args.stage_probe.read_text())
    catalog = json.loads(args.catalog.read_text())
    relink = json.loads(args.relink_receipt.read_text())
    assert len(manifest["regions"]) == 77
    assert catalog["precision_overrides"] == probe["precision_overrides"] == manifest["regions"]
    assert probe["capture_model_sha256"] == manifest["source_mlir_sha256"]
    assert catalog["profile_sha256"] == probe["profile_sha256"] == manifest["profile_sha256"]
    assert probe["status"] == "linked_experimental_numeric_qualification_pending"
    assert catalog["covered_contractions"] == probe["selected_operations"] == 200
    assert relink["schema"] == "mx_gemmini.stage_output_tap_relink.v1"
    assert relink["dump_cap_elements"] == 32000
    assert digest_file(args.elf) == relink["dump_elf_sha256"]
    counts = {"fp8": 0, "fp4": 0}
    for binding in catalog["bindings"]:
        pair = binding["target_dtypes"][:2]
        if pair == ["f8E4M3FN", "f8E4M3FN"]:
            counts["fp8"] += 1
        elif pair == ["f4E2M1FN", "f4E2M1FN"]:
            counts["fp4"] += 1
        else:
            raise ValueError(f"unexpected matrix pair: {pair}")
    assert counts == {"fp8": 122, "fp4": 78}

    summary = {
        "profile_sha256": catalog["profile_sha256"],
        "capture_model_sha256": probe["capture_model_sha256"],
        "precision_overrides": catalog["precision_overrides"],
        "bindings": [{"region": binding["region"],
                      "target_dtypes": binding["target_dtypes"]}
                     for binding in catalog["bindings"]],
    }
    (HERE / "first11_catalog_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (HERE / "first11_spike.log.gz").write_bytes(gzip.compress(log_text.encode(), mtime=0))
    (HERE / "first11_prefill.bin").write_bytes(logits.tobytes())
    (HERE / "first11_perf_report.txt").write_bytes(args.perf_report.read_bytes())
    names = ["first11_catalog_summary.json", "first11_spike.log.gz",
             "first11_prefill.bin", "first11_overrides.json",
             "first11_perf_report.txt", "reference.npy", "fp8_prefill.bin",
             "mixed_prefill.bin"]
    receipt = {
        "schema": "mx_gemmini.tinyllama_first11_spike.v1",
        "scope": "one captured 128-token prefill, 22 layers and head",
        "numerical_qualification": "not_established",
        "spike_isa": "rv64gc_zicntr",
        "perf_config": "e4m3_vpu",
        "perf_set": "mem.host_tracking=0",
        "profile_sha256": probe["profile_sha256"],
        "capture_model_sha256": probe["capture_model_sha256"],
        "matrix_sites": counts,
        "build_hash": probe["build_hash"],
        "stage_probe_sha256": digest(args.stage_probe.read_bytes()),
        "catalog_sha256": digest(args.catalog.read_bytes()),
        "relink_receipt_sha256": digest(args.relink_receipt.read_bytes()),
        "elf_sha256": relink["dump_elf_sha256"],
        "artifact_sha256": {name: digest((HERE / name).read_bytes()) for name in names},
        "observed": observed(),
    }
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    verify()
    print(json.dumps(receipt["observed"], indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--run-log", type=Path)
    parser.add_argument("--perf-report", type=Path)
    parser.add_argument("--exit-code", type=Path)
    parser.add_argument("--stage-probe", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--relink-receipt", type=Path)
    parser.add_argument("--elf", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        for name in ("run_log", "perf_report", "exit_code", "stage_probe", "catalog",
                     "relink_receipt", "elf"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} required for archive")
        archive(args)


if __name__ == "__main__":
    main()
