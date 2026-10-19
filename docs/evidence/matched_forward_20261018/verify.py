"""Reopen complete compiler words and retained equal-boundary source evidence."""

from array import array
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    values, metrics, digests = array("I"), {}, []
    lines = raw.decode().splitlines()
    assert lines.count("DONE") == 1
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "OUT_BEGIN":
            assert not values and parts == ["OUT_BEGIN", "f32", "5120000"]
        elif parts[0] == "OUT_CHUNK":
            assert parts[1] == "f32" and int(parts[2]) == len(values)
            assert int(parts[3]) == len(parts[4:])
            values.extend(map(int, parts[4:]))
        elif parts[0] == "OUT_SHA256":
            assert parts[1:4] == ["f32le", "5120000", "20480000"]
            digests.append(parts[4])
        elif parts[0] == "OUT_SHA256_INDEX":
            assert parts[1:5] == ["0", "rawbytes", "5120000", "20480000"]
            digests.append(parts[5])
        elif parts[0] == "METRIC":
            assert parts[1] not in metrics
            metrics[parts[1]] = parts[2]
    assert len(values) == 5120000 and values.itemsize == 4
    if sys.byteorder != "little":
        values.byteswap()
    raw = values.tobytes()
    assert digests == [sha(raw), sha(raw)]
    return np.frombuffer(raw, dtype="<f4").reshape(160, 32000), metrics


def metrics(reference, candidate):
    reference, candidate = reference.astype(np.float64), candidate.astype(np.float64)
    def logsoftmax(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))
    a, b = logsoftmax(reference), logsoftmax(candidate)
    kl = np.maximum(0, (np.exp(a) * (a - b)).sum(-1))
    return {"mean_kl_nats": float(kl.mean()), "worst_kl_nats": float(kl.max()),
            "top1_agreement": int((reference.argmax(-1) == candidate.argmax(-1)).sum()), "positions": 33}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    for name, digest in json.loads((here / "artifacts.json").read_text()).items():
        assert sha((here / name).read_bytes()) == digest, name
    with tarfile.open(here / "executions.tar.xz", mode="r:xz") as archive:
        data = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    inventory = json.loads(data.pop("members.json"))
    assert inventory.keys() == data.keys()
    for name, digest in inventory.items():
        assert sha(data[name]) == digest, name
    report = json.loads(data["comparison.json"])
    prior = here.parent / "nicolas_tinyllama_onepass160_same_input_20261014"
    boundary = here.parent / "nicolas_tinyllama_lookup_equal_boundary_20261015"
    for name, digest in json.loads((prior / "matched_archive_manifest.json").read_text())["files"].items():
        assert sha((prior / name).read_bytes()) == digest
    assert sha(data["capture/capture_receipt.json"]) == report["capture_receipt_sha256"]
    capture = json.loads(data["capture/capture_receipt.json"])
    assert sha(data["capture/model.mlir"]) == capture["artifacts"]["model.mlir"]["sha256"]
    reference = np.load(prior / "fp32_late_logits.npy", allow_pickle=False)
    top = np.load(prior / "fp32_argmax_160.npy", allow_pickle=False)
    source = gzip.decompress((boundary / "source_spike.log.gz").read_bytes())
    source_rows = [line.split() for line in source.decode().splitlines()
                   if line.startswith("session_logits_bf16 ")]
    assert [int(row[1]) for row in source_rows] == list(range(33))
    source_words = np.array([int(row[2][i:i + 4], 16)
                            for row in source_rows for i in range(0, len(row[2]), 4)], dtype="<u2")
    assert source_words.tobytes() == (prior / "source_reference_late_bf16.bin").read_bytes()
    source_values = (source_words.astype("<u4") << 16).view("<f4").reshape(33, 32000)
    cycle_lines = [line.split()[1] for line in source.decode().splitlines() if line.startswith("stage_cycles64 ")]
    assert len(cycle_lines) == 1 and int(cycle_lines[0], 16) == report["source_cycles"]
    assert sha(source) == report["source_boundary"]["source_log_sha256"]
    assert sha((boundary / "fair_comparison.json").read_bytes()) == report["source_boundary"]["fair_comparison_sha256"]
    assert report["same_logical_timing_boundary"] is True
    for label, row in report["records"].items():
        prefix = label + "/"
        console = data[prefix + "spike.log"]
        values, observed = parse(console)
        assert np.isfinite(values).all() and sha(console) == row["console_sha256"]
        assert sha(values.astype("<f4").tobytes()) == row["output_words_sha256"]
        assert int(observed["cycles"]) == row["cycles"] and int(observed["memref_rank_mismatch"]) == 0
        receipt = json.loads(data[prefix + "mx_stage_probe.json"])
        validation = json.loads(data[prefix + "mx_stage_output_validation.json"])
        recipe = json.loads(data[prefix + "compilation_recipe.json"])
        admission = json.loads(data[prefix + "admission.json"])
        assert receipt["spike"]["exit_code"] == 0 and observed["build_hash"] == receipt["build_hash"]
        assert validation["validated"] and validation["console_sha256"] == sha(console)
        assert validation["original_stage_receipt_sha256"] == sha(data[prefix + "mx_stage_probe.json"])
        assert recipe["status"] == "completed" and admission["status"] == "accepted"
        assert row["elf_sha256"] == recipe["executable"]["sha256"] == admission["elf_sha256"]
        assert sha(data[prefix + "compilation_recipe.json"]) == row["recipe_sha256"]
        assert sha(data[prefix + "admission.json"]) == row["admission_sha256"]
        for name, output in (("compiler", values[-33:]), ("source", source_values)):
            for key, value in metrics(reference, output).items():
                assert abs(value - row["quality_late_33"][name][key]) < 1e-12
        src, got = metrics(reference, source_values), metrics(reference, values[-33:])
        passed = (got["mean_kl_nats"] <= 1.05 * max(src["mean_kl_nats"], 1e-6)
                  and got["worst_kl_nats"] <= 1.05 * max(src["worst_kl_nats"], 1e-6)
                  and got["top1_agreement"] >= src["top1_agreement"])
        assert passed == row["quality_late_33"]["passed"]
        assert int((values.argmax(-1) == top).sum()) == row["top1_agreement_all_160"]
        if args.build_root:
            image = args.build_root / ("onepass160_" + ("baseline" if label == "baseline" else "optimized")) / "model.elf"
            with image.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == row["elf_sha256"]
        print(label, row["cycles"], "cycles; quality passed:", passed)
    assert report["source_relative_quality_passed"]
    assert report["candidate_to_source_cycle_ratio_if_qualified"] == report["records"]["candidate"]["cycles"] / report["source_cycles"]
    print("PASS: all compiler words, late33 source-relative gates and equal timing boundary")


if __name__ == "__main__":
    main()
