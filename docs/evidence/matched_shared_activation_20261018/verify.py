"""Reopen the full shared-preparation result against the retained corrected run."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def archive(path):
    with tarfile.open(path, "r:xz") as stream:
        data = {m.name: stream.extractfile(m).read() for m in stream.getmembers() if m.isfile()}
    inventory = json.loads(data.pop("members.json"))
    assert inventory.keys() == data.keys()
    for name, digest in inventory.items():
        assert sha(data[name]) == digest, name
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    for name, digest in json.loads((here / "artifacts.json").read_text()).items():
        assert sha((here / name).read_bytes()) == digest, name
    earlier = here.parent / "matched_forward_20261018"
    for name, digest in json.loads((earlier / "artifacts.json").read_text()).items():
        assert sha((earlier / name).read_bytes()) == digest, name
    spec = importlib.util.spec_from_file_location("retained_forward_reader", earlier / "verify.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    data, old = archive(here / "executions.tar.xz"), archive(earlier / "executions.tar.xz")
    report, previous = json.loads(data["comparison.json"]), json.loads(old["comparison.json"])
    source_root = here.parent / "nicolas_tinyllama_onepass160_same_input_20261014"
    for name, digest in json.loads((source_root / "matched_archive_manifest.json").read_text())["files"].items():
        assert sha((source_root / name).read_bytes()) == digest, name
    boundary_root = here.parent / "nicolas_tinyllama_lookup_equal_boundary_20261015"
    boundary_log = gzip.decompress((boundary_root / "source_spike.log.gz").read_bytes())
    assert sha(boundary_log) == report["source_boundary"]["source_log_sha256"]
    cycles = [int(line.split()[1], 16) for line in boundary_log.decode().splitlines() if line.startswith("stage_cycles64 ")]
    assert cycles == [report["source_cycles"]] == [12762360525]
    assert report["source_boundary"] == previous["source_boundary"] and report["same_logical_timing_boundary"]
    assert report["capture_receipt_sha256"] == sha(old["capture/capture_receipt.json"])
    reference = np.load(source_root / "fp32_late_logits.npy", allow_pickle=False)
    source_words = np.fromfile(source_root / "source_reference_late_bf16.bin", dtype="<u2")
    source = (source_words.astype("<u4") << 16).view("<f4").reshape(33, 32000)
    arrays = {}
    for label, row in report["records"].items():
        records, prefix = (old, "candidate/") if label == "baseline" else (data, "candidate/")
        values, observed = reader.parse(records[prefix + "spike.log"])
        arrays[label] = values
        assert np.isfinite(values).all() and sha(records[prefix + "spike.log"]) == row["console_sha256"]
        assert sha(values.astype("<f4").tobytes()) == row["output_words_sha256"]
        assert int(observed["cycles"]) == row["cycles"] and observed["memref_rank_mismatch"] == "0"
        receipt = json.loads(records[prefix + "mx_stage_probe.json"])
        validation = json.loads(records[prefix + "mx_stage_output_validation.json"])
        recipe = json.loads(records[prefix + "compilation_recipe.json"])
        admission = json.loads(records[prefix + "admission.json"])
        assert sha(records[prefix + "mx_stage_probe.json"]) == row["stage_receipt_sha256"]
        assert validation["validated"] and validation["original_stage_receipt_sha256"] == row["stage_receipt_sha256"]
        assert validation["console_sha256"] == row["console_sha256"] and receipt["spike"]["exit_code"] == 0
        assert observed["build_hash"] == receipt["build_hash"] and not receipt["op_profile"]
        assert recipe["status"] == "completed" and admission["status"] == "accepted"
        assert recipe["executable"]["sha256"] == admission["elf_sha256"] == row["elf_sha256"]
        assert sha(records[prefix + "compilation_recipe.json"]) == row["recipe_sha256"]
        assert sha(records[prefix + "admission.json"]) == row["admission_sha256"]
        for kind, logits in (("source", source), ("compiler", values[-33:])):
            for key, value in reader.metrics(reference, logits).items():
                assert abs(row["quality_late_33"][kind][key] - value) < 1e-12
        src, got = row["quality_late_33"]["source"], row["quality_late_33"]["compiler"]
        assert got["mean_kl_nats"] <= 1.05 * max(src["mean_kl_nats"], 1e-6)
        assert got["worst_kl_nats"] <= 1.05 * max(src["worst_kl_nats"], 1e-6)
        assert got["top1_agreement"] >= src["top1_agreement"]
        if args.build_root:
            name = "onepass160_optimized" if label == "baseline" else "onepass160_shared_activation"
            with (args.build_root / name / "model.elf").open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == row["elf_sha256"]
        print(label, row["cycles"], "cycles; unchanged source-relative gate passed")
    assert np.array_equal(arrays["baseline"].view("<u4"), arrays["candidate"].view("<u4"))
    assert report["compiler_words_equal"] and report["source_relative_quality_passed"]
    assert report["uninstrumented_cycle_comparison"]
    assert report["candidate_to_source_cycle_ratio_if_qualified"] == report["records"]["candidate"]["cycles"] / report["source_cycles"]
    rewrite = json.loads(data["candidate/device_host_abi/mx_shared_activation_rewrite.json"])
    assert len(rewrite["groups"]) == 44 and rewrite["removed_preparations"] == 66
    assert sha(data["candidate/device_host_abi/mx_shared_activation.mlir"]) == rewrite["output_sha256"]
    assert sha(data["candidate/device_catalog/mx_catalog.json"]) == rewrite["catalog_sha256"]
    assert sha(data["candidate/device_signatures.json"]) == rewrite["sidecar_sha256"]
    print("PASS: all5120000 compiler words exact; measured full-forward improvement, remaining source gap retained")


if __name__ == "__main__":
    main()
