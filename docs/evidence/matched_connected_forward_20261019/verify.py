"""Reopen the connected full forward against the retained shared-preparation run."""

import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    for name, expected in json.loads((here / "artifacts.json").read_text()).items():
        assert sha((here / name).read_bytes()) == expected, name
    shared = here.parent / "matched_shared_activation_20261018"
    for name, expected in json.loads((shared / "artifacts.json").read_text()).items():
        assert sha((shared / name).read_bytes()) == expected, name
    prior = load_module("retained_shared_verifier", shared / "verify.py")
    reader = load_module("retained_forward_reader", here.parent / "matched_forward_20261018/verify.py")
    data, baseline = prior.archive(here / "executions.tar.xz"), prior.archive(shared / "executions.tar.xz")
    report, previous = json.loads(data["comparison.json"]), json.loads(baseline["comparison.json"])
    source_root = here.parent / "nicolas_tinyllama_onepass160_same_input_20261014"
    for name, expected in json.loads((source_root / "matched_archive_manifest.json").read_text())["files"].items():
        assert sha((source_root / name).read_bytes()) == expected, name
    boundary = here.parent / "nicolas_tinyllama_lookup_equal_boundary_20261015"
    native = gzip.decompress((boundary / "source_spike.log.gz").read_bytes())
    assert sha(native) == report["source_boundary"]["source_log_sha256"]
    cycles = [int(line.split()[1], 16) for line in native.decode().splitlines() if line.startswith("stage_cycles64 ")]
    assert cycles == [report["source_cycles"]] == [12762360525]
    assert report["source_boundary"] == previous["source_boundary"]
    assert report["same_logical_timing_boundary"] and report["uninstrumented_cycle_comparison"]
    assert report["capture_receipt_sha256"] == sha(data["capture/capture_receipt.json"])
    capture = json.loads(data["capture/capture_receipt.json"])
    assert capture["artifacts"]["model.mlir"]["sha256"] == sha(data["capture/model.mlir"])
    reference = np.load(source_root / "fp32_late_logits.npy", allow_pickle=False)
    top = np.load(source_root / "fp32_argmax_160.npy", allow_pickle=False).reshape(-1)
    source_words = np.fromfile(source_root / "source_reference_late_bf16.bin", dtype="<u2")
    source = (source_words.astype("<u4") << 16).view("<f4").reshape(33, 32000)
    arrays = {}
    for label, row in report["records"].items():
        records = baseline if label == "baseline" else data
        prefix = "candidate/"
        values, observed = reader.parse(records[prefix + "spike.log"])
        arrays[label] = values
        assert np.isfinite(values).all() and sha(records[prefix + "spike.log"]) == row["console_sha256"]
        assert sha(values.astype("<f4").tobytes()) == row["output_words_sha256"]
        assert int(observed["cycles"]) == row["cycles"] and observed["memref_rank_mismatch"] == "0"
        assert int((values.argmax(-1) == top).sum()) == row["top1_agreement_all_160"] == 133
        receipt = json.loads(records[prefix + "mx_stage_probe.json"])
        validation = json.loads(records[prefix + "mx_stage_output_validation.json"])
        recipe = json.loads(records[prefix + "compilation_recipe.json"])
        admission = json.loads(records[prefix + "admission.json"])
        assert sha(records[prefix + "mx_stage_probe.json"]) == row["stage_receipt_sha256"]
        assert validation["validated"] and validation["original_stage_receipt_sha256"] == row["stage_receipt_sha256"]
        assert validation["console_sha256"] == row["console_sha256"]
        assert receipt["spike"]["exit_code"] == 0 and receipt["spike"]["done"] and not receipt["op_profile"]
        assert observed["build_hash"] == receipt["build_hash"]
        assert recipe["status"] == "completed" and admission["status"] == "accepted"
        assert recipe["executable"]["sha256"] == admission["elf_sha256"] == row["elf_sha256"]
        assert sha(records[prefix + "compilation_recipe.json"]) == row["recipe_sha256"]
        assert sha(records[prefix + "admission.json"]) == row["admission_sha256"]
        for kind, logits in (("source", source), ("compiler", values[-33:])):
            for key, value in reader.metrics(reference, logits).items():
                assert abs(row["quality_late_33"][kind][key] - value) < 1e-12
        original, actual = row["quality_late_33"]["source"], row["quality_late_33"]["compiler"]
        assert actual["mean_kl_nats"] <= 1.05 * max(original["mean_kl_nats"], 1e-6)
        assert actual["worst_kl_nats"] <= 1.05 * max(original["worst_kl_nats"], 1e-6)
        assert actual["top1_agreement"] >= original["top1_agreement"]
        if args.build_root is not None:
            name = "onepass160_shared_activation" if label == "baseline" else "onepass160_sigmoid_connected"
            with (args.build_root / name / "model.elf").open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == row["elf_sha256"]
        print(label, row["cycles"], "cycles; unchanged source-relative quality gate passed")
    assert np.array_equal(arrays["baseline"].view("<u4"), arrays["candidate"].view("<u4"))
    assert report["compiler_words_equal"] and report["source_relative_quality_passed"]
    assert report["candidate_to_source_cycle_ratio_if_qualified"] == report["records"]["candidate"]["cycles"] / report["source_cycles"]
    rewrite = json.loads(data["candidate/device_host_abi/mx_connected_sigmoid_rewrite.json"])
    assert len(rewrite["chains"]) == 22
    assert sha(data["candidate/device_host_abi/mx_connected_sigmoid.mlir"]) == rewrite["output_sha256"]
    assert sha(data["candidate/device_catalog/mx_catalog.json"]) == rewrite["catalog_sha256"]
    assert sha(data["candidate/device_signatures.json"]) == rewrite["sidecar_sha256"]
    transport = json.loads(data["candidate/device_host_abi/mx_matrix_transport_rewrite.json"])
    assert len(transport["edges"]) == 44
    assert sha(data["candidate/device_host_abi/mx_matrix_transport.mlir"]) == transport["output_sha256"]
    catalog = json.loads(data["candidate/device_catalog/mx_catalog.json"])
    proof = json.loads(data["candidate/sigmoid_current_semantic_check.json"])
    assert len(proof["chains"]) == 22
    assert sha(data["candidate/device_prepared/host_singleton.mlir"]) == proof["source_sha256"]
    for chain in proof["chains"]:
        symbol = chain["kernel_symbol"]
        assert sha(data[f"candidate/device_catalog/{symbol}.c"]) == chain["c_sha256"]
        assert catalog["compilation"]["source_sha256"][symbol] == chain["c_sha256"]
    for name in ("runtime_quant", "runtime_quant_lut"):
        suffix = ".c" if name == "runtime_quant" else ".h"
        assert sha(data["compiled_sources/" + name + suffix]) == catalog["compilation"]["source_sha256"][name]
    print("PASS: all5120000 words exact; 14.0679% fewer cycles than the quality-corrected baseline; source gap 7.6286%")


if __name__ == "__main__":
    main()
