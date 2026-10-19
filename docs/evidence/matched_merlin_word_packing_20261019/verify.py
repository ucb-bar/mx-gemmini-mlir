"""Check the generic Merlin word-packing full forward and unchanged gates."""

import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((HERE / "artifacts.json").read_text())
    raw = (HERE / "executions.tar.xz").read_bytes()
    assert sha(raw) == manifest["archive_sha256"] and len(raw) == manifest["archive_bytes"]
    data = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and ".." not in name.parts
            assert member.name not in data
            data[member.name] = archive.extractfile(member).read()
    assert set(data) == set(manifest["members"])
    for name, raw in data.items():
        assert sha(raw) == manifest["members"][name]["sha256"]
        assert len(raw) == manifest["members"][name]["bytes"]
    reader_path = HERE.parent / "matched_forward_20261018/verify.py"
    assert sha(reader_path.read_bytes()) == manifest["reader_sha256"]
    spec = importlib.util.spec_from_file_location("retained_forward_reader", reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)

    def load(name):
        return json.loads(data[name])

    report = load("comparison.json")
    capture = load("capture/capture_receipt.json")
    assert report["capture_receipt_sha256"] == sha(data["capture/capture_receipt.json"])
    assert capture["artifacts"]["model.mlir"]["sha256"] == sha(data["capture/model.mlir"])
    assert capture["artifacts"]["inputs.npz"]["sha256"] == sha(data["capture/inputs.npz"])
    source_root = HERE.parent / "nicolas_tinyllama_onepass160_same_input_20261014"
    source_manifest = json.loads((source_root / "matched_archive_manifest.json").read_text())
    for name, digest in source_manifest["files"].items():
        assert sha((source_root / name).read_bytes()) == digest
    boundary = HERE.parent / "nicolas_tinyllama_lookup_equal_boundary_20261015"
    console = gzip.decompress((boundary / "source_spike.log.gz").read_bytes())
    assert sha(console) == report["source_boundary"]["source_log_sha256"]
    assert sha((boundary / "fair_comparison.json").read_bytes()) == report["source_boundary"]["fair_comparison_sha256"]
    assert sha((boundary / "run_receipt.json").read_bytes()) == report["source_boundary"]["source_run_receipt_sha256"]
    assert [int(x, 16) for x in re.findall(rb"^stage_cycles64 0x([0-9a-f]{16})$", console, re.M)] == [report["source_cycles"]] == [12762360525]
    reference = np.load(source_root / "fp32_late_logits.npy", allow_pickle=False)
    top = np.load(source_root / "fp32_argmax_160.npy", allow_pickle=False).reshape(-1)
    source_words = np.fromfile(source_root / "source_reference_late_bf16.bin", dtype="<u2")
    source = (source_words.astype("<u4") << 16).view("<f4").reshape(33, 32000)
    arrays = {}
    for label, row in report["records"].items():
        prefix = label + "/"
        values, observed = reader.parse(data[prefix + "spike.log"])
        arrays[label] = values
        assert np.isfinite(values).all()
        assert sha(data[prefix + "spike.log"]) == row["console_sha256"]
        assert sha(values.astype("<f4").tobytes()) == row["output_words_sha256"]
        assert int(observed["cycles"]) == row["cycles"]
        assert observed["memref_rank_mismatch"] == "0" and not row["timing_instrumented"]
        assert int((values.argmax(-1) == top).sum()) == row["top1_agreement_all_160"] == 133
        receipt = load(prefix + "mx_stage_probe.json")
        validation = load(prefix + "mx_stage_output_validation.json")
        recipe = load(prefix + "compilation_recipe.json")
        admission = load(prefix + "admission.json")
        assert sha(data[prefix + "mx_stage_probe.json"]) == row["stage_receipt_sha256"]
        assert sha(data[prefix + "mx_stage_output_validation.json"]) == row["complete_output_validation_sha256"]
        assert validation["validated"] and validation["original_stage_receipt_sha256"] == row["stage_receipt_sha256"]
        assert validation["console_sha256"] == row["console_sha256"]
        assert receipt["spike"]["exit_code"] == 0 and receipt["spike"]["done"]
        assert load(prefix + "job/status.json")["exit_code"] == 0
        assert observed["build_hash"] == receipt["build_hash"]
        assert recipe["status"] == "completed" and admission["status"] == "accepted"
        assert recipe["executable"]["sha256"] == admission["elf_sha256"] == row["elf_sha256"]
        assert sha(data[prefix + "compilation_recipe.json"]) == row["recipe_sha256"]
        assert sha(data[prefix + "admission.json"]) == row["admission_sha256"]
        assert sha(data[prefix + "live_environment.json"]) == row["live_environment_sha256"]
        live = load(prefix + "live_environment.json")
        assert live["environment"]["GEMMINI_MODE"] == "both"
        assert live["environment"]["GEMMINI_PERF_CONFIG"] == "e4m3_vpu"
        assert live["environment"]["GEMMINI_PERF_SET"] == "mem.host_tracking=0"
        assert sha(data[prefix + "device_catalog/mx_catalog.json"]) == row["catalog_sha256"]
        assert sha(data[prefix + "device_catalog/runtime_quant.o"]) == ("edc9fd2f4a600078d02d7ccd5dd553042b7cdda91094cedfd692bd424fc1cda8" if label == "candidate" else "f52f5698f7c9d90b0789329cfe66001c383558c967774aae985adf3235083ae0")
        frozen = "frozen_merlin_quant/" if label == "candidate" else "frozen_control_quant/"
        catalog = load(prefix + "device_catalog/mx_catalog.json")
        assert sha(data[frozen + "runtime_quant.c"]) == catalog["compilation"]["source_sha256"]["runtime_quant"]
        patterns = [
            (r"gemmini perf \[e4m3_vpu\]: (\d+) commands, (\d+) loops, (\d+) fences",
             ["commands", "loops", "fences"]),
            (r"mesh: (\d+) tiles, (\d+) busy cycles; VPU: (\d+) commands",
             ["mesh_tiles", "mesh_busy_cycles", "vpu_commands"]),
            (r"load: (\d+) bytes .*scales: (\d+) bytes", ["load_bytes", "scale_bytes"]),
        ]
        for pattern, keys in patterns:
            matches = re.findall(pattern, data[prefix + "spike.log"].decode())
            assert len(matches) == 1
            for key, count in zip(keys, matches[0], strict=True):
                assert row["performance"][key] == int(count)
        rewrite = load(prefix + "device_host_abi/mx_vpu_elementwise_rewrite.json")
        assert len(rewrite["connected_multiplies"]) == 45
        assert receipt.get("prepare_vpu_broadcast_operands") is True
        assert sha(data[prefix + "device_host_abi/mx_vpu_elementwise.mlir"]) == rewrite["output_sha256"]
        for kind, logits in [("source", source), ("compiler", values[-33:])]:
            for key, value in reader.metrics(reference, logits).items():
                assert abs(row["quality_late_33"][kind][key] - value) < 1e-12
        original, actual = row["quality_late_33"]["source"], row["quality_late_33"]["compiler"]
        assert actual["mean_kl_nats"] <= 1.05 * max(original["mean_kl_nats"], 1e-6)
        assert actual["worst_kl_nats"] <= 1.05 * max(original["worst_kl_nats"], 1e-6)
        assert actual["top1_agreement"] >= original["top1_agreement"]
        if args.build_root is not None:
            name = "onepass160_rms_integer_quant_merlin" if label == "candidate" else "onepass160_rms_prepared"
            image = args.build_root / name / "model.elf"
            assert image.stat().st_size == row["elf_bytes"]
            with image.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == row["elf_sha256"]
        print(label, row["cycles"], "cycles; full words and unchanged quality gate checked")
    assert arrays["baseline"].tobytes() == arrays["candidate"].tobytes()
    assert report["compiler_words_equal"] and report["source_relative_quality_passed"]
    assert report["source_full_logits_available"] is False
    assert report["same_logical_timing_boundary"] and report["uninstrumented_cycle_comparison"]
    assert report["candidate_to_source_cycle_ratio_if_qualified"] == report["records"]["candidate"]["cycles"] / report["source_cycles"]
    print("PASS: all 5,120,000 compiler words identical; source gate covers 33 retained rows.")


if __name__ == "__main__":
    main()
