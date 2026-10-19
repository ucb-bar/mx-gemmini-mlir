"""Gate complete matched compiler forwards against the archived independent source.

The independent source retained 33 full logit rows and 160 top tokens. Report
that scope explicitly; full compiler dumps do not invent missing source rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.capture_matched_tinyllama_onepass import sha
from tools.compare_mx_model_quality import compare_tinyllama
from tools.report_stage_profile import parse_gemmini_perf


def compare(stage, archive, baseline, candidate, out):
    from merlin.runtime.backends.spike_model import parse_console

    if out.exists():
        raise ValueError("comparison output directory must be fresh")
    manifest = json.loads((archive / "matched_archive_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(archive / name) != digest:
            raise ValueError("independent source archive differs: " + name)
    capture = json.loads((stage / "capture_receipt.json").read_text())
    for name in ("model.mlir", "weights.safetensors", "inputs.npz", "golden.npy"):
        if sha(stage / name) != capture["artifacts"][name]["sha256"]:
            raise ValueError("capture differs: " + name)
    identity = json.loads((archive / "input_identity.json").read_text())
    if sha(stage / "weights.safetensors") != identity["compiler_weights_sha256"]:
        raise ValueError("checkpoint differs from independent source comparison")
    with np.load(stage / "inputs.npz", allow_pickle=False) as got:
        ids = got["in0"].reshape(-1)
    with np.load(archive / "input_ids.npz", allow_pickle=False) as expected:
        if not np.array_equal(ids, expected["in0"].reshape(-1)):
            raise ValueError("token IDs differ from independent source comparison")
    golden = np.load(stage / "golden.npy", allow_pickle=False)
    if golden.shape != (1, 160, 32000) or golden.dtype != np.float32:
        raise ValueError("matched forward requires the complete 160-token FP32 reference")
    reference = golden.reshape(160, 32000)
    if not np.array_equal(reference[-33:], np.load(archive / "fp32_late_logits.npy", allow_pickle=False)):
        raise ValueError("FP32 logits differ from independent source reference")
    source_words = np.fromfile(archive / "source_reference_late_bf16.bin", dtype="<u2")
    if source_words.size != 33 * 32000:
        raise ValueError("independent source late logits are incomplete")
    source = (source_words.astype("<u4") << 16).view("<f4").reshape(33, 32000)
    old = json.loads((archive / "matched_onepass_comparison.json").read_text())
    archived_build = json.loads((archive / "compiler_build_summary.json").read_text())
    archived_run = json.loads((archive / "compiler_run.json").read_text())
    old_tops = parse_console((archive / "compiler_spike.log").read_text())["argmax"]
    records, arrays = {}, {}
    for label, build in (("baseline", baseline), ("candidate", candidate)):
        receipt_path = build / "mx_stage_probe.json"
        receipt = json.loads(receipt_path.read_text())
        spike = receipt["spike"]
        validation_path = build / "mx_stage_output_validation.json"
        validation = json.loads(validation_path.read_text())
        if (validation.get("schema") != "mx_gemmini.complete_stage_output_validation.v1" or
                validation.get("validated") is not True or
                validation["original_stage_receipt_sha256"] != sha(receipt_path) or
                validation["capture_model_sha256"] != capture["artifacts"]["model.mlir"]["sha256"] or
                validation["console_sha256"] != sha(build / "spike.log")):
            raise ValueError(label + " complete-output validation differs")
        if (spike["exit_code"] != 0 or
                receipt["profile_sha256"] != archived_build["profile_sha256"] or
                receipt["capture_model_sha256"] != capture["artifacts"]["model.mlir"]["sha256"] or
                receipt["checkpoint_weights_sha256"] != identity["compiler_weights_sha256"] or
                not receipt["raw_selected_source_coverage_complete"] or
                spike["extension_sha256"] != archived_run["extension_sha256"]):
            raise ValueError(label + " build/run identity differs")
        catalog = json.loads((build / "device_catalog/mx_catalog.json").read_text())
        host = receipt.get("host_singleton_matrix_regions", [])
        if (len(catalog["bindings"]) + len(host) != 200 or
                any(binding["target_dtypes"][:2] != ["f8E4M3FN", "f8E4M3FN"]
                    for binding in catalog["bindings"])):
            raise ValueError(label + " matrix source obligations differ")
        observation = json.loads((build / "live_environment.json").read_text())
        env = observation["environment"]
        if (env["GEMMINI_MODE"] != "both" or env["GEMMINI_PERF_CONFIG"] != "e4m3_vpu" or
                env["GEMMINI_PERF_SET"] != "mem.host_tracking=0" or
                Path(observation["command"][-1]).resolve() != Path(receipt["elf"]).resolve()):
            raise ValueError(label + " observed execution environment differs")
        console_path = build / "spike.log"
        parsed = parse_console(console_path.read_text(), expected_output_elements=golden.size)
        values = parsed["outputs"].reshape(reference.shape)
        digest = hashlib.sha256(values.astype("<f4").tobytes()).hexdigest()
        if (parsed["output_dtype"] != "f32" or not np.isfinite(values).all() or
                parsed["metrics"].get("memref_rank_mismatch") != 0 or
                parsed["metrics"].get("build_hash") != receipt["build_hash"] or
                digest != validation["output_digests"]["first_typed"]["sha256"]):
            raise ValueError(label + " full output evidence differs")
        quality = compare_tinyllama(reference[-33:], source, values[-33:])
        records[label] = {
            "cycles": parsed["metrics"]["cycles"], "quality_late_33": quality,
            "complete_output_shape": list(golden.shape), "output_words_sha256": digest,
            "top1_agreement_all_160": int(np.count_nonzero(values.argmax(-1) == reference.argmax(-1))),
            "top1_matches_archived_compiler_all_160": int(np.count_nonzero(values.argmax(-1) == old_tops)),
            "performance": parse_gemmini_perf(Path(env["GEMMINI_PERF_OUT"]).read_text()),
            "stage_receipt_sha256": sha(receipt_path), "console_sha256": sha(console_path),
            "catalog_sha256": sha(build / "device_catalog/mx_catalog.json"),
            "live_environment_sha256": sha(build / "live_environment.json"),
            "complete_output_validation_sha256": sha(validation_path),
            "host_matrix_regions": host,
        }
        arrays[label] = values
    source_cycles = old["source_stage_cycles"]
    qualified = records["candidate"]["quality_late_33"]["passed"]
    report = {"schema": "mx_gemmini.matched_complete_forward_comparison.v1",
              "scope": "two_complete_compiler_forwards_vs_33_archived_independent_source_logit_rows",
              "source_cycles": source_cycles, "records": records,
              "source_relative_quality_passed": qualified,
              "candidate_to_source_cycle_ratio_if_qualified": (
                  records["candidate"]["cycles"] / source_cycles if qualified else None),
              "source_full_logits_available": False,
              "capture_receipt_sha256": sha(stage / "capture_receipt.json"),
              "source_identity_sha256": sha(archive / "input_identity.json")}
    out.mkdir(parents=True)
    for label, values in arrays.items():
        np.save(out / (label + "_full_logits.npy"), values.reshape(golden.shape))
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage", "archive", "baseline", "candidate", "out-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.stage, args.archive, args.baseline, args.candidate, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
