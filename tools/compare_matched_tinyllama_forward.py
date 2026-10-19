"""Gate complete matched compiler forwards against the archived independent source.

The independent source retained 33 full logit rows and 160 top tokens. Report
that scope explicitly; full compiler dumps do not invent missing source rows.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.capture_matched_tinyllama_onepass import sha
from tools.compare_mx_model_quality import compare_tinyllama
from tools.compare_fair_tinyllama_onepass import rows, unique
from tools.report_stage_profile import parse_gemmini_perf


def read_equal_boundary(archive, boundary):
    """Reopen the token-lookup control rather than the older embedding-only timer."""
    old = json.loads((archive / "matched_onepass_comparison.json").read_text())
    identity = json.loads((archive / "input_identity.json").read_text())
    prior_run = json.loads((archive / "compiler_run.json").read_text())
    report = json.loads((boundary / "fair_comparison.json").read_text())
    lookup = json.loads((boundary / "lookup_receipt.json").read_text())
    build = json.loads((boundary / "build_receipt.json").read_text())
    run = json.loads((boundary / "run_receipt.json").read_text())
    source = gzip.decompress((boundary / "source_spike.log.gz").read_bytes())
    original = gzip.decompress((archive / "source_timed.log.gz").read_bytes())
    if (report.get("schema") != "mx_gemmini.tinyllama_onepass_equal_boundary.v1"
            or not report.get("same_logical_input_output_boundary")
            or not report.get("same_token_ids_and_fp32_weights")
            or report["lookup_receipt_sha256"] != sha(boundary / "lookup_receipt.json")
            or report["source_build_receipt_sha256"] != sha(boundary / "build_receipt.json")
            or report["source_run_receipt_sha256"] != sha(boundary / "run_receipt.json")
            or run["build_receipt_sha256"] != sha(boundary / "build_receipt.json")
            or hashlib.sha256(source).hexdigest() != run["log_sha256"]
            or run["log_sha256"] != report["source_log_sha256"]
            or hashlib.sha256(original).hexdigest() != old["sha256"]["source_log"]
            or rows(source.decode()) != rows(original.decode())
            or "llama MODEL benchmark finished (22 layer(s) + lm_head)." not in source.decode()
            or lookup["bf16_mismatches"] != 0 or lookup["matched_bf16_elements"] != 160 * 2048
            or lookup["token_ids_sha256"] != identity["token_ids_sha256"]
            or lookup["compiler_weights_sha256"] != identity["compiler_weights_sha256"]
            or lookup["source_activations_sha256"] != identity["source_activation_sha256"]
            or build["source_blob_sha256"] != old["sha256"]["source_blob"]
            or build["source_header_sha256"] != old["sha256"]["source_header"]
            or build["source_sha256"] != lookup["lookup_source_sha256"]
            or build["embedding_table_sha256"] != lookup["embedding_table_sha256"]
            or build["elf_sha256"] != report["source_elf_sha256"]
            or run["extension_sha256"] != prior_run["extension_sha256"]
            or run["spike_sha256"] != prior_run["spike_sha256"]
            or run["spike_environment"] != {"GEMMINI_MODE": "both",
                "GEMMINI_PERF_CONFIG": "e4m3_vpu", "GEMMINI_PERF_SET": "mem.host_tracking=0"}
            or run["emitted_logit_rows"] != 33 or run["emitted_top_tokens"] != 160):
        raise ValueError("token-to-logits source control differs from retained source evidence")
    cycles = int(unique(r"^stage_cycles64 0x([0-9a-f]{16})$", source.decode()), 16)
    if cycles != run["stage_cycles"] or cycles != report["source_stage_cycles"]:
        raise ValueError("token-to-logits source cycles differ from the actual console")
    return {"cycles": cycles, "boundary": "token_ids_to_logits_including_embedding_lookup",
            "fair_comparison_sha256": sha(boundary / "fair_comparison.json"),
            "source_log_sha256": run["log_sha256"],
            "source_run_receipt_sha256": sha(boundary / "run_receipt.json"),
            "source_elf_sha256": build["elf_sha256"],
            "source_full_logits_available": False}


def compare(stage, archive, baseline, candidate, out, *, boundary_archive=None):
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
    if not np.array_equal(reference.argmax(-1), np.load(archive / "fp32_argmax_160.npy", allow_pickle=False)):
        raise ValueError("complete FP32 top tokens differ from the retained reference")
    source_words = np.fromfile(archive / "source_reference_late_bf16.bin", dtype="<u2")
    if source_words.size != 33 * 32000:
        raise ValueError("independent source late logits are incomplete")
    source = (source_words.astype("<u4") << 16).view("<f4").reshape(33, 32000)
    boundary = read_equal_boundary(archive, boundary_archive) if boundary_archive else None
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
        elf = build / "model.elf"
        elf_digest = sha(elf)
        recipe = json.loads((build / "compilation_recipe.json").read_text())
        admissions = list(build.glob("invocations/*/policy/admission.json"))
        if (recipe["status"] != "completed" or recipe["executable"]["sha256"] != elf_digest
                or len(admissions) != 1):
            raise ValueError(label + " final executable or completed recipe differs")
        admission = json.loads(admissions[0].read_text())
        extension_paths = [Path(arg.removeprefix("--extlib=")) for arg in spike["command"]
                           if arg.startswith("--extlib=")]
        if (admission["status"] != "accepted" or admission["elf_sha256"] != elf_digest
                or len(extension_paths) != 1 or sha(extension_paths[0]) != spike["extension_sha256"]
                or sha(Path(spike["command"][0])) != archived_run["spike_sha256"]
                or spike["command"] != observation["command"]):
            raise ValueError(label + " admitted image or observed simulator identity differs")
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
            "performance": parse_gemmini_perf(
                Path(env["GEMMINI_PERF_OUT"]).read_text() if env.get("GEMMINI_PERF_OUT")
                else console_path.read_text()),
            "performance_counter_source": env.get("GEMMINI_PERF_OUT") or str(console_path),
            "stage_receipt_sha256": sha(receipt_path), "console_sha256": sha(console_path),
            "catalog_sha256": sha(build / "device_catalog/mx_catalog.json"),
            "live_environment_sha256": sha(build / "live_environment.json"),
            "complete_output_validation_sha256": sha(validation_path),
            "host_matrix_regions": host,
            "elf_sha256": elf_digest, "elf_bytes": elf.stat().st_size,
            "recipe_sha256": sha(build / "compilation_recipe.json"),
            "admission_sha256": sha(admissions[0]), "arena_estimate": receipt["arena_estimate"],
        }
        arrays[label] = values
    source_cycles = boundary["cycles"] if boundary else None
    qualified = records["candidate"]["quality_late_33"]["passed"]
    report = {"schema": "mx_gemmini.matched_complete_forward_comparison.v1",
              "scope": "two_complete_compiler_forwards_vs_33_archived_independent_source_logit_rows",
              "source_cycles": source_cycles, "records": records,
              "source_boundary": boundary,
              "same_logical_timing_boundary": boundary is not None,
              "source_relative_quality_passed": qualified,
              "candidate_to_source_cycle_ratio_if_qualified": (
                  records["candidate"]["cycles"] / source_cycles if qualified and boundary else None),
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
    parser.add_argument("--boundary-archive", required=True, type=Path,
                        help="retained token-lookup source control with the same timing boundary")
    args = parser.parse_args()
    print(json.dumps(compare(args.stage, args.archive, args.baseline, args.candidate, args.out_dir,
                             boundary_archive=args.boundary_archive), indent=2))


if __name__ == "__main__":
    main()
