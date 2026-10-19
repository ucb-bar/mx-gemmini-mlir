"""Evaluate explicit-policy cached execution against its captured FP32 trajectory.

No independent cached reference execution is available. This reports FP32 drift
and preparation-inclusive modeled costs without inventing a relative gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.capture_matched_tinyllama_onepass import sha
from tools.compare_mx_model_quality import _kl
from tools.compare_tinyllama_session import metrics
from tools.link_merlin_session import _projected_stage_entry
from tools.report_stage_profile import parse_gemmini_perf
from tools.run_linked_merlin_session import (
    observe_preparation_output,
    observe_session_output,
    validate_admission_record,
)


def verify_vpu_policy_bindings(manifest, resolved_policy, receipt):
    from mx_gemmini_support.execution_policy import reconcile_vpu_region_sets

    available = {name: set(manifest[f"vpu_{name}"]["source_regions"])
                 for name in resolved_policy["enabled_vpu_regions"]}
    routed, aliases = reconcile_vpu_region_sets(available, resolved_policy["enabled_vpu_regions"])
    if (any(set(regions) != available[name] for name, regions in routed.items())
            or {name: sorted(regions) for name, regions in routed.items()} != manifest["resolved_vpu_regions"]
            or aliases != manifest["vpu_source_aliases"]
            or any(receipt[f"vpu_{name}_lowered_regions"] != len(regions) for name, regions in routed.items())):
        raise ValueError("compiled VPU obligations differ from selected source policy")


def verify_capture_and_builds(session_root, reference_archive, builds, policies, profile_path, link):
    """Reopen the fresh capture, original tokens/checkpoint and compiled inputs."""
    from m2m.capture.bundle_integrity import verify_bundle_integrity
    from merlin.llvmlower.session_bundle import load

    from mx_gemmini_support.execution_policy import load_policy
    from mx_gemmini_support.target_profile import load_profile, profile_sha256

    verify_bundle_integrity(session_root)
    session = load(session_root)
    if [(p.name, p.steps) for p in session.programs] != [("prefill", 1), ("decode", 32)]:
        raise ValueError("matched cached evaluation requires one prefill and32 decodes")
    capture = json.loads((session_root.parent / "receipt.json").read_text())
    identity = json.loads((reference_archive / "input_identity.json").read_text())
    archive_manifest = json.loads((reference_archive / "matched_archive_manifest.json").read_text())
    if (capture.get("schema") != "mx_gemmini.matched_cached_session_capture.v1"
            or archive_manifest["files"]["input_identity.json"] != sha(reference_archive / "input_identity.json")
            or capture["reference_identity_sha256"] != sha(reference_archive / "input_identity.json")
            or capture["weights_sha256"] != identity["compiler_weights_sha256"]):
        raise ValueError("fresh session differs from the matched checkpoint identity")
    with np.load(reference_archive / "input_ids.npz", allow_pickle=False) as archive:
        ids = archive["in0"].reshape(-1)
    with np.load(session.programs[0].bundle / "inputs.npz", allow_pickle=False) as archive:
        prefix = archive["in0"]
    with np.load(session.programs[1].bundle / "session_inputs.npz", allow_pickle=False) as archive:
        tokens = archive["decode_tokens"]
    if (ids.shape != (160,) or ids.dtype != np.int64 or prefix.shape != (1, 128)
            or hashlib.sha256(ids.tobytes()).hexdigest() != identity["token_ids_sha256"]
            or tokens.shape != (32, 1, 1) or prefix.dtype != ids.dtype or tokens.dtype != ids.dtype
            or not np.array_equal(prefix.reshape(-1), ids[:128])
            or not np.array_equal(tokens.reshape(-1), ids[128:])):
        raise ValueError("cached session token stream differs from matched160 tokens")
    expected_programs = [(p.name, p.steps, len(p.outputs)) for p in session.programs]
    recorded = [(p["name"], p["steps"], p["outputs"]) for p in link["programs"]]
    if (link.get("schema") != "mx_gemmini.linked_merlin_session.v1"
            or Path(link["session_root"]).resolve() != session_root.resolve()
            or recorded != expected_programs
            or link["bindings"] != [binding.__dict__ for binding in session.bindings]):
        raise ValueError("linked program schedule or recurrent state routes differ from capture")
    profile = load_profile(profile_path)
    if profile_sha256(profile) != link["profile_sha256"]:
        raise ValueError("linked target profile differs from the selected profile")
    records = {}
    for program, linked in zip(session.programs, link["programs"], strict=True):
        name, bundle, build = program.name, program.bundle, builds[program.name].resolve(strict=True)
        receipt_path = build / "mx_stage_probe.json"
        receipt = json.loads(receipt_path.read_text())
        policy, resolved_policy = load_policy(policies[name], (bundle / "model.mlir").read_text(), profile)
        checkpoint = sha(bundle / "weights.safetensors")
        model_sha = sha(bundle / "model.mlir")
        if (receipt.get("schema") != "mx_gemmini.merlin_stage_probe.v1"
                or receipt.get("status") != "linked_policy_numeric_qualification_pending"
                or receipt.get("capture_model_sha256") != model_sha
                or linked["capture_model_sha256"] != model_sha
                or capture["programs"][name]["model_sha256"] != model_sha
                or capture["programs"][name]["receipt_sha256"] != sha(bundle / "capture_receipt.json")
                or receipt.get("checkpoint_weights_sha256") != checkpoint
                or checkpoint != identity["compiler_weights_sha256"]
                or receipt.get("profile_sha256") != link["profile_sha256"]
                or receipt.get("execution_policy_sha256") != sha(policies[name])
                or receipt.get("execution_policy") != policy
                or receipt.get("raw_selected_source_coverage_complete") is not True
                or receipt.get("op_profile") or receipt.get("diagnostic_host_vpu")):
            raise ValueError(name + " build differs from the captured source or selected policy")
        manifest_path = Path(receipt["catalog_manifest"])
        if manifest_path.resolve() != (build / "device_catalog/mx_catalog.json").resolve():
            raise ValueError(name + " catalog is outside the selected build")
        manifest = json.loads(manifest_path.read_text())
        verify_vpu_policy_bindings(manifest, resolved_policy, receipt)
        active = {region: fmt for region, fmt in policy["matrix_formats"].items() if fmt != "host"}
        bindings = manifest["bindings"]
        if (manifest.get("coverage_complete") is not True or manifest.get("unbound_precision_proposals")
                or manifest["profile_sha256"] != link["profile_sha256"]
                or len(bindings) != len(active) or {b["region"] for b in bindings} != set(active)
                or receipt["selected_operations"] != len(active) or linked["selected_operations"] != len(active)
                or any(b["target_dtypes"][:2] != [
                    "f8E4M3FN" if active[b["region"]] == "mxfp8" else "f4E2M1FN"] * 2 for b in bindings)):
            raise ValueError(name + " compiled catalog differs from selected matrix obligations")
        objects = link["stage_object_sha256"][name]
        if set(objects) != {"model.o", "device/device_catalog_shim.o", "device_catalog/mx_catalog.o"}:
            raise ValueError(name + " linked object set is incomplete")
        for relative, digest in objects.items():
            if sha(build / relative) != digest:
                raise ValueError(name + " object differs from linked session: " + relative)
        if manifest["compilation"]["object_sha256"] != objects["device_catalog/mx_catalog.o"]:
            raise ValueError(name + " catalog object differs from its compilation")
        projection = _projected_stage_entry(program, build, receipt)
        if projection is not None:
            saved = json.loads((build / "entry_weight_projection/entry_weight_projection.json").read_text())
            expected = {"argument_table_sha256": saved["argument_table_sha256"],
                        "original_arguments": len(saved["original_to_projected"]),
                        "original_to_projected": saved["original_to_projected"],
                        "removed_original_indices": saved["removed_original_indices"],
                        "retained_arguments": len(saved["retained_original_indices"])}
            if link["entry_projections"].get(name) != expected:
                raise ValueError(name + " linked entry projection differs from rederived proof")
        elif name in link.get("entry_projections", {}):
            raise ValueError(name + " has an unselected linked entry projection")
        records[name] = {"build_receipt_sha256": sha(receipt_path), "policy_sha256": sha(policies[name]),
                         "catalog_sha256": sha(manifest_path), "source_sha256": model_sha,
                         "selected_matrix_operations": len(active), "host_matrix_operations": len(policy["matrix_formats"]) - len(active),
                         "prepacked_checkpoint_kernels": receipt.get("prepacked_checkpoint_kernels", 0),
                         "connected_multiply_pairs": len(receipt.get("connected_vpu_multiply_pairs", [])),
                         "connected_sigmoid_chains": len(receipt.get("connected_vpu_sigmoid_multiply_chains", [])),
                         "shared_activation_packing": receipt.get("share_activation_packing", False),
                         "matrix_vpu_transport": receipt.get("matrix_vpu_transport", False)}
    return session, records


def verify_completed_run(run_path, link_path, session):
    """Reparse all native records and reopen actual admitted image and full taps."""
    from merlin.llvmlower.session_outputs import (
        assembled_output_bytes,
        observed_output_dump_bytes,
    )

    run, link = (json.loads(path.read_text()) for path in (run_path, link_path))
    if (run.get("schema") != "mx_gemmini.merlin_session_spike_run.v1"
            or run.get("complete") is not True or run.get("exit_code") != 0
            or run.get("status") != "complete_numeric_qualification_pending"
            or run.get("link_receipt_sha256") != sha(link_path)
            or run.get("timing_config") != "e4m3_vpu" or run.get("timing_mode") != "both"
            or run.get("timing_overrides") != "mem.host_tracking=0"
            or run.get("timing_model_active") is not True
            or link.get("preparation_performance_schema") != "spike_rdcycle_rdinstret_reset_prepare.v1"
            or run.get("preparation_performance_schema") != link["preparation_performance_schema"]
            or run.get("performance_schema") != "spike_rdcycle_rdinstret_per_step.v1"
            or run.get("performance_schema") != link.get("performance_schema")
            or link.get("execution_timing_scope") != "reset_plus_program_state_handoff_plus_existing_step_intervals; console_and_phase_bookkeeping_excluded"
            or run.get("execution_timing_scope") != link.get("execution_timing_scope")):
        raise ValueError("evaluation needs a complete measured session with reset and state handoff")
    console_path = run_path.parent / "spike.log"
    if sha(console_path) != run["log_sha256"]:
        raise ValueError("session console changed after its receipt")
    console = console_path.read_text()
    if "gemmini: GEMMINI_MODE=both" not in console:
        raise ValueError("session console does not confirm selected timing mode")
    observation = observe_session_output(console, session, exit_code=0, require_perf=True)
    preparation = observe_preparation_output(console, session, observation)
    if any(run.get(key) != value for key, value in (observation | preparation).items()):
        raise ValueError("session performance or output records differ from actual console")
    command = run["command"]
    live = json.loads((run_path.parent / "live_environment.json").read_text())
    environment = live["environment"]
    extension = [Path(arg.removeprefix("--extlib=")) for arg in command if arg.startswith("--extlib=")]
    elf = Path(link["elf"])
    if (command != live["command"] or Path(command[-1]).resolve() != elf.resolve()
            or sha(Path(command[0])) != run["spike_sha256"] or len(extension) != 1
            or sha(extension[0]) != run["extension_sha256"]
            or environment.get("GEMMINI_MODE") != "both"
            or environment.get("GEMMINI_PERF_CONFIG") != "e4m3_vpu"
            or environment.get("GEMMINI_PERF_SET") != "mem.host_tracking=0"
            or elf.stat().st_size != link["elf_bytes"] or sha(elf) != link["elf_sha256"]):
        raise ValueError("observed simulator, settings or linked image changed")
    if not link.get("linked_elf_admission_required"):
        raise ValueError("session lacks explicit final-image admission")
    validate_admission_record(link)
    perf_path = Path(environment["GEMMINI_PERF_OUT"])
    if sha(perf_path) != run["timing_report_sha256"]:
        raise ValueError("timing report changed after the run")
    taps = [{"program": p, "name": name, "output": 0, "bytes": 128000}
            for p, name in enumerate(("prefill", "decode"))]
    if link.get("dump_outputs") != taps or link.get("dump_limit_bytes") != 128000:
        raise ValueError("evaluation needs every full vocabulary logit row")
    console_values = observed_output_dump_bytes(console, session, taps, observation["output_digests"], byte_limit=128000)
    dumps = run.get("output_dumps", [])
    if len(dumps) != 33 or {(r["program"], r["step"], r["output"]) for r in dumps} != set(console_values):
        raise ValueError("session must retain all33 complete logit dumps")
    rows = []
    for p, program in enumerate(session.programs):
        for step, raw in enumerate(assembled_output_bytes(session, run_path.parent, dumps, program=program.name, output=0)):
            if raw != console_values[p, step, 0]:
                raise ValueError("retained full output differs from native console")
            rows.append(np.frombuffer(raw, dtype="<f4").reshape(1, 32000))
    perf = parse_gemmini_perf(perf_path.read_text())
    if perf.get("mode") != "e4m3_vpu" or not perf.get("commands"):
        raise ValueError("session lacks the selected extension performance counters")
    return run, np.concatenate(rows), perf


def trajectory_metrics(actual, reference):
    if actual.shape != reference.shape or actual.shape != (33, 32000):
        raise ValueError("cached metrics require all33 equal full vocabulary rows")
    per_row = [{"program": "prefill" if i == 0 else "decode", "step": max(0, i - 1),
                **metrics(a.reshape(1, -1), b.reshape(1, -1))} for i, (a, b) in enumerate(zip(actual, reference, strict=True))]
    kl = _kl(reference.astype(np.float64), actual.astype(np.float64))
    for row, divergence in zip(per_row, kl, strict=True):
        row["kl_nats"] = float(divergence)
    return {"mean_kl_nats": float(kl.mean()), "worst_kl_nats": float(kl.max()),
            "top_token_matches": sum(row["top_token_equal"] for row in per_row), "positions": 33,
            "rows": per_row}


def evaluate(run_path, link_path, session_root, reference_archive, builds, policies, profile_path):
    from merlin.common.yaml import load_yaml

    link = json.loads(link_path.read_text())
    session, stages = verify_capture_and_builds(session_root, reference_archive, builds, policies, profile_path, link)
    run, actual, perf = verify_completed_run(run_path, link_path, session)
    prefill_path = session.programs[0].bundle / "golden.npy"
    decode_path = session.programs[1].bundle / "session_quality_fp32.npz"
    prefill = np.load(prefill_path, allow_pickle=False)
    child = load_yaml(session.programs[1].bundle / "session_contract.yaml")
    with np.load(decode_path, allow_pickle=False) as archive:
        decode = archive["logits"]
    if (prefill.shape != (1, 32000) or decode.shape != (32, 1, 32000)
            or prefill.dtype != np.float32 or decode.dtype != np.float32
            or child["quality"]["reference"] != "eager_fp32"
            or child["quality"]["reference_sha256"] != hashlib.sha256(decode.tobytes()).hexdigest()):
        raise ValueError("captured FP32 trajectory differs from its shape or contract digest")
    reference = np.concatenate((prefill, decode[:, 0]))
    return {"schema": "mx_gemmini.matched_cached_session_evaluation.v1",
            "scope": "128_token_prefill_plus32_cached_decodes_complete_logits_and_preparation_inclusive_cost",
            "reference_kind": "captured_fp32_recurrent_trajectory",
            "numerical_qualification": "diagnostic_only_no_independent_cached_source_gate",
            "independent_source_speedup": None,
            "run_receipt_sha256": sha(run_path), "link_receipt_sha256": sha(link_path),
            "live_environment_sha256": sha(run_path.parent / "live_environment.json"),
            "console_sha256": run["log_sha256"], "elf_sha256": link["elf_sha256"],
            "timing_report_sha256": run["timing_report_sha256"],
            "capture_receipt_sha256": sha(session_root.parent / "receipt.json"),
            "bundle_integrity_sha256": sha(session_root / "bundle_integrity.json"),
            "prefill_reference_sha256": sha(prefill_path), "decode_reference_sha256": sha(decode_path),
            "profile_sha256": link["profile_sha256"], "stages": stages,
            "quality": trajectory_metrics(actual, reference), "perf_counters_extension_scope": perf,
            "step_performance": run["step_performance"], "program_performance": run["program_performance"],
            "step_total_performance": run["total_performance"],
            "preparation_performance": run["preparation_performance"],
            "execution_total_performance": run["execution_total_performance"],
            "execution_timing_scope": run["execution_timing_scope"],
            "resources": {key: link[key] for key in ("elf_bytes", "arena_layout", "shared_weight_bytes", "static_malloc_estimates")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-receipt", "link-receipt", "session-root", "reference-archive", "prefill-build",
                 "decode-build", "prefill-policy", "decode-policy", "profile", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("--out must be fresh")
    report = evaluate(args.run_receipt, args.link_receipt, args.session_root, args.reference_archive,
                      {"prefill": args.prefill_build, "decode": args.decode_build},
                      {"prefill": args.prefill_policy, "decode": args.decode_policy}, args.profile)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cycles": report["execution_total_performance"]["cycles"],
                      "quality": {k: v for k, v in report["quality"].items() if k != "rows"}}))


if __name__ == "__main__":
    main()
