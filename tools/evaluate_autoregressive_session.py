"""Audit a complete captured decoder session and report floating-reference error.

Numerical metrics are diagnostics. This tool supplies neither a task-accuracy
threshold nor a speedup against an independent implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tools.compare_mx_model_quality import _kl
from tools.report_stage_profile import parse_gemmini_perf
from tools.run_linked_merlin_session import (
    observe_preparation_output,
    observe_session_output,
    validate_admission_record,
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_state(rows, *, vocabulary, cache_shape, prefill_tokens, decode_steps):
    """Check every vocabulary extent, K/V update and per-layer position."""
    if (
        len(cache_shape) != 6
        or cache_shape[0] != 2
        or cache_shape[2] != 1
        or cache_shape[4] != prefill_tokens + decode_steps
        or min(cache_shape) <= 0
    ):
        raise ValueError("expected complete batch-one K/V state")
    positions = [(0, 0, prefill_tokens)] + [
        (1, step, prefill_tokens + step + 1) for step in range(decode_steps)
    ]
    selected = {(r["program"], r["step"], r["output"]): r for r in rows}
    wanted = {(p, s, o) for p, s, _ in positions for o in range(3)}
    if len(selected) != len(rows) or set(selected) != wanted:
        raise ValueError("missing or duplicated recurrent output")
    caches = []
    for program, step, position in positions:
        logits, cache, lengths = [selected[program, step, o] for o in range(3)]
        expected = np.full(cache_shape[1], position, dtype="<i8").tobytes()
        if logits["bytes"] != vocabulary * 4:
            raise ValueError("incomplete vocabulary output")
        if cache["bytes"] != math.prod(cache_shape) * 4:
            raise ValueError("incomplete K/V state")
        if (
            lengths["bytes"] != len(expected)
            or lengths["sha256"] != hashlib.sha256(expected).hexdigest()
        ):
            raise ValueError("incorrect per-layer cache positions")
        caches.append(cache["sha256"])
    if len(set(caches)) != len(positions):
        raise ValueError("K/V state did not change at every append")
    return {
        "position_vectors_checked": len(positions),
        "distinct_complete_kv_states": len(caches),
        "final_position_per_layer": prefill_tokens + decode_steps,
    }


def metrics(actual, reference):
    if (
        actual.ndim != 2
        or actual.shape != reference.shape
        or actual.shape[0] < 2
        or actual.shape[1] < 2
        or actual.dtype != np.float32
        or reference.dtype != np.float32
        or not np.isfinite(actual).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("expected equal complete finite F32 vocabulary rows")
    got, expected = actual.astype(np.float64), reference.astype(np.float64)
    kl = _kl(expected, got)
    cosine = np.sum(got * expected, axis=1) / np.maximum(
        np.linalg.norm(got, axis=1) * np.linalg.norm(expected, axis=1), 1e-300
    )
    top = np.argmax(got, axis=1) == np.argmax(expected, axis=1)
    return {
        "mean_kl_nats": float(kl.mean()),
        "worst_kl_nats": float(kl.max()),
        "minimum_logit_cosine": float(np.clip(cosine.min(), -1, 1)),
        "top_token_matches": int(top.sum()),
        "positions": len(kl),
        "vocabulary": actual.shape[1],
        "all_logits_finite": True,
        "per_position_kl_nats": kl.tolist(),
        "per_position_top_token_equal": top.tolist(),
    }


def verify_target_identity(identity, *, profile, spike, extension):
    if (
        identity["profile_sha256"] != profile
        or identity["spike"]["sha256"] != spike
        or identity["extension"]["sha256"] != extension
    ):
        raise ValueError("execution differs from the predeclared simulator identity")


def evaluate(
    run_path, link_path, capture_root, profile_path, builds, policies, target_identity
):
    from m2m.capture.bundle_integrity import verify_bundle_integrity
    from merlin.llvmlower.session_bundle import load
    from merlin.llvmlower.session_outputs import (
        assembled_output_bytes,
        observed_output_dump_bytes,
    )

    from mx_gemmini_support.execution_policy import load_policy
    from mx_gemmini_support.target_profile import load_profile, profile_sha256

    verify_bundle_integrity(capture_root)
    session = load(capture_root)
    if session.program_names != ("prefill", "decode"):
        raise ValueError("expected a prefill/decode session")
    if set(builds) != set(session.program_names) or set(policies) != set(
        session.program_names
    ):
        raise ValueError("build and policy rosters must match the complete session")
    prefill, decode = session.programs
    if prefill.steps != 1 or decode.steps != 32 or prefill.outputs != decode.outputs:
        raise ValueError("expected one prefill and 32 compatible recurrent decodes")
    (shape, dtype), (cache_shape, cache_dtype), (positions, position_dtype) = (
        prefill.outputs
    )
    if (
        len(shape) != 2
        or shape[0] != 1
        or dtype != "f32"
        or cache_dtype != "f32"
        or positions != [cache_shape[1]]
        or position_dtype != "i64"
    ):
        raise ValueError("unsupported autoregressive output ABI")
    vocabulary = shape[1]
    with np.load(prefill.bundle / "inputs.npz", allow_pickle=False) as inputs:
        tokens = inputs["in0"]
    if tokens.shape != (1, 128) or tokens.dtype != np.int64:
        raise ValueError("expected the captured 128-token prefill")
    run, link = (json.loads(p.read_text()) for p in (run_path, link_path))
    profile = load_profile(profile_path)
    if (
        Path(link["session_root"]).resolve() != capture_root.resolve()
        or link["profile_sha256"] != profile_sha256(profile)
        or link["bindings"] != [b.__dict__ for b in session.bindings]
        or [(p["name"], p["steps"]) for p in link["programs"]]
        != [(p.name, p.steps) for p in session.programs]
    ):
        raise ValueError("linked session differs from its capture or target")
    stage_checks = {}
    for program, linked in zip(session.programs, link["programs"], strict=True):
        build, policy_path = builds[program.name], policies[program.name]
        probe = json.loads((build / "mx_stage_probe.json").read_text())
        policy, resolved = load_policy(
            policy_path, (program.bundle / "model.mlir").read_text(), profile
        )
        if (
            probe["capture_model_sha256"] != sha(program.bundle / "model.mlir")
            or linked["capture_model_sha256"] != probe["capture_model_sha256"]
            or probe["execution_policy"] != policy
            or probe["execution_policy_sha256"] != sha(policy_path)
            or probe["checkpoint_weights_sha256"]
            != sha(program.bundle / "weights.safetensors")
            or not probe["raw_selected_source_coverage_complete"]
            or probe.get("op_profile")
            or probe.get("diagnostic_host_vpu")
        ):
            raise ValueError("compiled stage differs from its captured policy")
        from tools.evaluate_matched_tinyllama_session import verify_vpu_policy_bindings

        catalog_path = build / "device_catalog/mx_catalog.json"
        catalog = json.loads(catalog_path.read_text())
        active = {r: f for r, f in policy["matrix_formats"].items() if f != "host"}
        bindings = catalog["bindings"]
        if (
            not catalog.get("coverage_complete")
            or catalog.get("unbound_precision_proposals")
            or catalog["profile_sha256"] != link["profile_sha256"]
            or catalog["checkpoint_weights_sha256"]
            != probe["checkpoint_weights_sha256"]
            or {b["region"] for b in bindings} != set(active)
            or len(bindings) != len(active)
            or linked["selected_operations"] != len(active)
            or probe["selected_operations"] != len(active)
            or any(
                b["target_dtypes"][:2]
                != ["f8E4M3FN" if active[b["region"]] == "mxfp8" else "f4E2M1FN"] * 2
                for b in bindings
            )
        ):
            raise ValueError(
                "actual matrix catalog differs from its selected precision policy"
            )
        verify_vpu_policy_bindings(catalog, resolved, probe)
        objects = link["stage_object_sha256"][program.name]
        if set(objects) != {
            "model.o",
            "device/device_catalog_shim.o",
            "device_catalog/mx_catalog.o",
        }:
            raise ValueError("incomplete linked stage object set")
        if any(sha(build / name) != digest for name, digest in objects.items()):
            raise ValueError("linked stage objects changed")
        if (
            catalog["compilation"]["object_sha256"]
            != objects["device_catalog/mx_catalog.o"]
        ):
            raise ValueError("catalog object differs from its recorded compilation")
        stage_checks[program.name] = {
            "probe_sha256": sha(build / "mx_stage_probe.json"),
            "selected_operations": probe["selected_operations"],
            "execution_policy_sha256": sha(policy_path),
        }
    if (
        run.get("complete") is not True
        or run.get("exit_code") != 0
        or run.get("status") != "complete_numeric_qualification_pending"
        or run.get("link_receipt_sha256") != sha(link_path)
        or run.get("timing_config") != "e4m3_vpu"
        or run.get("timing_mode") != "both"
        or run.get("timing_overrides") != "mem.host_tracking=0"
        or not run.get("timing_model_active")
        or not link.get("linked_elf_admission_required")
        or run.get("performance_schema") != "spike_rdcycle_rdinstret_per_step.v1"
        or link.get("performance_schema") != run.get("performance_schema")
        or run.get("preparation_performance_schema")
        != "spike_rdcycle_rdinstret_reset_prepare.v1"
        or link.get("preparation_performance_schema")
        != run.get("preparation_performance_schema")
        or link.get("execution_timing_scope")
        != "reset_plus_program_state_handoff_plus_existing_step_intervals; console_and_phase_bookkeeping_excluded"
        or run.get("execution_timing_scope") != link.get("execution_timing_scope")
    ):
        raise ValueError("expected a complete admitted execution with pinned timing")
    verify_target_identity(
        json.loads(target_identity.read_text()),
        profile=link["profile_sha256"],
        spike=run["spike_sha256"],
        extension=run["extension_sha256"],
    )
    validate_admission_record(link)
    elf = Path(link["elf"])
    if elf.stat().st_size != link["elf_bytes"] or sha(elf) != link["elf_sha256"]:
        raise ValueError("measured ELF differs from the linked image")
    log_path = run_path.parent / "spike.log"
    if sha(log_path) != run["log_sha256"]:
        raise ValueError("measured console changed")
    console = log_path.read_text()
    observation = observe_session_output(
        console, session, exit_code=0, require_perf=True
    )
    preparation = observe_preparation_output(console, session, observation)
    if any(run.get(k) != v for k, v in (observation | preparation).items()):
        raise ValueError("recorded outputs or timing disagree with the console")
    live = json.loads((run_path.parent / "live_environment.json").read_text())
    command, env = live["command"], live["environment"]
    extension = [
        Path(s.removeprefix("--extlib=")) for s in command if s.startswith("--extlib=")
    ]
    if (
        command != run["command"]
        or Path(command[-1]).resolve() != elf.resolve()
        or sha(command[0]) != run["spike_sha256"]
        or len(extension) != 1
        or sha(extension[0]) != run["extension_sha256"]
        or env.get("GEMMINI_MODE") != "both"
        or env.get("GEMMINI_PERF_CONFIG") != "e4m3_vpu"
        or env.get("GEMMINI_PERF_SET") != "mem.host_tracking=0"
    ):
        raise ValueError("measured simulator or execution settings changed")
    perf_path = Path(env["GEMMINI_PERF_OUT"])
    if sha(perf_path) != run["timing_report_sha256"]:
        raise ValueError("measured performance counters changed")
    taps = [
        {"program": i, "name": p.name, "output": 0, "bytes": vocabulary * 4}
        for i, p in enumerate(session.programs)
    ]
    if (
        link.get("dump_outputs") != taps
        or link.get("dump_limit_bytes") != vocabulary * 4
    ):
        raise ValueError("expected every full vocabulary row")
    dumped = observed_output_dump_bytes(
        console, session, taps, observation["output_digests"], byte_limit=vocabulary * 4
    )
    rows = []
    for i, program in enumerate(session.programs):
        for step, raw in enumerate(
            assembled_output_bytes(
                session,
                run_path.parent,
                run["output_dumps"],
                program=program.name,
                output=0,
            )
        ):
            if raw != dumped[i, step, 0]:
                raise ValueError("retained logits differ from the actual console")
            rows.append(np.frombuffer(raw, dtype="<f4").reshape(1, vocabulary))
    actual = np.concatenate(rows)
    with np.load(
        decode.bundle / "session_quality_fp32.npz", allow_pickle=False
    ) as golden:
        reference = np.concatenate(
            [
                np.load(prefill.bundle / "golden.npy", allow_pickle=False),
                golden["logits"].reshape(32, vocabulary),
            ]
        )
    state = check_state(
        observation["output_digests"],
        vocabulary=vocabulary,
        cache_shape=cache_shape,
        prefill_tokens=128,
        decode_steps=32,
    )
    return {
        "schema": "mx_gemmini.autoregressive_session_audit.v1",
        "status": "complete_execution_audited_numeric_diagnostics_only",
        "independent_baseline_qualified": False,
        "hardware_qualified": False,
        "run_receipt_sha256": sha(run_path),
        "link_receipt_sha256": sha(link_path),
        "elf_sha256": link["elf_sha256"],
        "profile_sha256": link["profile_sha256"],
        "target_identity_sha256": sha(target_identity),
        "checkpoint": session.raw.get("provenance", {}).get("checkpoint"),
        "inclusive_performance": preparation["execution_total_performance"],
        "stage_checks": stage_checks,
        "state_checks": state,
        "logits_checked": int(actual.size),
        "quality": metrics(actual, reference),
        "accelerator_counters": parse_gemmini_perf(perf_path.read_text()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "run-receipt",
        "link-receipt",
        "capture",
        "profile",
        "target-identity",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--stage-build", action="append", required=True)
    parser.add_argument("--policy", action="append", required=True)
    args = parser.parse_args()
    from tools.link_merlin_session import _stage_paths

    if args.out.exists():
        parser.error("output must be fresh")
    result = evaluate(
        args.run_receipt,
        args.link_receipt,
        args.capture,
        args.profile,
        _stage_paths(args.stage_build),
        _stage_paths(args.policy),
        args.target_identity,
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "logits_checked", "inclusive_performance")
            }
        )
    )


if __name__ == "__main__":
    main()
