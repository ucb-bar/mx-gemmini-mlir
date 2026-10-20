"""Compare every recorded cached TinyLlama logit against an independent port.

Numerical acceptance uses the same rule as the qualified full-forward work:
mean and worst KL may be at most 5% above the source baseline, and top-token
agreement with the floating reference must be at least as good. Timing is
reported separately when an actual measured executable is unavailable.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from tools.audit_radiance_compiler_weights import audit as audit_weights
from tools.audit_recurrent_session_state import audit, file_digest
from tools.qualify_radiance_rocket_llama import check_accelerator_work, parse_console


def quality(logits, reference):
    if logits.shape != reference.shape or logits.shape != (33, 32000):
        raise ValueError(
            "cached quality needs every vocabulary logit at all 33 positions"
        )
    if not np.isfinite(logits).all() or not np.isfinite(reference).all():
        raise ValueError("nonfinite cached logits")

    def log_prob(x):
        x = x.astype(np.float64)
        x -= x.max(axis=1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=1, keepdims=True))

    p, q = log_prob(reference), log_prob(logits)
    kl = np.maximum(np.sum(np.exp(p) * (p - q), axis=1), 0)
    top = logits.argmax(axis=1) == reference.argmax(axis=1)
    return {
        "mean_kl_nats": float(kl.mean()),
        "worst_kl_nats": float(kl.max()),
        "top_token_matches": int(top.sum()),
        "positions": 33,
        "per_position_kl_nats": kl.tolist(),
        "per_position_top_token_equal": top.tolist(),
    }


def gates(candidate, source):
    return {
        "mean_kl_within_5_percent": candidate["mean_kl_nats"]
        <= 1.05 * source["mean_kl_nats"],
        "worst_kl_within_5_percent": candidate["worst_kl_nats"]
        <= 1.05 * source["worst_kl_nats"],
        "top_token_matches_at_least_source": candidate["top_token_matches"]
        >= source["top_token_matches"],
    }


def compare(port_dir, run_path, link_path, capture, prefill_build, decode_build):
    port_path = port_dir / "receipt.json"
    port = json.loads(port_path.read_text())
    run = json.loads(run_path.read_text())
    link = json.loads(link_path.read_text())
    if (
        port["status"] != "completed_requires_matched_comparison"
        or port["mode"] != "recorded_cached_tinyllama"
        or port["session_steps"] != 33
        or port["cache_capacity"] != 160
        or port["independent_host_graph"] is not True
    ):
        raise ValueError(
            "independent port has not completed the complete matched session"
        )
    compiler_audit = audit("tinyllama", run_path, link_path)
    weight_audits = {
        "prefill": audit_weights(
            port_dir, prefill_build, compiler_elf=link["elf"], source_stage="prefill"
        ),
        "decode": audit_weights(
            port_dir,
            decode_build,
            compiler_elf=link["elf"],
            symbol_prefix="merlin_stage_1_",
            source_stage="decode",
        ),
    }
    if any(
        row["compiler_elf_sha256"] != link["elf_sha256"]
        for row in weight_audits.values()
    ):
        raise ValueError("encoded weight audit belongs to another compiler image")
    if [p["capture_model_sha256"] for p in link["programs"]] != [
        file_digest(capture / "stages/prefill/model.mlir"),
        file_digest(capture / "stages/decode/model.mlir"),
    ]:
        raise ValueError("compiler session capture differs")
    for filename, expected in port["session_capture_sha256"].items():
        if file_digest(capture / filename) != expected:
            raise ValueError("source session capture differs: " + filename)
    for filename, expected in port["capture_files_sha256"].items():
        if file_digest(capture / "stages/prefill" / filename) != expected:
            raise ValueError("source prefill capture differs: " + filename)
    if (
        port["tools_sha256"]["spike"] != run["spike_sha256"]
        or port["extension_sha256"] != run["extension_sha256"]
        or port["profile_sha256"] != link["profile_sha256"]
        or port["environment"]
        != {
            "GEMMINI_MODE": run["timing_mode"],
            "GEMMINI_PERF_CONFIG": run["timing_config"],
            "GEMMINI_PERF_SET": run["timing_overrides"],
        }
    ):
        raise ValueError("source and compiler hardware or timing model differs")
    for filename in (
        "session.elf",
        "phases.bin",
        "token_ids.bin",
        "checks.json",
        "spike.log",
        "perf.txt",
    ):
        if file_digest(port_dir / filename) != port["files_sha256"][filename]:
            raise ValueError("retained port artifact differs: " + filename)
    if file_digest(port_dir / "session.elf") != port["elf_sha256"]:
        raise ValueError("actual independent ELF differs")
    checks = json.loads((port_dir / "checks.json").read_text())
    console = (port_dir / "spike.log").read_text()
    cycles, _, values = parse_console(console, checks, port["phase_count"], False)
    reset = re.findall(r"^CACHED_RESET ([1-9][0-9]*)$", console, re.MULTILINE)
    steps = [
        tuple(map(int, row))
        for row in re.findall(
            r"^CACHED_STEP (\d+) (\d+) ([1-9][0-9]*)$", console, re.MULTILINE
        )
    ]
    if len(reset) != 1 or [(s, p) for s, p, _ in steps] != [
        (s, 128 + s) for s in range(33)
    ]:
        raise ValueError("source recurrent execution did not visit every position")
    if (
        cycles != int(reset[0]) + sum(t for _, _, t in steps)
        or cycles != port["cycles"]
    ):
        raise ValueError("source inclusive timing does not sum")
    program = np.fromfile(port_dir / "phases.bin", "<u4").reshape(-1, 32)
    if len(program) != port["phase_count"]:
        raise ValueError("source descriptor count differs")
    work = check_accelerator_work(program, (port_dir / "perf.txt").read_text())
    source_logits = np.frombuffer(values[0], "<f4").reshape(33, 32000)
    state = np.frombuffer(values[1], "<f4").reshape(2, 22, 4, 160, 64)
    if not np.isfinite(state).all():
        raise ValueError("nonfinite final source cache")
    with np.load(capture / "stages/prefill/inputs.npz", allow_pickle=False) as data:
        prefix_ids = data["in0"].reshape(-1)
    with np.load(
        capture / "stages/decode/session_inputs.npz", allow_pickle=False
    ) as data:
        decode_ids = data["decode_tokens"].reshape(-1)
    if not np.array_equal(
        np.fromfile(port_dir / "token_ids.bin", "<i8"),
        np.concatenate([prefix_ids, decode_ids]),
    ):
        raise ValueError(
            "source token stream differs from the compiler's recorded stream"
        )
    reference = [np.load(capture / "stages/prefill/golden.npy", allow_pickle=False)]
    with np.load(
        capture / "stages/decode/session_quality_fp32.npz", allow_pickle=False
    ) as data:
        reference.extend(data["logits"])
    reference = np.stack(reference).reshape(33, 32000)
    rows = {(r["program"], r["step"], r["output"]): r for r in run["output_dumps"]}
    keys = [(0, 0, 0)] + [(1, s, 0) for s in range(32)]
    candidate = np.stack(
        [
            np.frombuffer((run_path.parent / rows[k]["path"]).read_bytes(), "<f4")
            for k in keys
        ]
    )
    source_quality, candidate_quality = (
        quality(source_logits, reference),
        quality(candidate, reference),
    )
    acceptance = gates(candidate_quality, source_quality)
    ratio = cycles / compiler_audit["measured_cycles"]
    qualified = all(acceptance.values()) and compiler_audit["actual_final_elf_reopened"]
    return {
        "schema": "mx_gemmini.independent_cached_tinyllama_comparison.v1",
        "port_receipt_sha256": file_digest(port_path),
        "compiler_state_audit": compiler_audit,
        "encoded_checkpoint_audits": weight_audits,
        "source_cycles": cycles,
        "compiler_cycles": compiler_audit["measured_cycles"],
        "observed_source_over_compiler": ratio,
        "source_prefill_cycles": steps[0][2],
        "source_reset_cycles": int(reset[0]),
        "source_decode_cycles": sum(t for _, _, t in steps[1:]),
        "source_quality": source_quality,
        "compiler_quality": candidate_quality,
        "numerical_gates": acceptance,
        "accelerator_work": work,
        "comparison_qualified": qualified,
        "qualified_compiler_speedup": ratio if qualified else None,
        "faster_and_qualified": qualified and ratio > 1,
        "complete_output_words_per_side": 33 * 32000,
        "source_final_kv_words_checked_finite": state.size,
        "precision_scope": "matched_MX_FP8_E8M0_block32_BF16_matrix_outputs; source_and_compiler_row_arithmetic_differ",
        "workload_scope": "fixed_160_token_stream;_128_prefill_plus_32_teacher_forced_single_token_decodes;_modeled_cycles",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "port-dir",
        "run-receipt",
        "link-receipt",
        "capture",
        "compiler-prefill-build",
        "compiler-decode-build",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output must be fresh")
    report = compare(
        args.port_dir,
        args.run_receipt,
        args.link_receipt,
        args.capture,
        args.compiler_prefill_build,
        args.compiler_decode_build,
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
