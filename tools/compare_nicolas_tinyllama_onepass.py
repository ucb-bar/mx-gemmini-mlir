"""Compare source and compiled TinyLlama on the same 160-token forward.

The source starts from precomputed BF16 embeddings. The compiler accepts the
same token IDs and performs the embedding lookup, so cycle boundaries differ
by that host operation. This is also distinct from cached 33-step generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_one(pattern: str, text: str, label: str) -> str:
    matches = re.findall(pattern, text, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"expected one {label}, found {len(matches)}")
    return matches[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64, copy=False)
    bb = b.astype(np.float64, copy=False)
    return float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--source-header", type=Path, required=True)
    parser.add_argument("--source-blob", type=Path, required=True)
    parser.add_argument("--compiler-log", type=Path, required=True)
    parser.add_argument("--compiler-build-receipt", type=Path, required=True)
    parser.add_argument("--compiler-run-receipt", type=Path)
    parser.add_argument("--compiler-bundle", type=Path, required=True)
    parser.add_argument("--input-identity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run_receipt = args.compiler_run_receipt or args.compiler_log.parent / "mx_spike_run.json"

    identity = json.loads(args.input_identity.read_text())
    if (identity.get("schema") != "mx_gemmini.nicolas_tinyllama_onepass_input_identity.v1"
            or identity.get("source_and_compiler_token_ids_exact") is not True
            or identity.get("source_and_compiler_fp32_weights_exact") is not True
            or identity.get("matched_fp32_weight_tensors") != 200):
        raise ValueError("source/compiler inputs have not been proven identical")
    capture_receipt = args.compiler_bundle / "capture_receipt.json"
    capture = json.loads(capture_receipt.read_text())
    if (sha(capture_receipt) != identity["compiler_capture_receipt_sha256"]
            or capture["artifacts"]["model.mlir"]["sha256"] !=
            identity["compiler_mlir_sha256"]):
        raise ValueError("input identity belongs to a different compiler capture")
    build = json.loads(args.compiler_build_receipt.read_text())
    if (build.get("schema") != "mx_gemmini.merlin_stage_probe.v1"
            or build.get("capture_model_sha256") != identity["compiler_mlir_sha256"]
            or build.get("selected_operations") != 200):
        raise ValueError("compiler build is not the matched all-layer capture")
    bindings = json.loads(Path(build["catalog_manifest"]).read_text())["bindings"]
    if (len(bindings) != 200 or any(
            row["target_dtypes"][:2] != ["f8E4M3FN", "f8E4M3FN"]
            for row in bindings)):
        raise ValueError("compiler did not use all-FP8 MX contractions")
    run = json.loads(run_receipt.read_text())
    environment = run.get("spike_environment") or {}
    if (run.get("schema") != "mx_gemmini.linked_stage_spike_run.v1"
            or run.get("done") is not True
            or run.get("build_receipt_sha256") != sha(args.compiler_build_receipt)
            or environment.get("GEMMINI_MODE") != "both"
            or environment.get("GEMMINI_PERF_CONFIG") != "e4m3_vpu"
            or environment.get("GEMMINI_PERF_SET") !=
            "mem.host_tracking=0"
            or "--isa=rv64gc_zicntr" not in run.get("command", [])):
        raise ValueError("compiler Spike run has a different extension configuration")

    source_log = args.source_log.read_text()
    compiler_log = args.compiler_log.read_text()
    if run.get("log_sha256") != sha(args.compiler_log):
        raise ValueError("compiler log differs from its run receipt")
    if ("llama MODEL benchmark finished (22 layer(s) + lm_head)." not in source_log
            or "\nDONE\n" not in "\n" + compiler_log):
        raise ValueError("both Spike runs must finish")
    source_cycles = int(require_one(r"^stage_cycles64 0x([0-9a-f]{16})$",
                                    source_log, "source cycles"), 16)
    compiler_cycles = int(require_one(r"^METRIC cycles (\d+)$", compiler_log,
                                      "compiler cycles"))
    build_hash = require_one(r"^METRIC build_hash ([0-9a-f]+)$", compiler_log,
                             "compiler build hash")
    if build_hash != build["build_hash"]:
        raise ValueError("compiler console belongs to another build")
    words = require_one(r"^ARGMAX 160((?: \d+){160})$", compiler_log,
                        "160 compiler argmax values")
    compiler_tops = np.fromstring(words, dtype=np.int64, sep=" ")
    rows = re.findall(r"^session_logits_bf16 (\d+) ([0-9a-f]{128000})$",
                      source_log, re.MULTILINE)
    if len(rows) != 33 or [int(index) for index, _ in rows] != list(range(33)):
        raise ValueError("source did not report positions 127..159")
    source_bits = np.stack([
        np.frombuffer(bytes.fromhex(encoded), dtype=">u2").astype("<u2")
        for _, encoded in rows])
    source_logits = (source_bits.astype("<u4") << 16).view("<f4")
    header = args.source_header.read_text()
    offset = int(require_one(r"^#define LLAMA_OFF_LOGITS_OUT\s+(\d+)u$",
                             header, "source logit offset"))
    with args.source_blob.open("rb") as stream:
        stream.seek(offset)
        reference = np.frombuffer(stream.read(160 * 32000 * 2),
                                  dtype="<u2").reshape(160, 32000)
    if reference.shape != (160, 32000):
        raise ValueError("source reference is truncated")
    golden = np.load(args.compiler_bundle / "golden.npy", allow_pickle=False)
    if golden.shape != (1, 160, 32000):
        raise ValueError("compiler golden tensor has unexpected shape")
    source_tops = np.argmax(source_logits, axis=1)
    golden_tops = np.argmax(golden[0], axis=1)
    report = {
        "schema": "mx_gemmini.nicolas_tinyllama_onepass_comparison.v1",
        "scope": "one 160-token causal forward through 22 layers and LM head",
        "input_identity_receipt_sha256": sha(args.input_identity),
        "source_actual_rows_match_source_reference": bool(
            np.array_equal(source_bits, reference[127:160])),
        "source_reference_mismatches_per_row": [
            int(np.count_nonzero(source_bits[i] != reference[127 + i]))
            for i in range(33)],
        "source_stage_cycles": source_cycles,
        "compiler_stage_cycles": compiler_cycles,
        "cycle_boundary_difference": (
            "source starts from precomputed BF16 embeddings; compiler performs "
            "token embedding lookup. Both use the same token IDs and checkpoint weights."),
        "operation_placement_difference": (
            "source uses MX matrices, host softmax/RoPE/normalization/SwiGLU, "
            "and VPU residual adds; compiler uses MX matrices and also routes "
            "eligible softmax, add, elementwise, unary, and reduction operations "
            "to VPU"),
        "compiler_vpu_lowered_regions": {
            name: build[f"vpu_{name}_lowered_regions"]
            for name in ("softmax", "add", "elementwise", "unary", "reduce")
        },
        "source_vs_compiler_top_matches_33": int(np.count_nonzero(
            source_tops == compiler_tops[127:160])),
        "source_vs_fp32_top_matches_33": int(np.count_nonzero(
            source_tops == golden_tops[127:160])),
        "compiler_vs_fp32_top_matches_160": int(np.count_nonzero(
            compiler_tops == golden_tops)),
        "compiler_vs_fp32_top_matches_33": int(np.count_nonzero(
            compiler_tops[127:160] == golden_tops[127:160])),
        "source_vs_fp32_late_cosine_mean": float(np.mean([
            cosine(source_logits[i], golden[0, 127 + i]) for i in range(33)])),
        "compiler_argmax_160": compiler_tops.tolist(),
        "source_argmax_33": source_tops.tolist(),
        "sha256": {
            "source_log": sha(args.source_log),
            "compiler_log": sha(args.compiler_log),
            "compiler_build_receipt": sha(args.compiler_build_receipt),
            "compiler_run_receipt": sha(run_receipt),
            "source_header": sha(args.source_header),
            "source_blob": sha(args.source_blob),
        },
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "source_stage_cycles", "compiler_stage_cycles",
        "source_vs_compiler_top_matches_33",
        "compiler_vs_fp32_top_matches_33",
        "source_actual_rows_match_source_reference")}, indent=2))


if __name__ == "__main__":
    main()
