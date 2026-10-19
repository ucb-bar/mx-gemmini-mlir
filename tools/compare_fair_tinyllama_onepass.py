"""Compare a token-lookup reference source run with the compiled one-pass model.

This reports equal-input, token-to-logits cycles and both programs'
quality against the captured FP32 model. It never labels the ratio an
equal-quality speedup unless the numerical acceptance policy is met.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique(pattern: str, source: str) -> str:
    matches = re.findall(pattern, source, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"expected one match for {pattern!r}; got {len(matches)}")
    return matches[0]


def rows(log: str) -> list[str]:
    pairs = re.findall(r"^session_logits_bf16 (\d+) ([0-9a-f]{128000})$",
                       log, re.MULTILINE)
    if [int(index) for index, _ in pairs] != list(range(33)):
        raise ValueError("source must emit all 33 late BF16 logit rows")
    return [value for _, value in pairs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-log", type=Path, required=True)
    parser.add_argument("--source-elf", type=Path, required=True)
    parser.add_argument("--source-build-receipt", type=Path, required=True)
    parser.add_argument("--source-run-receipt", type=Path, required=True)
    parser.add_argument("--lookup-receipt", type=Path, required=True)
    parser.add_argument("--lookup-source", type=Path, required=True)
    parser.add_argument("--lookup-table", type=Path, required=True)
    parser.add_argument("--original-source-log-gz", type=Path, required=True)
    parser.add_argument("--compiler-comparison", type=Path, required=True)
    parser.add_argument("--compiler-log", type=Path, required=True)
    parser.add_argument("--compiler-run-receipt", type=Path, required=True)
    parser.add_argument("--input-identity", type=Path, required=True)
    parser.add_argument("--fp32-argmax", type=Path, required=True)
    parser.add_argument("--source-blob", type=Path)
    parser.add_argument("--source-header", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.source_blob is None) != (args.source_header is None):
        parser.error("source blob and header must be provided together")

    lookup = json.loads(args.lookup_receipt.read_text())
    build = json.loads(args.source_build_receipt.read_text())
    run = json.loads(args.source_run_receipt.read_text())
    earlier = json.loads(args.compiler_comparison.read_text())
    compiler_run = json.loads(args.compiler_run_receipt.read_text())
    identity = json.loads(args.input_identity.read_text())
    if (lookup.get("schema") != "mx_gemmini.nicolas_tinyllama_lookup_source.v1"
            or lookup.get("matched_bf16_elements") != 160 * 2048
            or lookup.get("bf16_mismatches") != 0
            or lookup.get("lookup_source_sha256") != sha(args.lookup_source)
            or lookup.get("embedding_table_sha256") != sha(args.lookup_table)
            or lookup.get("token_ids_sha256") != identity.get("token_ids_sha256")
            or lookup.get("compiler_weights_sha256") !=
            identity.get("compiler_weights_sha256")
            or lookup.get("source_activations_sha256") !=
            identity.get("source_activation_sha256")):
        raise ValueError("lookup source or embedding table changed")
    if (build.get("schema") != "mx_gemmini.nicolas_tinyllama_lookup_build.v1"
            or build.get("source_sha256") != sha(args.lookup_source)
            or build.get("embedding_table_sha256") != sha(args.lookup_table)
            or build.get("source_blob_sha256") !=
            earlier.get("sha256", {}).get("source_blob")
            or build.get("source_header_sha256") !=
            earlier.get("sha256", {}).get("source_header")
            or build.get("elf_sha256") != sha(args.source_elf)
            or build.get("spike_extension_config") != "e4m3_vpu"
            or build.get("spike_mem_host_tracking") != 0):
        raise ValueError("source executable is not bound to the lookup source")
    if (run.get("schema") != "mx_gemmini.nicolas_tinyllama_lookup_spike.v1"
            or run.get("build_receipt_sha256") != sha(args.source_build_receipt)
            or run.get("log_sha256") != sha(args.source_log)
            or run.get("emitted_logit_rows") != 33
            or run.get("emitted_top_tokens") != 160
            or run.get("spike_environment", {}).get("GEMMINI_MODE") != "both"
            or run.get("spike_environment", {}).get("GEMMINI_PERF_CONFIG")
            != "e4m3_vpu"
            or run.get("spike_environment", {}).get("GEMMINI_PERF_SET")
            != "mem.host_tracking=0"
            or "--isa=rv64gc_zicntr" not in run.get("spike_command", [])
            or run.get("spike_sha256") != compiler_run.get("spike_sha256")
            or run.get("extension_sha256") !=
            compiler_run.get("extension_sha256")):
        raise ValueError("source run does not use the matched Spike configuration")
    if (earlier.get("schema") != "mx_gemmini.nicolas_tinyllama_onepass_comparison.v1"
            or earlier.get("input_identity_receipt_sha256") != sha(args.input_identity)
            or earlier.get("source_actual_rows_match_source_reference") is not True
            or earlier.get("sha256", {}).get("compiler_run_receipt") !=
            sha(args.compiler_run_receipt)
            or earlier.get("compiler_vs_fp32_top_matches_33") is None
            or earlier.get("source_vs_fp32_top_matches_33") is None
            or identity.get("source_and_compiler_token_ids_exact") is not True
            or identity.get("source_and_compiler_fp32_weights_exact") is not True):
        raise ValueError("prior source/compiler quality evidence is invalid")
    old_log = gzip.open(args.original_source_log_gz, "rt").read()
    new_log = args.source_log.read_text()
    compiler_log = args.compiler_log.read_text()
    if (hashlib.sha256(old_log.encode()).hexdigest() !=
            earlier.get("sha256", {}).get("source_log")
            or sha(args.compiler_log) != earlier.get("sha256", {}).get("compiler_log")):
        raise ValueError("archived source/compiler logs changed")
    if (rows(old_log) != rows(new_log)
            or "llama MODEL benchmark finished (22 layer(s) + lm_head)." not in new_log
            or "\nDONE\n" not in "\n" + compiler_log):
        raise ValueError("new source run did not preserve the full source output")
    old_cycles = int(unique(r"^stage_cycles64 0x([0-9a-f]{16})$", old_log), 16)
    if old_cycles != earlier["source_stage_cycles"]:
        raise ValueError("original source cycles differ from archived comparison")
    compiler_cycles = int(unique(r"^METRIC cycles (\d+)$", compiler_log))
    if compiler_cycles != earlier["compiler_stage_cycles"]:
        raise ValueError("compiler cycles differ from archived comparison")
    source_cycles = int(unique(r"^stage_cycles64 0x([0-9a-f]{16})$", new_log), 16)
    if source_cycles != run.get("stage_cycles"):
        raise ValueError("source run receipt has another cycle count")
    source_tops = [int(value) for value in unique(
        r"^source_argmax_160((?: \d+){160})$", new_log).split()]
    generated_tops = None
    if args.source_blob is not None:
        if (sha(args.source_blob) != earlier["sha256"]["source_blob"]
                or sha(args.source_header) != earlier["sha256"]["source_header"]):
            raise ValueError("generated Nicolas source reference changed")
        offset = int(unique(r"^#define LLAMA_OFF_LOGITS_OUT\s+(\d+)u$",
                            args.source_header.read_text()))
        bf16 = np.memmap(args.source_blob, mode="r", dtype="<u2",
                         offset=offset, shape=(160, 32000))
        fp32 = (np.asarray(bf16, dtype="<u4") << 16).view("<f4")
        generated_tops = np.argmax(fp32, axis=1).tolist()
    compiler_tops = earlier["compiler_argmax_160"]
    fp32_tops = np.load(args.fp32_argmax, allow_pickle=False).reshape(-1)
    if (len(source_tops) != 160 or len(compiler_tops) != 160
            or fp32_tops.shape != (160,)
            or source_tops[127:] != earlier["source_argmax_33"]
            or sum(source_tops[i] == int(fp32_tops[i])
                   for i in range(127, 160)) !=
            earlier["source_vs_fp32_top_matches_33"]
            or sum(compiler_tops[i] == int(fp32_tops[i])
                   for i in range(160)) !=
            earlier["compiler_vs_fp32_top_matches_160"]):
        raise ValueError("source or compiler top-token lists differ from archive")
    report = {
        "schema": "mx_gemmini.tinyllama_onepass_equal_boundary.v1",
        "scope": "160 token IDs through all 22 decoder layers and LM head",
        "source_stage_cycles": source_cycles,
        "compiler_stage_cycles": compiler_cycles,
        "compiler_over_source_cycles": compiler_cycles / source_cycles,
        "compiler_cycle_difference_percent":
            100 * (compiler_cycles / source_cycles - 1),
        "same_token_ids_and_fp32_weights": True,
        "same_all_fp8_matrix_policy": True,
        "same_logical_input_output_boundary": True,
        "source_late_rows_unchanged_by_lookup": True,
        "source_full_top_tokens_match_generated_reference":
            source_tops == generated_tops if generated_tops is not None else None,
        "source_vs_fp32_late_top_matches_33":
            earlier["source_vs_fp32_top_matches_33"],
        "compiler_vs_fp32_late_top_matches_33":
            earlier["compiler_vs_fp32_top_matches_33"],
        "source_vs_compiler_late_top_matches_33":
            earlier["source_vs_compiler_top_matches_33"],
        "source_vs_fp32_top_matches_160": sum(
            source_tops[i] == int(fp32_tops[i]) for i in range(160)),
        "compiler_vs_fp32_top_matches_160":
            earlier["compiler_vs_fp32_top_matches_160"],
        "source_vs_compiler_top_matches_160": sum(
            source_tops[i] == compiler_tops[i] for i in range(160)),
        "equal_quality_speedup_valid": False,
        "quality_note": (
            "Source and compiler use different host/VPU policies and do not "
            "meet an agreed equal-quality threshold."),
        "source_elf_sha256": sha(args.source_elf),
        "source_log_sha256": sha(args.source_log),
        "lookup_receipt_sha256": sha(args.lookup_receipt),
        "source_build_receipt_sha256": sha(args.source_build_receipt),
        "source_run_receipt_sha256": sha(args.source_run_receipt),
        "compiler_log_sha256": sha(args.compiler_log),
        "compiler_run_receipt_sha256": sha(args.compiler_run_receipt),
    }
    if generated_tops is not None:
        report["generated_source_top_tokens_160"] = generated_tops
        report["source_vs_generated_top_matches_160"] = sum(
            left == right for left, right in zip(source_tops, generated_tops))
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in (
        "source_stage_cycles", "compiler_stage_cycles",
        "compiler_cycle_difference_percent", "equal_quality_speedup_valid")},
        indent=2))


if __name__ == "__main__":
    main()
