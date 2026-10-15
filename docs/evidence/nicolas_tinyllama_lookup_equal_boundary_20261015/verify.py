"""Check the compact independent TinyLlama token-to-logits Spike archive."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "nicolas_tinyllama_onepass160_same_input_20261014"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def one(pattern: str, log: str) -> str:
    matches = re.findall(pattern, log, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern!r}; found {len(matches)}")
    return matches[0]


def rows(log: str) -> list[str]:
    found = re.findall(r"^session_logits_bf16 (\d+) ([0-9a-f]{128000})$",
                       log, re.MULTILINE)
    if [int(index) for index, _ in found] != list(range(33)):
        raise ValueError("expected all 33 BF16 logit rows")
    return [value for _, value in found]


def check() -> None:
    report = read_json(HERE / "fair_comparison.json")
    lookup_data = (HERE / "lookup_receipt.json").read_bytes()
    build_data = (HERE / "build_receipt.json").read_bytes()
    run_data = (HERE / "run_receipt.json").read_bytes()
    lookup = json.loads(lookup_data)
    build = json.loads(build_data)
    run = json.loads(run_data)
    source_data = gzip.decompress((HERE / "source_spike.log.gz").read_bytes())
    source = source_data.decode()
    prior = read_json(PRIOR / "matched_onepass_comparison.json")
    old_data = gzip.decompress((PRIOR / "source_timed.log.gz").read_bytes())
    old = old_data.decode()
    compiler = (PRIOR / "compiler_spike.log").read_text()
    compiler_run = read_json(PRIOR / "compiler_run.json")
    identity_data = (PRIOR / "input_identity.json").read_bytes()
    identity = json.loads(identity_data)
    fp32 = np.load(PRIOR / "fp32_argmax_160.npy", allow_pickle=False)

    if (report["schema"] != "mx_gemmini.tinyllama_onepass_equal_boundary.v1"
            or lookup["schema"] != "mx_gemmini.nicolas_tinyllama_lookup_source.v1"
            or build["schema"] != "mx_gemmini.nicolas_tinyllama_lookup_build.v1"
            or run["schema"] != "mx_gemmini.nicolas_tinyllama_lookup_spike.v1"
            or report["lookup_receipt_sha256"] != digest(lookup_data)
            or report["source_build_receipt_sha256"] != digest(build_data)
            or report["source_run_receipt_sha256"] != digest(run_data)
            or run["build_receipt_sha256"] != digest(build_data)
            or report["source_log_sha256"] != digest(source_data)
            or run["log_sha256"] != digest(source_data)
            or digest(old_data) != prior["sha256"]["source_log"]
            or report["compiler_log_sha256"] != digest(compiler.encode())
            or report["compiler_run_receipt_sha256"] !=
            digest((PRIOR / "compiler_run.json").read_bytes())
            or run["spike_sha256"] != compiler_run["spike_sha256"]
            or run["extension_sha256"] != compiler_run["extension_sha256"]
            or prior["input_identity_receipt_sha256"] != digest(identity_data)
            or lookup["token_ids_sha256"] != identity["token_ids_sha256"]
            or lookup["compiler_weights_sha256"] !=
            identity["compiler_weights_sha256"]
            or lookup["source_activations_sha256"] !=
            identity["source_activation_sha256"]
            or lookup["source_c_sha256"] !=
            "ffd14dc19420dfb803cb55e49013a7aaf957ce6dd703b4a375212eed2fec9254"
            or lookup["bf16_mismatches"] != 0
            or lookup["matched_bf16_elements"] != 160 * 2048
            or build["source_blob_sha256"] != prior["sha256"]["source_blob"]
            or build["source_header_sha256"] != prior["sha256"]["source_header"]
            or report["source_elf_sha256"] != build["elf_sha256"]
            or rows(source) != rows(old)
            or "llama MODEL benchmark finished (22 layer(s) + lm_head)." not in source):
        raise ValueError("archive integrity or source-output check failed")

    cycles = int(one(r"^stage_cycles64 0x([0-9a-f]{16})$", source), 16)
    tops = [int(value) for value in one(
        r"^source_argmax_160((?: \d+){160})$", source).split()]
    generated = report["generated_source_top_tokens_160"]
    compiler_tops = prior["compiler_argmax_160"]
    if (cycles != report["source_stage_cycles"]
            or cycles != run["stage_cycles"]
            or report["compiler_stage_cycles"] !=
            int(one(r"^METRIC cycles (\d+)$", compiler))
            or tops[127:] != prior["source_argmax_33"]
            or len(generated) != 160
            or report["source_vs_generated_top_matches_160"] !=
            sum(left == right for left, right in zip(tops, generated))
            or report["source_full_top_tokens_match_generated_reference"] !=
            (tops == generated)
            or report["source_vs_fp32_top_matches_160"] !=
            sum(tops[index] == int(fp32[index]) for index in range(160))
            or report["source_vs_compiler_top_matches_160"] !=
            sum(tops[index] == compiler_tops[index] for index in range(160))
            or report["compiler_vs_fp32_top_matches_160"] !=
            prior["compiler_vs_fp32_top_matches_160"]
            or report["equal_quality_speedup_valid"] is not False):
        raise ValueError("cycle or quality report differs from raw Spike output")
    print(f"verified 22 layers; source {cycles:,} cycles; "
          f"compiler {report['compiler_stage_cycles']:,} cycles; "
          f"FP32 top matches {report['source_vs_fp32_top_matches_160']}/160 "
          f"versus {report['compiler_vs_fp32_top_matches_160']}/160")


if __name__ == "__main__":
    check()
