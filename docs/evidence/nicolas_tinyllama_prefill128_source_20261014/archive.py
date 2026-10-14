"""Archive completed modeled Spike logs for the captured TinyLlama prefill."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
from verify import HERE, compare, compiler_logits, cycle_counts, digest, source_logits, verify


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def perf_counters(data: bytes) -> dict[str, int]:
    text = data.decode()
    main = re.search(
        r"^gemmini perf \[e4m3_vpu\]: (\d+) commands, (\d+) loops, "
        r"(\d+) fences; last activity (\d+);", text, re.M)
    mesh = re.search(r"^  mesh: (\d+) tiles, (\d+) busy cycles; "
                     r"VPU: (\d+) commands$", text, re.M)
    assert main is not None and mesh is not None
    keys = ("commands", "loops", "fences", "last_activity",
            "mesh_tiles", "mesh_busy_cycles", "vpu_commands")
    return dict(zip(keys, map(int, (*main.groups(), *mesh.groups()))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-log", "compiler-log", "diagnostic-log",
                 "source-perf", "compiler-perf", "source-elf", "compiler-elf",
                 "source-blob", "source-header", "compiler-catalog", "compiler-harness-object",
                 "compiler-objdump", "spike-extlib"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    logs = {
        "source": args.source_log.read_bytes(),
        "compiler": args.compiler_log.read_bytes(),
        "diagnostic": args.diagnostic_log.read_bytes(),
    }
    source_text = logs["source"].decode()
    compiler_text = logs["compiler"].decode()
    diagnostic_text = logs["diagnostic"].decode()
    assert "llama MODEL benchmark finished (22 layer(s) + lm_head)." in source_text
    assert "llama MODEL test PASSED (1 layer(s) + lm_head" in diagnostic_text
    assert re.search(r"^DONE$", compiler_text, re.M)
    assert "gemmini: GEMMINI_MODE=both" in source_text
    assert "gemmini: GEMMINI_MODE=both" in compiler_text
    cycles = cycle_counts(source_text, compiler_text)
    source = source_logits(source_text)
    compiler = compiler_logits(compiler_text)
    reference = np.load(HERE / "pytorch_last_logits.npy", allow_pickle=False).reshape(-1)
    assert source.size == compiler.size == reference.size == 32000
    header = args.source_header.read_text()
    def header_int(name: str) -> int:
        match = re.search(rf"^#define {name}\s+(\d+)u?\b", header, re.M)
        assert match is not None, name
        return int(match.group(1))
    assert header_int("LLAMA_M") == 128 and header_int("LLAMA_V") == 32000
    offset = header_int("LLAMA_OFF_LOGITS_OUT") + 127 * 32000 * 2
    generated_last = np.memmap(args.source_blob, dtype="<u2", mode="r",
                               offset=offset, shape=(32000,)).copy()
    source_bf16 = (source.view("<u4") >> 16).astype("<u2")
    assert np.array_equal(source_bf16, generated_last)
    (HERE / "source_generated_last_bf16.bin").write_bytes(generated_last.tobytes())
    comparison = {
        "source_vs_compiler": compare(source, compiler),
        "source_vs_pytorch": compare(source, reference),
        "compiler_vs_pytorch": compare(compiler, reference),
    }
    catalog_raw = json.loads(args.compiler_catalog.read_text())
    summary = json.loads((HERE / "catalog_summary.json").read_text())
    assert catalog_raw["covered_contractions"] == len(catalog_raw["bindings"]) == 200
    assert catalog_raw["profile_sha256"] == summary["profile_sha256"]
    assert catalog_raw["checkpoint_weights_sha256"] == (
        summary["checkpoint_weights_sha256"])
    assert len(catalog_raw["vpu_add"]["source_regions"]) == 88
    assert all(binding["target_dtypes"][:2] == ["f8E4M3FN", "f8E4M3FN"]
               for binding in catalog_raw["bindings"])
    weight_receipt = json.loads((HERE / "weight_receipt.json").read_text())
    assert weight_receipt["exact_fp32_checkpoint_tensors_matched"] == 200
    disassembly = subprocess.run(
        [str(args.compiler_objdump), "-d", "-M", "no-aliases",
         str(args.compiler_harness_object)], check=True, capture_output=True,
        text=True).stdout
    reads = [line.strip() for line in disassembly.splitlines()
             if re.search(r"\bcsrrs\s+\w+,cycle,zero\b", line)]
    assert len(reads) == 2
    counter_text = "\n".join(reads) + "\n"
    (HERE / "compiler_counter_disassembly.txt").write_text(counter_text)

    artifacts = {
        "source_spike.log.gz": gzip.compress(logs["source"], mtime=0),
        "compiler_spike.log.gz": gzip.compress(logs["compiler"], mtime=0),
        "one_layer_diagnostic.log.gz": gzip.compress(logs["diagnostic"], mtime=0),
        "source_perf_report.txt": args.source_perf.read_bytes(),
        "compiler_perf_report.txt": args.compiler_perf.read_bytes(),
    }
    for name, data in artifacts.items():
        (HERE / name).write_bytes(data)
    small_files = (
        "all_fp8_overrides.json", "catalog_summary.json", "compiler_inputs.npz",
        "source_capture_inputs.npz", "pytorch_last_logits.npy",
        "weight_receipt.json", "m128_tiling.patch",
        "compiler_counter_disassembly.txt", *artifacts,
        "source_generated_last_bf16.bin",
    )
    receipt = {
        "schema": "mx_gemmini.tinyllama_prefill128_source.v1",
        "scope": "one 128-token prefill, 22 layers and final head",
        "cycle_ratio_valid": False,
        "cycle_ratio_reasons": [
            "different host math, VPU operations, and setup boundaries",
            "source attention uses 128 valid keys; compiler issues 160 cache slots",
            "full captured TinyLlama session uses mixed FP4/FP8 and 32 decode steps",
        ],
        "spike_isa": "rv64gc_zicntr",
        "perf_config": "e4m3_vpu",
        "perf_set": "mem.host_tracking=0",
        "source_commit": "0344bcba70b899039ccd0ac0801ca5605d46d3eb",
        "rtl_commit": "266c593f2cb51d7e3fe83fc0317072b585ac3c52",
        "spike_extlib_sha256": file_hash(args.spike_extlib),
        "source_elf_sha256": file_hash(args.source_elf),
        "compiler_elf_sha256": file_hash(args.compiler_elf),
        "source_blob_sha256": file_hash(args.source_blob),
        "source_header_sha256": file_hash(args.source_header),
        "source_output_matches_generated_quantized_logits": True,
        "compiler_catalog_sha256": file_hash(args.compiler_catalog),
        "compiler_harness_object_sha256": file_hash(args.compiler_harness_object),
        "compiler_build_hash": re.search(
            r"^METRIC build_hash ([0-9a-f]+)$", compiler_text, re.M).group(1),
        "profile_sha256": summary["profile_sha256"],
        "compiler_checkpoint_weights_sha256": (
            weight_receipt["compiler_checkpoint_weights_sha256"]),
        "checkpoint_fp32_tensors_matched": 200,
        "source_capture_npz_sha256": weight_receipt["source_capture_npz_sha256"],
        "compiler_mx_bindings": 200,
        "compiler_mx_precision": "mxfp8",
        "compiler_vpu_add_regions": 88,
        "source_vpu_residual_add_calls": 44,
        "source_attention_valid_keys": 128,
        "compiler_attention_key_slots": 160,
        "source_next_token_label": [26793],
        "source_timing_boundary": "before gemmini_flush through final head fence",
        "compiler_timing_boundary": "before session reset through state commit",
        "stage_cycles": cycles,
        "comparisons": comparison,
        "perf_counters": {
            "source": perf_counters(artifacts["source_perf_report.txt"]),
            "compiler": perf_counters(artifacts["compiler_perf_report.txt"]),
        },
        "vpu_residual_patch_sha256": file_hash(
            HERE.parent / "nicolas_tinyllama_full_source_vpu_20261014/vpu_residual.patch"),
        "artifact_sha256": {name: digest((HERE / name).read_bytes())
                            for name in small_files},
    }
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    verify()
    print("archived 128-token source and compiler prefill evidence")


if __name__ == "__main__":
    main()
