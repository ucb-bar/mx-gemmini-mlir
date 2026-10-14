"""Archive matched full-depth MX+VPU Spike logs after both runs finish."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from verify_full import HERE, SCALAR_SOURCE, compare, compiler_logits, source_logits


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def perf_counters(report: bytes) -> dict[str, int]:
    text = report.decode()
    main = re.search(
        r"^gemmini perf \[e4m3_vpu\]: (\d+) commands, (\d+) loops, "
        r"(\d+) fences; last activity (\d+);", text, re.M)
    mesh = re.search(r"^  mesh: (\d+) tiles, (\d+) busy cycles; "
                     r"VPU: (\d+) commands$", text, re.M)
    assert main and mesh
    return dict(zip(("commands", "loops", "fences", "last_activity",
                     "mesh_tiles", "mesh_busy_cycles", "vpu_commands"),
                    map(int, (*main.groups(), *mesh.groups()))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-log", "compiler-log", "source-perf", "compiler-perf",
                 "source-elf", "compiler-elf", "source-blob", "weight-receipt",
                 "compiler-catalog", "compiler-probe", "compiler-harness-object",
                 "compiler-objdump"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    scalar_receipt = json.loads((SCALAR_SOURCE / "receipt.json").read_text())
    weight_receipt = json.loads(args.weight_receipt.read_text())
    assert weight_receipt["exact_fp32_checkpoint_tensors_matched"] == 200
    assert weight_receipt["compiler_checkpoint_weights_sha256"] == (
        scalar_receipt["compiler_checkpoint_weights_sha256"])
    assert weight_receipt["input_ids_npz_sha256"] == scalar_receipt["input_ids_npz_sha256"]
    assert file_hash(args.source_blob) == scalar_receipt["source_blob_sha256"]
    catalog = json.loads(args.compiler_catalog.read_text())
    probe = json.loads(args.compiler_probe.read_text())
    assert catalog["covered_contractions"] == probe["selected_operations"] == 200
    assert all(kernel["precision"] == "mxfp8" for kernel in catalog["kernels"])
    assert probe["enabled_vpu_classes"] == ["add"]
    assert probe["vpu_add_lowered_regions"] == 88
    assert all(probe[f"vpu_{kind}_lowered_regions"] == 0 for kind in (
        "softmax", "elementwise", "unary", "reduce"))
    disassembly = subprocess.run(
        [str(args.compiler_objdump), "-d", "-M", "no-aliases",
         str(args.compiler_harness_object)], check=True, capture_output=True,
        text=True).stdout
    counter_reads = [line.strip() for line in disassembly.splitlines()
                     if re.search(r"\bcsrrs\s+\w+,cycle,zero\b", line)]
    assert len(counter_reads) == 2
    assert not re.search(r"\bcsrrs\s+\w+,mcycle,zero\b", disassembly)
    counter_disassembly = ("\n".join(counter_reads) + "\n").encode()

    raw = {name: path.read_bytes() for name, path in (
        ("source", args.source_log), ("compiler", args.compiler_log))}
    text = {name: data.decode() for name, data in raw.items()}
    for name in ("source", "compiler"):
        assert "gemmini: GEMMINI_MODE=both" in text[name]
    assert "llama MODEL benchmark finished (22 layer(s) + lm_head)." in text["source"]
    assert re.search(r"^DONE$", text["compiler"], re.M)
    assert re.search(r"^METRIC build_hash ([0-9a-f]+)$",
                     text["compiler"], re.M).group(1) == probe["build_hash"]
    source = source_logits(text["source"])
    compiler = compiler_logits(text["compiler"])
    assert source.size == compiler.size == 32000
    scalar_text = gzip.decompress((SCALAR_SOURCE / "source_spike.log.gz").read_bytes()).decode()
    scalar = source_logits(scalar_text)
    reference = np.load(SCALAR_SOURCE / "pytorch_last_logits.npy", allow_pickle=False)
    assert scalar.size == reference.size == 32000

    perf = {name: path.read_bytes() for name, path in (
        ("source", args.source_perf), ("compiler", args.compiler_perf))}
    for data in perf.values():
        assert data.startswith(b"gemmini perf [e4m3_vpu]:")

    receipt = {
        "schema": "mx_gemmini.nicolas_tinyllama_full_vpu_residual_spike.v1",
        "scope": "one 64-token TinyLlama prefill, all 22 decoder layers and final head",
        "cycle_ratio_valid": False,
        "cycle_ratio_reasons": [
            "source and compiler host algorithms and timer setup differ",
            "logits require numerical rather than bit-exact cross-program comparison",
        ],
        "source_timing_mode": "both",
        "compiler_timing_mode": "both",
        "compiler_counter_csr": "cycle",
        "compiler_counter_reads": 2,
        "compiler_counter_disassembly_sha256": hashlib.sha256(
            counter_disassembly).hexdigest(),
        "compiler_harness_object_sha256": file_hash(args.compiler_harness_object),
        "spike_isa": "rv64gc_zicntr",
        "perf_config": "e4m3_vpu",
        "perf_set": "mem.host_tracking=0",
        "source_commit": scalar_receipt["source_commit"],
        "rtl_commit": scalar_receipt["rtl_commit"],
        "spike_extlib_sha256": scalar_receipt["spike_extlib_sha256"],
        "input_ids_npz_sha256": scalar_receipt["input_ids_npz_sha256"],
        "source_blob_sha256": scalar_receipt["source_blob_sha256"],
        "compiler_catalog_sha256": file_hash(args.compiler_catalog),
        "compiler_probe_sha256": file_hash(args.compiler_probe),
        "compiler_build_hash": probe["build_hash"],
        "compiler_mx_contractions": 200,
        "compiler_mx_format": "mxfp8",
        "compiler_vpu_add_lowered_regions": 88,
        "source_vpu_residual_add_calls": 44,
        "source_capture_npz_sha256": weight_receipt["source_capture_npz_sha256"],
        "exact_fp32_checkpoint_tensors_matched": 200,
        "vpu_patch_sha256": file_hash(HERE / "vpu_residual.patch"),
        "source_elf_sha256": file_hash(args.source_elf),
        "compiler_elf_sha256": file_hash(args.compiler_elf),
        "source_vpu_spike_log_sha256": hashlib.sha256(raw["source"]).hexdigest(),
        "compiler_spike_log_sha256": hashlib.sha256(raw["compiler"]).hexdigest(),
        "source_perf_report_sha256": hashlib.sha256(perf["source"]).hexdigest(),
        "compiler_perf_report_sha256": hashlib.sha256(perf["compiler"]).hexdigest(),
        "source_perf_counters": perf_counters(perf["source"]),
        "compiler_perf_counters": perf_counters(perf["compiler"]),
        "source_vpu_logits_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        "compiler_logits_sha256": hashlib.sha256(compiler.tobytes()).hexdigest(),
        "source_vpu_matches_scalar_bits": bool(np.array_equal(
            source.view(np.uint32), scalar.view(np.uint32))),
        "source_stage_cycles": int(re.search(
            r"^stage_cycles64 0x([0-9a-f]+)$", text["source"], re.M).group(1), 16),
        "compiler_stage_cycles": int(re.search(
            r"^METRIC cycles ([0-9]+)$", text["compiler"], re.M).group(1)),
        "comparisons": {
            "source_vpu_vs_compiler": compare(source, compiler),
            "source_vpu_vs_pytorch": compare(source, reference),
            "compiler_vs_pytorch": compare(compiler, reference),
        },
    }
    for key in ("capture_model", "checkpoint_weights", "precision_overrides",
                "profile"):
        receipt[f"compiler_{key}_sha256"] = scalar_receipt[f"compiler_{key}_sha256"]
    for name, archive in (("source", "source_full_vpu.log.gz"),
                          ("compiler", "compiler_add_zicntr.log.gz")):
        (HERE / archive).write_bytes(gzip.compress(raw[name], mtime=0))
        (HERE / f"{name}_perf_report.txt").write_bytes(perf[name])
    (HERE / "full_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (HERE / "compiler_counter_disassembly.txt").write_bytes(counter_disassembly)
    print("archived full-depth MX+VPU source and compiler modeled runs")


if __name__ == "__main__":
    main()
