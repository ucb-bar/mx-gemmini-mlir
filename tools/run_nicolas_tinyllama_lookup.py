"""Build and run the independent TinyLlama source with token lookup on Spike."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(command: list[str], *, cwd: Path | None = None,
        env: dict[str, str] | None = None, log: Path | None = None) -> None:
    if log is None:
        subprocess.run(command, cwd=cwd, env=env, check=True)
    else:
        with log.open("w") as stream:
            subprocess.run(command, cwd=cwd, env=env, check=True,
                           stdout=stream, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True,
                        help="output of prepare_nicolas_tinyllama_lookup")
    parser.add_argument("--source-blob", type=Path, required=True)
    parser.add_argument("--source-header-dir", type=Path, required=True)
    parser.add_argument("--prior-comparison", type=Path, required=True)
    parser.add_argument("--npu-root", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, required=True)
    parser.add_argument("--riscv-root", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    prepared = args.prepared.resolve(strict=True)
    source = prepared / "llama_model_lookup.c"
    table = prepared / "embedding_fp32.bin"
    prep = json.loads((prepared / "receipt.json").read_text())
    prior = json.loads(args.prior_comparison.read_text())
    blob = args.source_blob.resolve(strict=True)
    if (prep.get("schema") != "mx_gemmini.nicolas_tinyllama_lookup_source.v1"
            or prep["lookup_source_sha256"] != sha(source)
            or prep["embedding_table_sha256"] != sha(table)
            or prior.get("sha256", {}).get("source_blob") != sha(blob)):
        raise ValueError("lookup source or regenerated source blob differs")
    if blob.name != "llama_model_m160.bin" or table.name != "embedding_fp32.bin":
        raise ValueError("binary object symbols require the expected basenames")
    header_dir = args.source_header_dir.resolve(strict=True)
    header = header_dir / "llama_model_m160.h"
    if not header.is_file():
        raise ValueError("generated source header is missing")
    if prior.get("sha256", {}).get("source_header") != sha(header):
        raise ValueError("generated source header differs from archived source")
    npu = args.npu_root.resolve(strict=True)
    rocc = args.rtl_root.resolve(strict=True) / "software/gemmini-rocc-tests"
    riscv = args.riscv_root.resolve(strict=True)
    ext = args.extension.resolve(strict=True)
    bin_dir = riscv / "bin"
    cc = bin_dir / "riscv64-unknown-elf-gcc"
    ld = bin_dir / "riscv64-unknown-elf-ld"
    spike = bin_dir / "spike"
    bench = rocc / "riscv-tests/benchmarks/common"
    out = args.out_dir
    if shutil.disk_usage(out.parent).free < 12 * 1024**3:
        raise RuntimeError("need at least 12 GiB free for source objects and ELF")
    out.mkdir(parents=True, exist_ok=False)

    blob_obj = out / "llama_model_m160_blob.o"
    table_obj = out / "embedding_fp32.o"
    run([str(ld), "-r", "-b", "binary", "-o", str(blob_obj), blob.name],
        cwd=blob.parent, log=out / "blob_object.log")
    run([str(ld), "-r", "-b", "binary", "-o", str(table_obj), table.name],
        cwd=table.parent, log=out / "embedding_object.log")
    elf = out / "llama_model_lookup.elf"
    flags = [
        "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany",
        "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
        "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
        "-march=rv64gc", "-Wa,-march=rv64gc",
        f"-I{npu / 'baremetal/mxgemmini/include'}", f"-I{header_dir}",
        f"-I{rocc}", f"-I{rocc / 'riscv-tests'}",
        f"-I{rocc / 'riscv-tests/env'}", f"-I{bench}",
        f"-I{riscv / 'include/riscv'}", "-DBAREMETAL=1",
        "-DSPIKE_SIM", "-DLLAMA_VPU_RESIDUAL",
        '-DLLAMA_MODEL_HEADER="llama_model_m160.h"',
        "-DLLAMA_STAGE_TIMING", "-DLLAMA_DUMP_SESSION_LOGITS",
    ]
    build_command = [
        str(cc), *flags, "-nostdlib", "-nostartfiles", "-static",
        "-T", str(bench / "test.ld"), str(source),
        str(bench / "syscalls.c"), str(bench / "crt.S"),
        str(blob_obj), str(table_obj), "-lm", "-lgcc", "-o", str(elf),
    ]
    run(build_command, log=out / "build.log")
    build = {
        "schema": "mx_gemmini.nicolas_tinyllama_lookup_build.v1",
        "source_sha256": sha(source),
        "embedding_table_sha256": sha(table),
        "source_blob_sha256": sha(blob),
        "source_header_sha256": sha(header),
        "elf_sha256": sha(elf),
        "command": build_command,
        "spike_extension_config": "e4m3_vpu",
        "spike_mem_host_tracking": 0,
    }
    (out / "build_receipt.json").write_text(json.dumps(build, indent=2,
                                                       sort_keys=True) + "\n")

    env = os.environ.copy()
    env.update({"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
                "GEMMINI_PERF_SET": "mem.host_tracking=0",
                "GEMMINI_PERF_OUT": str(out / "source_perf.txt")})
    spike_command = [str(spike), "--isa=rv64gc_zicntr",
                     f"--extlib={ext}", "--extension=gemmini",
                     "-m0x80000000:0x2c0000000", str(elf)]
    log = out / "source_spike.log"
    run(spike_command, env=env, log=log)
    body = log.read_text()
    cycles = re.findall(r"^stage_cycles64 0x([0-9a-f]{16})$", body,
                        re.MULTILINE)
    rows = re.findall(r"^session_logits_bf16 (\d+) [0-9a-f]{128000}$",
                      body, re.MULTILINE)
    tops = re.findall(r"^source_argmax_160((?: \d+){160})$", body,
                      re.MULTILINE)
    if (len(cycles) != 1 or [int(row) for row in rows] != list(range(33))
            or len(tops) != 1
            or "llama MODEL benchmark finished (22 layer(s) + lm_head)." not in body):
        raise ValueError("source Spike run did not complete all 22 layers")
    receipt = {
        "schema": "mx_gemmini.nicolas_tinyllama_lookup_spike.v1",
        "build_receipt_sha256": sha(out / "build_receipt.json"),
        "log_sha256": sha(log),
        "extension_sha256": sha(ext),
        "spike_sha256": sha(spike),
        "spike_command": spike_command,
        "spike_environment": {key: env[key] for key in (
            "GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET")},
        "stage_cycles": int(cycles[0], 16),
        "emitted_logit_rows": len(rows),
        "emitted_top_tokens": len(tops[0].split()),
    }
    (out / "run_receipt.json").write_text(json.dumps(receipt, indent=2,
                                                     sort_keys=True) + "\n")
    print(json.dumps({"stage_cycles": receipt["stage_cycles"],
                      "emitted_logit_rows": len(rows), "out": str(out)}))


if __name__ == "__main__":
    main()
