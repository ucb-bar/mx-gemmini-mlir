"""Qualify whole-stage multiply selection, its ranked ABI, and generated kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from math import prod
from pathlib import Path

from mx_gemmini_support.merlin_connected_multiply import (
    chain_adapter,
    emit_chain_kernel,
    emit_chain_shim,
    multiply_pairs,
)
from mx_gemmini_support.merlin_vpu_elementwise import (
    MXVPUElementwisePostOffload,
    selected_elementwise_sites,
)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_connected_elementwise_spike import driver
from tools.qualify_nicolas_dense_attention import _perf_counters
from tools.qualify_vpu_row_softmax_spike import _run


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "profile", "riscv-root", "extlib", "startup-dir", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--expect-fallback", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("output directory must be fresh")
    import numpy as np
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.xdsl_dialects.lowering import connected_dispatch, tensor_chains

    capture = args.capture.resolve(strict=True)
    receipt = json.loads((capture / "receipt.json").read_text())
    for name in ("source.mlir", "inputs.npz", "bf16_policy_golden.npy"):
        if receipt["files_sha256"][name] != _sha(capture / name):
            raise ValueError("capture changed: " + name)
    profile = load_profile(args.profile)
    source = capture / "source.mlir"
    pair, = multiply_pairs(parse_mlir_file(source), selected_elementwise_sites(source.read_text(), profile))
    _, adapter = chain_adapter(pair, profile)
    code, removed = emit_chain_kernel(pair, profile)
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    sidecar = out / "sidecar.json"
    sidecar.write_text("{}")
    rewritten = MXVPUElementwisePostOffload(profile, connect_multiplies=True)(source, sidecar, out / "rewrite")
    module = parse_mlir_file(rewritten)
    module.verify()
    calls = [op for op in module.walk() if op.name == "func.call"]
    if len(calls) != 1 or calls[0].callee.root_reference.data != pair["symbol"]:
        raise ValueError("stage rewrite and compiled kernel disagree")
    with np.load(capture / "inputs.npz", allow_pickle=False) as stored:
        arrays = [stored[f"in{i}"] for i in range(3)]
    arrays.append(np.load(capture / "bf16_policy_golden.npy", allow_pickle=False))
    if any(list(a.shape) != pair["shape"] or a.dtype != np.float32 for a in arrays):
        raise ValueError("capture arrays differ from source signatures")
    (out / "payload.bin").write_bytes(b"".join(a.astype("<f4").tobytes() for a in arrays))
    (out / "payload.S").write_text('.section .rodata\n.balign 64\n.global payload\npayload:\n.incbin "payload.bin"\n')
    (out / "driver.c").write_text(driver(adapter.elements))
    cc = args.riscv_root.resolve() / "bin/riscv64-unknown-elf-gcc"
    bench = args.startup_dir.resolve()
    flags = [str(cc), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DBAREMETAL=1",
             "-mcmodel=medany", "-O2", "-fno-common", "-fno-builtin-printf",
             "-fno-tree-loop-distribute-patterns", "-march=rv64gc_zicntr",
             "-I", str(bench), "-I", str(bench.parent.parent), "-I", str(bench.parent.parent / "env")]
    common = []
    for i, path in enumerate((out / "driver.c", out / "payload.S", *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S")))):
        obj = out / f"common_{i}.o"
        _run([*flags, "-c", str(path), "-o", str(obj)], out, f"compile_{i}.log")
        common.append(obj)
    shape = pair["shape"]
    dims = ",".join(map(str, shape))
    strides = ",".join(str(prod(shape[i + 1:])) for i in range(len(shape)))
    # Exercise the actual generated ranked C interface, not only the pointer kernel.
    bridge = f'''
    void run_chain(const float *a,const float *b,const float *c,float *middle,float *o) {{
      mx_chain_memref aa={{(void *)a,(void *)a,0,{{{dims}}},{{{strides}}}}};
      mx_chain_memref bb={{(void *)b,(void *)b,0,{{{dims}}},{{{strides}}}}};
      mx_chain_memref cc={{(void *)c,(void *)c,0,{{{dims}}},{{{strides}}}}};
      mx_chain_memref oo={{o,o,0,{{{dims}}},{{{strides}}}}},result;
      _mlir_ciface_{pair['symbol']}(&result,&aa,&bb,&cc,&oo);
      if(result.aligned!=o || result.offset) __builtin_trap();
    }}\n'''
    results = {}
    for name, text in (("materialized", adapter.lower(set()).source),
                       ("connected", code + emit_chain_shim(pair) + bridge)):
        directory = out / name
        directory.mkdir()
        issuer, obj, elf = directory / "issue.c", directory / "issue.o", directory / "section.elf"
        issuer.write_text(text)
        _run([*flags, "-c", str(issuer), "-o", str(obj)], directory, "compile.log")
        _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T", str(bench / "test.ld"),
              str(obj), *map(str, common), "-lm", "-lgcc", "-o", str(elf)], directory, "link.log")
        env = os.environ | {"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
                            "GEMMINI_PERF_SET": "mem.host_tracking=0", "GEMMINI_PERF_OUT": str(directory / "perf.txt")}
        command = [str(args.riscv_root.resolve() / "bin/spike"), "--isa=rv64gc_zicntr",
                   f"--extlib={args.extlib.resolve()}", "--extension=gemmini", str(elf)]
        log = directory / "spike.log"
        with log.open("w") as stream:
            run = subprocess.run(command, cwd=directory, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, timeout=600, check=False)
        rows = [dict(word.split("=", 1) for word in line.split()[1:])
                for line in log.read_text().splitlines() if line.startswith("SECTION ")]
        if run.returncode or len(rows) != 3 or any(int(row["mismatches"]) for row in rows):
            raise ValueError("output qualification failed: " + str(log))
        results[name] = {"checks": rows, "counters": _perf_counters((directory / "perf.txt").read_text()),
                             "files_sha256": {p.name: _sha(p) for p in (issuer, obj, elf, log, directory / "perf.txt")}}
    if len({row["hash"] for r in results.values() for row in r["checks"]}) != 1:
        raise ValueError("paired output hashes differ")
    a, b = results["materialized"], results["connected"]
    expected = 0 if args.expect_fallback else removed // 2 * 3
    if a["counters"]["load_bytes"] - b["counters"]["load_bytes"] != expected:
        raise ValueError("load saving differs from selected handoff")
    report = {"schema": "mx_gemmini.stage_multiply_spike.v1", "shape": shape, "results": results,
                  "expected_fallback": args.expect_fallback, "boundary_policy": "bf16_rne_each_vpu_call",
                  "removed_transfer_bytes_per_call": 0 if args.expect_fallback else removed,
                  "warm_saved_cycles": min(int(r["cycles"]) for r in a["checks"][1:]) -
                                    max(int(r["cycles"]) for r in b["checks"][1:]),
                  "profile_sha256": profile_sha256(profile), "extension_sha256": _sha(args.extlib),
                  "gcc_sha256": _sha(cc), "spike_sha256": _sha(args.riscv_root / "bin/spike"),
                  "profile_source_verification": "archived_profile_only_no_live_rtl_check",
                  "capture_receipt": receipt, "command": command,
                  "source_files_sha256": {name: _sha(out / name) for name in
                                       ("driver.c", "payload.S", "payload.bin")},
                  "implementation_sha256": {str(p): _sha(p) for p in (
                      Path(__file__).resolve(), Path(tensor_chains.__file__), Path(connected_dispatch.__file__),
                      Path(__file__).resolve().parents[1] / "mx_gemmini_support/merlin_connected_multiply.py",
                      Path(__file__).resolve().parents[1] / "mx_gemmini_support/merlin_vpu_elementwise.py",
                      Path(__file__).resolve().parents[1] / "mx_gemmini_support/connected_elementwise.py")}}
    (out / "receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
