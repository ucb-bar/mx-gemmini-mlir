"""Qualify a captured f32 multiply section under its selected BF16 VPU policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

from mx_gemmini_support.connected_elementwise import F32MultiplyAdapter
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_dense_attention import _perf_counters
from tools.qualify_vpu_row_softmax_spike import _run


def driver(elements: int, *, input_elements: tuple[int, int, int] | None = None) -> str:
    counts = input_elements or (elements,) * 3
    if len(counts) != 3 or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError("multiply driver needs three positive input extents")
    return f'''#include <stdint.h>
#include <stdio.h>
extern const unsigned char payload[];
void run_chain(const float *,const float *,const float *,float *,float *);
static float middle[{elements}] __attribute__((aligned(64)));
static float output[{elements}] __attribute__((aligned(64)));
static uint64_t cycle(void) {{ uint64_t n;
  asm volatile("rdcycle %0" : "=r"(n) :: "memory"); return n; }}
int main(void) {{
  const float *a=(const float *)payload, *b=a+{counts[0]}, *c=b+{counts[1]};
  const uint32_t *golden=(const uint32_t *)(c+{counts[2]});
  for(unsigned run=0;run<3;++run) {{
    uint64_t start=cycle(); run_chain(a,b,c,middle,output);
    uint64_t elapsed=cycle()-start;
    unsigned mismatches=0; uint32_t hash=2166136261u;
    for(unsigned i=0;i<{elements};++i) {{
      union {{float f;uint32_t u;}} word; word.f=output[i];
      mismatches+=word.u!=golden[i]; hash=(hash^word.u)*16777619u;
    }}
    printf("SECTION run=%u cycles=%lu mismatches=%u hash=%u\\n",run,elapsed,mismatches,hash);
    if(mismatches) return 3;
  }}
  return 0;
}}\n'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "profile", "rtl-root", "riscv-root", "extlib", "startup-dir", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--boundary-policy", required=True, choices=("bf16_rne_each_vpu_call",))
    parser.add_argument("--expect-fallback", action="store_true",
                        help="qualify an out-of-range fixture that must execute the original kernels")
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("output directory must be fresh")
    import numpy as np
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering import connected_dispatch
    from merlin.xdsl_dialects.lowering.dispatch_program import (
        lower_model_to_dispatch_program,
    )

    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    source = (args.capture / "source.mlir").read_text()
    outlined, program = lower_model_to_dispatch_program(parse_mlir_text(source))
    adapter = F32MultiplyAdapter(outlined.module, program.to_dict(), profile,
                                 boundary_policy=args.boundary_policy)
    plan, optimized = connected_dispatch.lower_connected_dispatches(program, adapter, require_profitable=False)
    if len(plan.handoffs) != 1:
        raise ValueError("capture does not have one legal private multiply intermediate")
    with np.load(args.capture / "inputs.npz", allow_pickle=False) as stored:
        arrays = [stored[f"in{i}"] for i in range(3)]
    arrays.append(np.load(args.capture / "bf16_policy_golden.npy", allow_pickle=False))
    if any(tuple(a.shape) != adapter.shape or a.dtype != np.float32 for a in arrays):
        raise ValueError("capture tensors differ from the typed f32 dispatch signatures")
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    payload = out / "payload.bin"
    payload.write_bytes(b"".join(a.astype("<f4").tobytes() for a in arrays))
    (out / "source.mlir").write_text(source)
    (out / "outlined.mlir").write_text(str(outlined.module))
    (out / "dispatch_program.json").write_text(json.dumps(program.to_dict(), indent=2) + "\n")
    (out / "plan.json").write_text(json.dumps(asdict(plan), indent=2) + "\n")
    (out / "driver.c").write_text(driver(adapter.elements))
    (out / "payload.S").write_text(
        '.section .rodata\n.balign 64\n.global payload\npayload:\n.incbin "payload.bin"\n')
    bench = args.startup_dir.resolve()
    cc = args.riscv_root.resolve() / "bin/riscv64-unknown-elf-gcc"
    flags = [str(cc), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DBAREMETAL=1",
             "-mcmodel=medany", "-O2", "-fno-common", "-fno-builtin-printf",
             "-fno-tree-loop-distribute-patterns", "-march=rv64gc_zicntr",
             "-I", str(bench), "-I", str(bench.parent.parent),
             "-I", str(bench.parent.parent / "env")]
    common = []
    for i, path in enumerate((out / "driver.c", out / "payload.S",
                              *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S")))):
        obj = out / f"common_{i}.o"
        _run([*flags, "-c", str(path), "-o", str(obj)], out, f"compile_common_{i}.log")
        common.append(obj)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    results = {}
    for name, code in (("materialized", adapter.lower(set())), ("connected", optimized)):
        directory = out / name
        directory.mkdir()
        (directory / "issue.c").write_text(code.source)
        obj, elf = directory / "issue.o", directory / "section.elf"
        _run([*flags, "-c", str(directory / "issue.c"), "-o", str(obj)], directory, "compile.log")
        _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T", str(bench / "test.ld"),
              str(obj), *map(str, common), "-lm", "-lgcc", "-o", str(elf)], directory, "link.log")
        env = os.environ | {"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
                            "GEMMINI_PERF_SET": "mem.host_tracking=0",
                            "GEMMINI_PERF_OUT": str(directory / "perf.txt")}
        command = [str(args.riscv_root.resolve() / "bin/spike"), "--isa=rv64gc_zicntr",
                   f"--extlib={args.extlib.resolve()}", "--extension=gemmini", str(elf)]
        with (directory / "spike.log").open("w") as log:
            run = subprocess.run(command, cwd=directory, env=env, text=True,
                                 stdout=log, stderr=subprocess.STDOUT, timeout=600)
        checks = re.findall(r"SECTION run=(\d+) cycles=(\d+) mismatches=(\d+) hash=(\d+)",
                            (directory / "spike.log").read_text())
        if run.returncode or len(checks) != 3 or any(int(row[2]) for row in checks):
            raise RuntimeError(f"{name} failed output qualification; see {directory / 'spike.log'}")
        results[name] = dict(checks=[[int(v) for v in row] for row in checks],
                             counters=_perf_counters((directory / "perf.txt").read_text()),
                             removed_transfer_bytes_per_call=code.removed_transfer_bytes,
                             files_sha256={p.name: sha(p) for p in (
                                 directory / "issue.c", elf, directory / "spike.log", directory / "perf.txt")})
        print(json.dumps({name: results[name]}), flush=True)
    if len({row[3] for result in results.values() for row in result["checks"]}) != 1:
        raise ValueError("paired programs have different output hashes")
    baseline, connected = results["materialized"], results["connected"]
    expected_load_saving = 0 if args.expect_fallback else optimized.removed_transfer_bytes // 2 * 3
    if baseline["counters"]["load_bytes"] - connected["counters"]["load_bytes"] != expected_load_saving:
        raise ValueError("measured load reduction differs from the resident handoff")
    saving = min(row[1] for row in baseline["checks"][1:]) - max(row[1] for row in connected["checks"][1:])
    measured = F32MultiplyAdapter(outlined.module, program.to_dict(), profile,
                                 boundary_policy=args.boundary_policy,
                                 measured_savings={(h.producer, h.consumer, h.buffer): saving for h in plan.handoffs})
    production_plan, production_code = connected_dispatch.lower_connected_dispatches(program, measured)
    expected = optimized if saving > 0 else adapter.lower(set())
    if production_code != expected:
        raise ValueError("production lowering differs from the qualified selection")
    (out / "calibrated_plan.json").write_text(json.dumps(asdict(production_plan), indent=2) + "\n")
    receipt = dict(schema="mx_gemmini.connected_elementwise_spike.v1", shape=list(adapter.shape),
                   boundary_policy=args.boundary_policy,
                   expected_fallback=args.expect_fallback,
                   elements=adapter.elements, results=results, measured_warm_saved_cycles=saving,
                   production_selected=bool(production_plan.handoffs), profile_sha256=profile_sha256(profile),
                   extension_sha256=sha(args.extlib), gcc_sha256=sha(cc),
                   spike_sha256=sha(args.riscv_root.resolve() / "bin/spike"),
                   capture_receipt=json.loads((args.capture / "receipt.json").read_text()),
                   planner_sha256=sha(Path(connected_dispatch.__file__)),
                   implementation_sha256={path.name: sha(path) for path in (
                       Path(__file__), Path(__file__).parents[1] / "mx_gemmini_support/connected_elementwise.py",
                       Path(__file__).parents[1] / "mx_gemmini_support/connected_vpu.py",
                       Path(__file__).parents[1] / "mx_gemmini_support/vpu_add_kernel.py")},
                   timing=dict(GEMMINI_MODE="both", GEMMINI_PERF_CONFIG="e4m3_vpu",
                               GEMMINI_PERF_SET="mem.host_tracking=0"),
                   files_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.suffix != ".o"})
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
