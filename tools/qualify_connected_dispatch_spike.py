"""Compile a Merlin-selected BF16 VPU handoff and compare both paths on Spike."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import struct

from mx_gemmini_support.command_ir import emit_c
from mx_gemmini_support.connected_vpu import VpuDispatchAdapter, scalar_vpu_fragments
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_dense_attention import _perf_counters
from tools.qualify_vpu_row_softmax_spike import _run


def capture_graph(rows: int):
    import m2m
    import torch
    from m2m.coverage import opaque_report

    class ScalarChain(torch.nn.Module):
        def forward(self, x):
            return (x * 1.1015625) + 0.333984375

    model = ScalarChain().eval()
    values = torch.arange(rows * 8).reshape(rows, 8)
    example = (((values * 17) % 257 - 128).float() / 63.0).to(torch.bfloat16)
    capture = m2m.convert(model, (example,), backend="fx_importer", capture_trace=True)
    if not capture.ok or opaque_report(capture.mlir_text):
        raise ValueError("PyTorch scalar chain did not capture to complete typed MLIR")
    return capture.mlir_text, capture.capture_trace, example.view(torch.uint16).flatten().tolist(), \
        model(example).view(torch.uint16).flatten().tolist()


def build_case(profile: dict, rows: int):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects.lowering.connected_dispatch import HandoffFacts, plan_connected_dispatches
    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering.compute_groups import Group
    from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program, prune_dead_nodes
    from merlin.xdsl_dialects.lowering.outline import outline_dispatches

    source, trace, inputs, golden = capture_graph(rows)
    module = parse_mlir_text(source)
    function = next(op for op in module.walk() if op.name == "func.func")
    members = {}
    for op in function.body.block.ops:
        region = mq.attr_str(op, "prov.region_id")
        if region:
            members.setdefault(region, []).append(op)
    # The cast back to BF16 is part of each source operation's numerical
    # contract. Outline that whole source region before planning handoffs.
    groups = [Group(index, "vpu", None, operations) for index, operations in enumerate(members.values())]
    outlined = outline_dispatches(module, groups=groups)
    program = prune_dead_nodes(build_dispatch_program(outlined))
    if len(program.nodes) != 2 or any(node.kind != "dispatch" for node in program.nodes):
        raise ValueError("qualification requires exactly two outlined VPU dispatches")
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    fragments = scalar_vpu_fragments(outlined.module, program.to_dict(), profile, source_sha256=source_sha)
    adapter = VpuDispatchAdapter(program.to_dict(), profile, fragments)
    positions = {id(node): index for index, node in enumerate(program.nodes)}

    def evidence(producer, consumer, buffer):
        facts = adapter.handoff_facts(positions[id(producer)], positions[id(consumer)], buffer.id)
        return HandoffFacts(**facts) if facts else None

    plan = plan_connected_dispatches(program, evidence, require_profitable=False)
    return source, program, fragments, adapter, plan, trace, inputs, golden, outlined.module


def driver(elements: int, symbol: str, inputs: list[int], golden: list[int]) -> str:
    return f'''#include <stdint.h>
#include <stdio.h>
void {symbol}(const void *, const void *, const void *);
static const uint16_t input[{elements}] __attribute__((aligned(64))) = {{{','.join(map(str, inputs))}}};
static const uint16_t golden[{elements}] = {{{','.join(map(str, golden))}}};
static uint16_t middle[{elements}] __attribute__((aligned(64)));
static uint16_t output[{elements}] __attribute__((aligned(64)));
static uint16_t narrow(float x) {{ union {{float f; uint32_t u;}} w; w.f=x;
  w.u+=0x7fff+((w.u>>16)&1); return w.u>>16; }}
static float widen(uint16_t x) {{union {{float f; uint32_t u;}} w; w.u=(uint32_t)x<<16; return w.f;}}
static uint64_t cycle(void) {{uint64_t n; asm volatile("rdcycle %0" : "=r"(n) :: "memory"); return n;}}
int main(void) {{
  for (unsigned run=0;run<3;++run) {{
    uint64_t start=cycle(); {symbol}(input,middle,output); uint64_t elapsed=cycle()-start;
    unsigned mismatches=0, rounding_sensitive=0; uint32_t hash=2166136261u;
    for (unsigned i=0;i<{elements};++i) {{
      uint16_t want=narrow(widen(narrow(widen(input[i])*1.1015625f))+0.333984375f);
      rounding_sensitive+=want!=narrow(widen(input[i])*1.1015625f+0.333984375f);
      mismatches+=(want!=output[i]) || (golden[i]!=output[i]); hash=(hash^output[i])*16777619u;
    }}
    printf("HANDOFF run=%u cycles=%lu mismatches=%u hash=%u\\n",run,elapsed,mismatches,hash);
    printf("ROUNDING sensitive=%u\\n",rounding_sensitive);
    if(mismatches || !rounding_sensitive) return 3;
  }}
  return 0;
}}\n'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "rtl-root", "riscv-root", "extlib", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--startup-dir", type=Path, help="baremetal benchmarks/common directory")
    parser.add_argument("--model2mlir-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.rows <= 1024 or args.out_dir.exists():
        parser.error("rows must be 1..1024 and output directory must be fresh")
    sys.path.insert(0, str(args.model2mlir_root.resolve()))
    import m2m

    if Path(m2m.__file__).resolve().parents[1] != args.model2mlir_root.resolve():
        raise ValueError("model2MLIR resolved to another checkout")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    source, program, fragments, adapter, plan, trace, inputs, golden, outlined = build_case(profile, args.rows)
    selected = {(h.producer, h.consumer, h.buffer) for h in plan.handoffs}
    if len(selected) != 1:
        raise ValueError("generic planner did not select the expected private intermediate")
    from merlin.xdsl_dialects.lowering.connected_dispatch import require_realized
    from merlin.xdsl_dialects.lowering import connected_dispatch

    realized_plan, optimized = connected_dispatch.lower_connected_dispatches(
        program, adapter, require_profitable=False)
    if realized_plan != plan:
        raise ValueError("backend compilation changed the selected generic plan")

    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    (out / "source.mlir").write_text(source)
    (out / "outlined.mlir").write_text(str(outlined))
    (out / "capture_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    (out / "input.bin").write_bytes(struct.pack(f"<{len(inputs)}H", *inputs))
    (out / "golden.bin").write_bytes(struct.pack(f"<{len(golden)}H", *golden))
    (out / "dispatch_program.json").write_text(json.dumps(program.to_dict(), indent=2) + "\n")
    (out / "plan.json").write_text(json.dumps(asdict(plan), indent=2) + "\n")
    for index, fragment in enumerate(fragments):
        (out / f"kernel_{index}.mlir").write_text(fragment["mlir"])
    bench = (args.startup_dir.resolve() if args.startup_dir else
             args.rtl_root.resolve() / "software/gemmini-rocc-tests/riscv-tests/benchmarks/common")
    cc = args.riscv_root.resolve() / "bin/riscv64-unknown-elf-gcc"
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    results = {}
    for variant, edges in (("baseline", set()), ("connected", selected)):
        directory = out / variant
        directory.mkdir()
        lowered = optimized if edges else adapter.lower(set())
        if edges:
            require_realized(plan, set(lowered.realized))
        buffers = (program.nodes[0].inputs[0], program.nodes[0].outputs[0], program.results[0])
        (directory / "issue.c").write_text(emit_c(
            list(lowered.commands), transport="rocket_rocc", buffers=buffers, symbol="run_chain"))
        (directory / "driver.c").write_text(driver(args.rows * 8, "run_chain", inputs, golden))
        (directory / "commands.json").write_text(json.dumps(
            [asdict(command) for command in lowered.commands], indent=2) + "\n")
        objects = []
        for index, path in enumerate((directory / "issue.c", directory / "driver.c",
                                     *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S")))):
            obj = directory / f"part_{index}.o"
            _run([str(cc), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DBAREMETAL=1",
                  "-mcmodel=medany", "-O2", "-fno-common", "-fno-builtin-printf",
                  "-fno-tree-loop-distribute-patterns", "-march=rv64gc_zicntr",
                  "-I", str(bench), "-I", str(bench.parent.parent),
                  "-I", str(bench.parent.parent / "env"),
                  "-c", str(path), "-o", str(obj)], directory, f"compile_{index}.log")
            objects.append(obj)
        elf = directory / "chain.elf"
        _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T", str(bench / "test.ld"),
              *(str(obj) for obj in objects), "-lm", "-lgcc", "-o", str(elf)], directory, "link.log")
        env = os.environ | {"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
                            "GEMMINI_PERF_SET": "mem.host_tracking=0",
                            "GEMMINI_PERF_OUT": str(directory / "perf.txt")}
        command = [str(args.riscv_root.resolve() / "bin/spike"), "--isa=rv64gc_zicntr",
                   f"--extlib={args.extlib.resolve()}", "--extension=gemmini", str(elf)]
        run = subprocess.run(command, cwd=directory, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        (directory / "spike.log").write_text(run.stdout)
        checks = re.findall(r"HANDOFF run=(\d+) cycles=(\d+) mismatches=(\d+) hash=(\d+)", run.stdout)
        if run.returncode or len(checks) != 3 or any(int(row[2]) for row in checks):
            raise RuntimeError(f"{variant} failed output qualification; see {directory / 'spike.log'}")
        results[variant] = dict(
            checks=[[int(value) for value in row] for row in checks],
            counters=_perf_counters((directory / "perf.txt").read_text()),
            removed_transfer_bytes_per_call=lowered.removed_transfer_bytes,
            command_count=len(lowered.commands), issuer_sha256=sha(directory / "issue.c"),
            elf_sha256=sha(elf), spike_log_sha256=sha(directory / "spike.log"),
            commands_sha256=sha(directory / "commands.json"), driver_sha256=sha(directory / "driver.c"),
            perf_sha256=sha(directory / "perf.txt"), command=command)
    if len({row[3] for variant in results.values() for row in variant["checks"]}) != 1:
        raise ValueError("baseline and connected output hashes differ")
    baseline, connected = results["baseline"], results["connected"]
    if baseline["driver_sha256"] != connected["driver_sha256"]:
        raise ValueError("paired programs have different input/checker/timer code")
    if baseline["counters"]["load_bytes"] - connected["counters"]["load_bytes"] != args.rows * 16 * 3:
        raise ValueError("measured load reduction disagrees with selected resident edge")
    saving = min(row[1] for row in baseline["checks"][1:]) - max(row[1] for row in connected["checks"][1:])
    calibrated = VpuDispatchAdapter(program.to_dict(), profile, fragments,
                                    measured_savings={edge: saving for edge in selected})
    calibrated_plan, calibrated_code = connected_dispatch.lower_connected_dispatches(program, calibrated)
    expected_code = optimized if saving > 0 else adapter.lower(set())
    if calibrated_code.commands != expected_code.commands:
        raise ValueError("measured production selection differs from the qualified command stream")
    (out / "calibrated_plan.json").write_text(json.dumps(asdict(calibrated_plan), indent=2) + "\n")
    receipt = dict(schema="mx_gemmini.connected_dispatch_spike.v1", rows=args.rows,
                   profile_sha256=profile_sha256(profile), extension_sha256=sha(args.extlib),
                   source_mlir_sha256=sha(out / "source.mlir"), results=results,
                   model2mlir_revision=subprocess.check_output(
                       ["git", "-C", str(args.model2mlir_root), "rev-parse", "HEAD"], text=True).strip(),
                   input_sha256=sha(out / "input.bin"), golden_sha256=sha(out / "golden.bin"),
                   capture_trace_sha256=sha(out / "capture_trace.json"),
                   measured_warm_saved_cycles=saving,
                   production_selected=bool(calibrated_plan.handoffs),
                   calibrated_plan_sha256=sha(out / "calibrated_plan.json"),
                   timing=dict(GEMMINI_MODE="both", GEMMINI_PERF_CONFIG="e4m3_vpu",
                               GEMMINI_PERF_SET="mem.host_tracking=0"),
                   planner_sha256=sha(Path(connected_dispatch.__file__)),
                   adapter_sha256=sha(Path(__file__).resolve().parents[1] / "mx_gemmini_support/connected_vpu.py"),
                   runner_sha256=sha(Path(__file__)), gcc_sha256=sha(cc),
                   spike_sha256=sha(args.riscv_root.resolve() / "bin/spike"),
                   startup_sha256={path.name: sha(path) for path in (
                       *bench.glob("*.c"), *bench.glob("*.S"), bench / "test.ld")})
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
