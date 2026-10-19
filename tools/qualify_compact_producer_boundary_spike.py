"""Check all compact producer words and the original explicit NaN trap on Spike."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from math import prod
from pathlib import Path

import numpy as np

from mx_gemmini_support.merlin_connected_multiply import (
    emit_chain_kernel,
    multiply_pairs,
)
from mx_gemmini_support.merlin_vpu_elementwise import selected_elementwise_sites
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_vpu_row_softmax_spike import _run


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def driver(count, inputs, dtype):
    compact = dtype == "bf16"
    scalar, guard = ("uint16_t", 32) if compact else ("float", 16)
    sentinel = "0xa5a5u" if compact else "word_float(0xa5a50000u)"
    read = "storage[i]" if compact else "word_bits(storage[i])"
    normalized = "word" if compact else "word>>16"
    low = "0" if compact else "word&65535u"
    return f'''#include <stdint.h>
#include <stdio.h>
extern const unsigned char payload[];
void produce(const float *,const float *,const float *,{scalar} *);
void __attribute__((noreturn)) tohost_exit(uintptr_t);
static {scalar} storage[{count + 2 * guard}] __attribute__((aligned(64)));
static float word_float(uint32_t bits){{union{{float f;uint32_t u;}}w;w.u=bits;return w.f;}}
static uint32_t word_bits(float value){{union{{float f;uint32_t u;}}w;w.f=value;return w.u;}}
static uint32_t raw(unsigned i){{return {read};}}
static void dump(void) {{
  unsigned guard=0,low=0;
  for(unsigned i=0;i<{guard};++i)guard+=raw(i)!={0xa5a5 if compact else 0xa5a50000}u ||
    raw({count + guard}u+i)!={0xa5a5 if compact else 0xa5a50000}u;
  for(unsigned i=0;i<{count};++i) {{
    if(i%32u==0)printf("WORDS_CHUNK %u ",i);
    uint32_t word=raw(i+{guard});low|={low};
    printf("%04x ",(unsigned)({normalized}));
    if(i%32u==31u || i=={count - 1}u)printf("\\n");
  }}
  printf("WORDS_STATUS guard=%u low_bits=%u\\n",guard,low);
}}
uintptr_t handle_trap(uintptr_t cause,uintptr_t pc,uintptr_t regs[32]) {{
  (void)regs;dump();printf("REFUSAL_TRAP cause=%lu pc=%lu\\n",(unsigned long)cause,(unsigned long)pc);
  tohost_exit(77);
}}
int main(void) {{
  for(unsigned i=0;i<{count + 2 * guard};++i)storage[i]={sentinel};
  const float *a=(const float *)payload,*b=a+{inputs[0]},*c=b+{inputs[1]};
  printf("BEGIN\\n");produce(a,b,c,storage+{guard});
  dump();printf("DONE\\n");return 0;
}}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["capture", "profile", "riscv-root", "extlib", "startup-dir", "out-dir"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expect-refusal", action="store_true")
    parser.add_argument("--structural-elf-admission", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists() or not args.structural_elf_admission:
        parser.error("fresh output directory and explicit structural admission required")
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.device_build import (
        DeviceRouting,
        admit_linked_device_elf,
        freeze_device_elf_admission,
    )
    from merlin.model_check.admission import service

    capture = args.capture.resolve(strict=True)
    recorded = json.loads((capture / "receipt.json").read_text())
    for name in ["source.mlir", "inputs.npz"]:
        if sha(capture / name) != recorded["files_sha256"][name]:
            raise ValueError("producer capture changed")
    profile = load_profile(args.profile)
    pair, = multiply_pairs(parse_mlir_file(capture / "source.mlir"), selected_elementwise_sites(
        (capture / "source.mlir").read_text(), profile), include_broadcasts=True)
    count = prod(pair["shape"])
    if count > 1048576:
        raise ValueError("boundary observer requires bounded complete output")
    with np.load(capture / "inputs.npz", allow_pickle=False) as stored:
        arrays = [stored[f"in{i}"] for i in range(3)]
    if any(a.dtype != np.float32 or list(a.shape) != shape for a, shape in zip(arrays, pair["input_shapes"], strict=True)):
        raise ValueError("input signature differs from captured source")
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    for name in ["source.mlir", "inputs.npz", "receipt.json"]:
        (out / ("capture_" + name)).write_bytes((capture / name).read_bytes())
    (out / "payload.bin").write_bytes(b"".join(a.astype("<f4").tobytes() for a in arrays))
    (out / "payload.S").write_text('.section .rodata\n.balign 64\n.global payload\npayload:\n.incbin "payload.bin"\n')
    startup, riscv = args.startup_dir.resolve(strict=True), args.riscv_root.resolve(strict=True)
    cc, objdump = riscv / "bin/riscv64-unknown-elf-gcc", riscv / "bin/riscv64-unknown-elf-objdump"
    flags = [str(cc), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DBAREMETAL=1", "-mcmodel=medany", "-O2",
             "-fno-common", "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns", "-march=rv64gc_zicntr",
             "-I", str(startup), "-I", str(startup.parent.parent), "-I", str(startup.parent.parent / "env")]
    route = DeviceRouting(device="mx_gemmini", package_dir=Path(__file__).resolve().parents[1],
                          operand_dtype="f32", accum_dtype="bf16", linked_elf_admission=service("mx_gemmini"))
    selection = freeze_device_elf_admission(route)
    results, outputs = {}, []
    commands = []
    for dtype in ["f32", "bf16"]:
        directory = out / dtype
        directory.mkdir()
        selected = pair | {"symbol": "produce", "kernel_symbol": "produce"}
        source, _ = emit_chain_kernel(selected, profile, prepare_broadcasts=True, output_dtype=dtype)
        (directory / "producer.c").write_text(source)
        (directory / "driver.c").write_text(driver(count, [a.size for a in arrays], dtype))
        objects = []
        for index, path in enumerate([directory / "producer.c", directory / "driver.c", out / "payload.S",
                                      *sorted(startup.glob("*.c")), *sorted(startup.glob("*.S"))]):
            obj = directory / f"object{index}.o"
            command = [*flags, "-c", str(path), "-o", str(obj)]
            commands.append(command)
            _run(command, out, f"{dtype}_compile{index}.log")
            objects.append(obj)
        elf = directory / "section.elf"
        command = [str(cc), "-nostdlib", "-nostartfiles", "-static", "-T", str(startup / "test.ld"),
                   *map(str, objects), "-lm", "-lgcc", "-o", str(elf)]
        commands.append(command)
        _run(command, out, dtype + "_link.log")
        admission = admit_linked_device_elf(route, selection, elf=elf, linked_sha256=sha(elf), directory=directory)
        environment = os.environ | {"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
                                    "GEMMINI_PERF_SET": "mem.host_tracking=0", "GEMMINI_PERF_OUT": str(directory / "perf.txt")}
        command = [str(riscv / "bin/spike"), "--isa=rv64gc_zicntr", "--extlib=" + str(args.extlib.resolve()),
                   "--extension=gemmini", str(elf)]
        with (directory / "spike.log").open("wb") as log:
            result = subprocess.run(command, cwd=out, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=600, check=False)
        lines = (directory / "spike.log").read_text().splitlines()
        words, traps, statuses = [], [], []
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "WORDS_CHUNK":
                if int(parts[1]) != len(words):
                    raise ValueError("producer output chunk order changed")
                words.extend(int(v, 16) for v in parts[2:])
            elif parts[0] in {"REFUSAL_TRAP", "WORDS_STATUS"}:
                values = {k: int(v) for k, v in (p.split("=", 1) for p in parts[1:])}
                (traps if parts[0] == "REFUSAL_TRAP" else statuses).append(values)
        if (len(words) != count or len(statuses) != 1 or statuses[0] != {"guard": 0, "low_bits": 0}
                or lines.count("BEGIN") != 1):
            raise ValueError("producer did not publish complete normalized boundary observations")
        if args.expect_refusal:
            if result.returncode != 77 or len(traps) != 1 or traps[0]["cause"] != 3 or "DONE" in lines:
                raise ValueError("original explicit NaN refusal changed")
            pc = traps[0]["pc"]
            disassembly = subprocess.check_output([str(objdump), "-d", f"--start-address={pc}",
                                                   f"--stop-address={pc + 4}", str(elf)], text=True)
            decoded = [s.split() for s in disassembly.splitlines() if s.split() and s.split()[0] == f"{pc:x}:"]
            if len(decoded) != 1 or decoded[0][-1] != "ebreak":
                raise ValueError("observed refusal does not identify the actual explicit breakpoint")
            (directory / "refusal_disassembly.log").write_text(disassembly)
        elif result.returncode or traps or lines.count("DONE") != 1:
            raise ValueError("producer changed successful return behavior")
        raw = np.asarray(words, dtype="<u2").tobytes()
        (directory / "observed_words.bin").write_bytes(raw)
        outputs.append(raw)
        results[dtype] = {"exit_code": result.returncode, "word_sha256": hashlib.sha256(raw).hexdigest(),
                          "traps": traps, "status": statuses[0], "command": command, "elf_admission": admission,
                          "environment": {k: environment[k] for k in ["GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET"]}}
    if outputs[0] != outputs[1]:
        raise ValueError("compact producer changes normalized words or partial output before refusal")
    report = {"schema": "mx_gemmini.compact_producer_boundary.v1", "scope": "complete_boundary_differential_and_explicit_refusal_no_timing_or_independent_quality",
              "shape": pair["shape"], "input_shapes": pair["input_shapes"], "expect_refusal": args.expect_refusal,
              "results": results, "commands": commands, "profile_sha256": profile_sha256(profile),
              "extension_sha256": sha(args.extlib), "gcc_sha256": sha(cc), "objdump_sha256": sha(objdump),
              "spike_sha256": sha(riscv / "bin/spike"),
              "files_sha256": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}}
    (out / "receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"words": count, "refusal": args.expect_refusal, "equal": True}), flush=True)


if __name__ == "__main__":
    main()
