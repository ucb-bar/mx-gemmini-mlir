"""Measure complete native batched adapters with exact contiguous RHS repetition.

All host padding/packing, issuer calls and output conversion are timed. Complete
words, independent integer goldens, exterior guards and exceptional partial
outputs are checked outside the timer. This is a section, not a model result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from mx_gemmini_support.dense_contraction import emit_dense_batch_contraction
from mx_gemmini_support.dense_kernel import emit_dense_f32_batch_kernel
from mx_gemmini_support.runtime_group_preparation import emit_runtime_group_preparation
from mx_gemmini_support.target_profile import load_profile, profile_sha256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def driver(batch, m, n, k, repeat, *, refusal, compact_rhs=False):
    count, ae, be = batch * m * n, batch * m * k, batch * k * n
    if compact_rhs:
        be //= repeat
    source_index = f"(i/{k * n}u)*{repeat * k * n}u+i%{k * n}u" if compact_rhs else "i"
    copies = 1 if compact_rhs else repeat
    return f'''#include <stdint.h>
#include <stdio.h>
extern const unsigned char payload[];
void produce(const float *,const float *,float *);
void __attribute__((noreturn)) tohost_exit(uintptr_t);
static float a[{ae + 32}],b[{be + 32}],out[{count + 32}] __attribute__((aligned(64)));
static unsigned run;
static float value(uint32_t u){{union{{float f;uint32_t u;}}w;w.u=u;return w.f;}}
static uint32_t bits(float f){{union{{float f;uint32_t u;}}w;w.f=f;return w.u;}}
static uint64_t cycle(void){{uint64_t n;asm volatile("rdcycle %0":"=r"(n)::"memory");return n;}}
static void dump(void){{
 unsigned guard=0;
 for(unsigned i=0;i<16;++i)guard+=bits(a[i])!=0xa5a50000u||bits(a[{ae}+16+i])!=0xa5a50000u||bits(b[i])!=0xa5a50000u||bits(b[{be}+16+i])!=0xa5a50000u||bits(out[i])!=0xa5a50000u||bits(out[{count}+16+i])!=0xa5a50000u;
 for(unsigned i=0;i<{count};++i){{if(i%16==0)printf("WORDS %u %u ",run,i);printf("%08x ",bits(out[16+i]));if(i%16==15||i=={count - 1})printf("\\n");}}
 printf("STATUS run=%u guard=%u\\n",run,guard);
}}
uintptr_t handle_trap(uintptr_t cause,uintptr_t pc,uintptr_t regs[32]){{
 (void)regs;dump();printf("TRAP cause=%lu pc=%lu\\n",(unsigned long)cause,(unsigned long)pc);tohost_exit(77);
}}
int main(void){{
 for(run=0;run<{1 if refusal else 3};++run){{
  for(unsigned i=0;i<{ae + 32};++i)a[i]=value(0xa5a50000u);
  for(unsigned i=0;i<{be + 32};++i)b[i]=value(0xa5a50000u);
  for(unsigned i=0;i<{count + 32};++i)out[i]=value(0xa5a50000u);
  for(unsigned i=0;i<{ae};++i)a[i+16]=((const float*)payload)[i];
  for(unsigned i=0;i<{be};++i)b[i+16]=((const float*)payload)[{ae}+{source_index}];
  if(!{int(refusal)})for(unsigned group=0;group<{batch // repeat};++group)if((run==1&&group==0)||(run==2&&group==1))for(unsigned r=0;r<{copies};++r)b[16+(group*{copies}+r)*{k * n}]=-b[16+(group*{copies}+r)*{k * n}];
  uint64_t t=cycle();produce(a+16,b+16,out+16);t=cycle()-t;
  printf("PERF run=%u cycles=%lu\\n",run,(unsigned long)t);dump();
 }}
 printf("DONE\\n");return 0;
}}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "riscv-root", "extlib", "startup-dir", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("batch", "m", "n", "k", "repeat"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--precision", choices=("FP4", "FP8"), default="FP8")
    parser.add_argument("--virtual-zero-padding", action="store_true")
    parser.add_argument("--compact-rhs", action="store_true",
                        help="Give the retained endpoint one original RHS matrix per group")
    parser.add_argument("--refusal", choices=("none", "lhs", "rhs"), default="none")
    parser.add_argument("--refusal-bits", type=lambda word: int(word, 0), default=0x7FC12345)
    parser.add_argument("--structural-elf-admission", action="store_true")
    args = parser.parse_args()
    batch, m, n, k, repeat = args.batch, args.m, args.n, args.k, args.repeat
    if (args.out_dir.exists() or not args.structural_elf_admission
            or min(batch, m, n, k) <= 0 or repeat <= 1 or batch % repeat
            or batch // repeat < 2 or batch * m * n > 1048576
            or max(batch * m * k, batch * k * n) > 16777216 or k > 256):
        parser.error("fresh admitted bounded shapes and at least two complete repeated groups required")
    from merlin.llvmlower.device_build import (
        DeviceRouting,
        admit_linked_device_elf,
        freeze_device_elf_admission,
    )
    from merlin.model_check.admission import service

    profile = load_profile(args.profile)
    generated = emit_dense_batch_contraction(profile, (batch, m, n, k), precision=args.precision,
                                             symbol="produce", issuer_symbol="native_issue",
                                             virtual_zero_padding=args.virtual_zero_padding)
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    rng = np.random.default_rng(64219)
    a = (rng.integers(0, 2, (batch, m, k), dtype=np.int8) * 2 - 1).astype("<f4")
    groups = (rng.integers(0, 2, (batch // repeat, k, n), dtype=np.int8) * 2 - 1).astype("<f4")
    b = np.repeat(groups, repeat, axis=0)
    if args.refusal == "lhs":
        a.view("<u4")[repeat + 1, 0, 0] = args.refusal_bits
    elif args.refusal == "rhs":
        b.view("<u4")[repeat:2 * repeat, 0, 0] = args.refusal_bits
    (out / "payload.bin").write_bytes(a.tobytes() + b.tobytes())
    (out / "payload.S").write_text('.section .rodata\n.balign 64\n.global payload\npayload:\n.incbin "payload.bin"\n')
    for name, values in (("a", a), ("b", b)):
        np.save(out / (name + ".npy"), values)
    (out / "runtime_group_prepare.h").write_text(emit_runtime_group_preparation())
    support = Path(__file__).resolve().parents[1] / "mx_gemmini_support"
    for name in ("runtime_quant.c", "runtime_quant.h", "runtime_quant_lut.h"):
        (out / name).write_bytes((support / name).read_bytes())
    (out / "issuer.c").write_text(generated.issuer.source)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    startup, riscv = args.startup_dir.resolve(strict=True), args.riscv_root.resolve(strict=True)
    cc = riscv / "bin/riscv64-unknown-elf-gcc"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DBAREMETAL=1", "-mcmodel=medany", "-O2",
             "-ffreestanding", "-fno-fast-math", "-fno-common", "-fno-builtin", "-fno-tree-loop-distribute-patterns",
             "-DMX_QUANT_GROUP_WORDS", "-DMX_QUANT_INLINE_GROUP", "-march=rv64gc", "-mabi=lp64d",
             "-I", str(out), "-I", str(startup), "-I", str(startup.parent.parent), "-I", str(startup.parent.parent / "env")]
    commands, results, all_words = [], {}, []

    def invoke(command, label):
        command = list(map(str, command))
        commands.append(command)
        result = subprocess.run(command, cwd=out, capture_output=True, check=False)
        (out / (label + ".log")).write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(label + " failed")

    route = DeviceRouting(device="mx_gemmini", package_dir=Path(__file__).resolve().parents[1],
                          operand_dtype="f32", accum_dtype="bf16", linked_elf_admission=service("mx_gemmini"))
    selection = freeze_device_elf_admission(route)
    for role, selected_repeat in (("baseline", 1), ("retained", repeat)):
        directory = out / role
        directory.mkdir()
        kernel = emit_dense_f32_batch_kernel(generated.issuer, (batch, m, n, k), symbol="produce",
                                            issuer_symbol="native_issue", rhs_batch_repeat=selected_repeat,
                                            virtual_zero_padding=args.virtual_zero_padding,
                                            compact_rhs=args.compact_rhs and role == "retained")
        (directory / "adapter.c").write_text(kernel.source)
        (directory / "driver.c").write_text(driver(batch, m, n, k, repeat, refusal=args.refusal != "none",
                                                   compact_rhs=args.compact_rhs and role == "retained"))
        objects = []
        paths = [directory / "adapter.c", directory / "driver.c", out / "issuer.c", out / "runtime_quant.c", out / "payload.S",
                 *sorted(startup.glob("*.c")), *sorted(startup.glob("*.S"))]
        for index, path in enumerate(paths):
            obj = directory / f"object{index}.o"
            invoke([cc, *flags, "-c", path, "-o", obj], role + f"_compile{index}")
            objects.append(obj)
        elf = directory / "section.elf"
        invoke([cc, "-nostdlib", "-nostartfiles", "-static", "-T", startup / "test.ld", *objects,
                "-lm", "-lgcc", "-o", elf], role + "_link")
        admission = admit_linked_device_elf(route, selection, elf=elf, linked_sha256=sha(elf), directory=directory)
        environment = os.environ | {"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
                                    "GEMMINI_PERF_SET": "mem.host_tracking=0", "GEMMINI_PERF_OUT": str(directory / "perf.txt")}
        command = [str(riscv / "bin/spike"), "--isa=rv64gc_zicntr", "--extlib=" + str(args.extlib.resolve()),
                   "--extension=gemmini", str(elf)]
        with (directory / "spike.log").open("wb") as log:
            process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=False)
        words, performance, statuses, traps = {}, [], [], []
        lines = (directory / "spike.log").read_text().splitlines()
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "WORDS":
                run, start = map(int, parts[1:3])
                values = words.setdefault(run, [])
                if start != len(values):
                    raise ValueError("batch output observation is incomplete or unordered")
                values.extend(int(word, 16) for word in parts[3:])
            elif parts[0] in {"PERF", "STATUS", "TRAP"}:
                values = {key: int(value) for key, value in (field.split("=", 1) for field in parts[1:])}
                {"PERF": performance, "STATUS": statuses, "TRAP": traps}[parts[0]].append(values)
        expected_runs = 1 if args.refusal != "none" else 3
        if (list(words) != list(range(expected_runs)) or any(len(row) != batch * m * n for row in words.values())
                or statuses != [{"run": i, "guard": 0} for i in range(expected_runs)]):
            raise ValueError("batch adapter changed output extent or exterior guards")
        if args.refusal != "none":
            if process.returncode != 77 or len(traps) != 1 or traps[0]["cause"] != 3 or performance or "DONE" in lines:
                raise ValueError("batch adapter changed its original explicit refusal")
            pc = traps[0]["pc"]
            disassembly = subprocess.check_output([str(riscv / "bin/riscv64-unknown-elf-objdump"), "-d",
                                                  f"--start-address={pc}", f"--stop-address={pc + 4}", str(elf)], text=True)
            rows = [line.split() for line in disassembly.splitlines() if line.split() and line.split()[0] == f"{pc:x}:"]
            if len(rows) != 1 or rows[0][-1] != "ebreak":
                raise ValueError("batch refusal was not the actual explicit breakpoint")
            (directory / "refusal_disassembly.log").write_text(disassembly)
        else:
            if process.returncode or traps or len(performance) != 3 or lines.count("DONE") != 1:
                raise ValueError("batch adapter did not complete all independent invocations")
            for run, observed in words.items():
                changed = b.copy()
                if run == 1:
                    changed[:repeat, 0, 0] *= -1
                elif run == 2:
                    changed[repeat:2 * repeat, 0, 0] *= -1
                golden = (a.astype(np.int32) @ changed.astype(np.int32)).astype("<f4").view("<u4")
                actual = np.asarray(observed, dtype="<u4").reshape(batch, m, n)
                if not np.array_equal(actual, golden):
                    raise ValueError("native batch output differs from complete independent integer product")
                np.save(directory / f"golden_{run}.npy", golden)
        raw = np.asarray(list(words.values()), dtype="<u4").tobytes()
        (directory / "observed_words.bin").write_bytes(raw)
        all_words.append(raw)
        results[role] = {"exit_code": process.returncode, "words_sha256": hashlib.sha256(raw).hexdigest(),
                         "performance": performance, "status": statuses, "traps": traps,
                         "admission": admission, "command": command, "workspace_bytes": kernel.workspace_bytes,
                         "environment": {key: environment[key] for key in ("GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET", "GEMMINI_PERF_OUT")}}
    if all_words[0] != all_words[1]:
        raise ValueError("batch preparation reuse changes full words or partial outputs before refusal")
    report = {"schema": "mx_gemmini.batch_rhs_reuse_section.v1", "scope": "complete_batched_adapter_sections_not_whole_model",
              "shape_bmnk": [batch, m, n, k], "repeat": repeat, "precision": args.precision, "refusal": args.refusal,
              "results": results, "commands": commands, "profile_sha256": profile_sha256(profile),
              "extension_sha256": sha(args.extlib), "compiler_sha256": sha(cc), "spike_sha256": sha(riscv / "bin/spike"),
              **({"compact_rhs": True} if args.compact_rhs else {}),
              "files_sha256": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}}
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"shape": report["shape_bmnk"], "repeat": repeat, "refusal": args.refusal,
                      "performance": {key: value["performance"] for key, value in results.items()}}), flush=True)


if __name__ == "__main__":
    main()
