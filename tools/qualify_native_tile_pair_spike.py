"""Compare checked FP8 native partitions on the same encoded operands.

The timer covers dispatch, configuration, transfers and compute, excluding host
packing and output validation. This is an issuer section, not a model result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from mx_gemmini_support.scheduled_native_issuer import emit_scheduled_native_issuer
from mx_gemmini_support.spike_timing import reject_deadlocked_timing
from mx_gemmini_support.target_profile import load_profile, profile_sha256


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def driver(count, offsets, independent, candidate_first=False):
    return f'''#include <stdint.h>
#include <stdio.h>
extern const unsigned char payload[];
void legacy_issue(const void *,const void *,const void *,const void *,const void *,const void *);
void candidate_issue(const void *,const void *,const void *,const void *,const void *,const void *);
typedef void (*issuer)(const void *,const void *,const void *,const void *,const void *,const void *);
static uint16_t before[{count + 64}] __attribute__((aligned(64)));
static uint16_t after[{count + 64}] __attribute__((aligned(64)));
static uint8_t scratch[2048] __attribute__((aligned(64)));
static uint64_t cycle(void){{uint64_t n;asm volatile("rdcycle %0":"=r"(n)::"memory");return n;}}
__attribute__((noinline)) static uint64_t measure(issuer fn,uint16_t *out){{
  uint64_t begin=cycle();
  fn(payload+{offsets['a']},payload+{offsets['asc']},payload+{offsets['b']},payload+{offsets['bsc']},out,scratch);
  return cycle()-begin;
}}
int main(void){{
  const uint16_t *gold=(const uint16_t *)(payload+{offsets['golden']});
  for(unsigned run=0;run<3;++run){{
    for(unsigned i=0;i<{count + 64};++i)before[i]=after[i]=0xa55a;
    uint64_t old,new;
    if((run+{int(candidate_first)})&1){{new=measure(candidate_issue,after+32);old=measure(legacy_issue,before+32);}}
    else{{old=measure(legacy_issue,before+32);new=measure(candidate_issue,after+32);}}
    unsigned mismatch=0,oracle=0,guard=0;uint32_t hash=2166136261u;
    for(unsigned i=0;i<{count};++i){{
      mismatch+=before[i+32]!=after[i+32];
      if({int(independent)})oracle+=before[i+32]!=gold[i] || after[i+32]!=gold[i];
      hash=(hash^before[i+32])*16777619u;
    }}
    for(unsigned i=0;i<32;++i)guard+=before[i]!=0xa55a || after[i]!=0xa55a ||
      before[{count}+32+i]!=0xa55a || after[{count}+32+i]!=0xa55a;
    printf("TILE run=%u baseline=%lu candidate=%lu mismatch=%u oracle=%u guard=%u hash=%u\\n",run,old,new,mismatch,oracle,guard,hash);
    if(mismatch || oracle || guard)return 3;
  }}
  printf("DONE\\n");return 0;
}}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "riscv-root", "extlib", "startup-dir", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for axis in ("m", "n", "k"):
        parser.add_argument("--" + axis, type=int, required=True)
    parser.add_argument("--tile", required=True, help="explicit candidate M,N,K partition")
    parser.add_argument("--pattern", choices=("integer", "fp8"), default="integer")
    parser.add_argument("--candidate-first", action="store_true", help="reverse the initial call order for cold controls")
    parser.add_argument("--structural-elf-admission", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists() or not args.structural_elf_admission:
        parser.error("fresh output and explicit structural admission are required")
    shape, tiles = (args.m, args.n, args.k), tuple(int(value) for value in args.tile.split(","))
    if any(value <= 0 for value in shape) or max(args.m * args.k, args.k * args.n, args.m * args.n) > 16777216:
        parser.error("bounded positive source extents are required")
    profile = load_profile(args.profile)
    schedules = {"baseline": emit_scheduled_native_issuer(profile, shape, precision="FP8", symbol="legacy_issue"),
                 "candidate": emit_scheduled_native_issuer(profile, shape, precision="FP8", symbol="candidate_issue", tile_mnk=tiles)}
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    rng = np.random.default_rng(581203)
    if args.pattern == "integer":
        a = rng.integers(0, 2, (args.m, args.k), dtype=np.int8) * 2 - 1
        b = rng.integers(0, 2, (args.k, args.n), dtype=np.int8) * 2 - 1
        a[0] = 1
        b[:, 0] = 1
        ac = np.where(a > 0, 0x38, 0xb8).astype(np.uint8)
        bc = np.where(b > 0, 0x38, 0xb8).astype(np.uint8)
        asc, bsc = np.full((args.k // 32, args.m), 127, np.uint8), np.full((args.k // 32, args.n), 127, np.uint8)
        words = (a.astype(np.int32) @ b.astype(np.int32)).astype("<f4").view("<u4")
        golden = ((words + np.uint32(0x7fff) + ((words >> 16) & 1)) >> 16).astype("<u2")
    else:
        ac = rng.integers(0x28, 0x51, (args.m, args.k), dtype=np.uint8)
        bc = rng.integers(0x28, 0x51, (args.k, args.n), dtype=np.uint8)
        ac |= rng.integers(0, 2, ac.shape, dtype=np.uint8) << 7
        bc |= rng.integers(0, 2, bc.shape, dtype=np.uint8) << 7
        asc = rng.integers(125, 130, (args.k // 32, args.m), dtype=np.uint8)
        bsc = rng.integers(125, 130, (args.k // 32, args.n), dtype=np.uint8)
        golden = np.zeros((args.m, args.n), dtype="<u2")
    payload, offsets = bytearray(), {}
    for name, values in (("a", ac), ("asc", asc), ("b", bc), ("bsc", bsc), ("golden", golden)):
        payload.extend(bytes((-len(payload)) % 64))
        offsets[name] = len(payload)
        payload.extend(values.tobytes())
        np.save(out / (name + ".npy"), values)
    (out / "payload.bin").write_bytes(payload)
    (out / "payload.S").write_text('.section .rodata\n.balign 64\n.global payload\npayload:\n.incbin "payload.bin"\n')
    (out / "driver.c").write_text(driver(args.m * args.n, offsets, args.pattern == "integer", args.candidate_first))
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    from mx_gemmini_support import scheduled_native_issuer
    (out / "scheduled_native_issuer.py").write_bytes(Path(scheduled_native_issuer.__file__).read_bytes())
    for name, schedule in schedules.items():
        (out / (name + ".c")).write_text(schedule.issuer.source)
        (out / (name + ".mk")).write_text(schedule.text)
    startup = args.startup_dir.resolve(strict=True)
    cc = args.riscv_root.resolve() / "bin/riscv64-unknown-elf-gcc"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DBAREMETAL=1", "-mcmodel=medany", "-O2",
             "-ffreestanding", "-fno-fast-math", "-fno-common", "-fno-builtin", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-mabi=lp64d", "-I", str(startup), "-I", str(startup.parent.parent),
             "-I", str(startup.parent.parent / "env")]
    commands, objects = [], []

    def run(command, label):
        command = list(map(str, command))
        commands.append(command)
        result = subprocess.run(command, cwd=out, capture_output=True, check=False)
        (out / (label + ".log")).write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(label + " failed")

    for i, path in enumerate((out / "baseline.c", out / "candidate.c", out / "driver.c", out / "payload.S",
                              *sorted(startup.glob("*.c")), *sorted(startup.glob("*.S")))):
        obj = out / f"part{i}.o"
        run([cc, *flags, "-c", path, "-o", obj], f"compile{i}")
        objects.append(obj)
    elf = out / "section.elf"
    run([cc, "-nostdlib", "-nostartfiles", "-static", "-T", startup / "test.ld", *objects, "-lm", "-lgcc", "-o", elf], "link")
    from merlin.llvmlower.device_build import (
        DeviceRouting,
        admit_linked_device_elf,
        freeze_device_elf_admission,
    )
    from merlin.model_check.admission import service
    route = DeviceRouting(device="mx_gemmini", package_dir=Path(__file__).resolve().parents[1], operand_dtype="f32",
                          accum_dtype="bf16", linked_elf_admission=service("mx_gemmini"))
    admission = admit_linked_device_elf(route, freeze_device_elf_admission(route), elf=elf, linked_sha256=sha(elf), directory=out)
    command = [str(args.riscv_root.resolve() / "bin/spike"), "--isa=rv64gc_zicntr",
               "--extlib=" + str(args.extlib.resolve()), "--extension=gemmini", str(elf)]
    environment = os.environ | {"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
                                "GEMMINI_PERF_SET": "mem.host_tracking=0", "GEMMINI_PERF_OUT": str(out / "perf.txt")}
    invocation = {"commands": commands, "command": command, "admission": admission,
                  "environment": {key: environment[key] for key in ("GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET", "GEMMINI_PERF_OUT")}}
    (out / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
    with (out / "spike.log").open("wb") as log:
        result = subprocess.run(command, cwd=out, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=3600)
    lines = (out / "spike.log").read_text().splitlines()
    reject_deadlocked_timing((out / "spike.log").read_text())
    rows = [dict(part.split("=", 1) for part in line.split()[1:]) for line in lines if line.startswith("TILE ")]
    report = {"schema": "mx_gemmini.native_tile_pair.v1", "scope": "encoded_issuer_section_only", "shape_mnk": shape,
              "pattern": args.pattern, "seed": 581203, "payload_offsets": offsets, "exit_code": result.returncode,
              "candidate_first": args.candidate_first,
              "profile_sha256": profile_sha256(profile), "extension_sha256": sha(args.extlib), "compiler_sha256": sha(cc),
              "checks": rows, "schedules": {name: {"tile_mnk": s.chunks, "schedule_digest": s.digest,
                                                   "instruction_set_digest": s.instruction_set_digest} for name, s in schedules.items()},
              "files_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / "receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if (result.returncode or lines.count("DONE") != 1 or [int(row["run"]) for row in rows] != [0, 1, 2]
            or any(int(row[key]) for row in rows for key in ("mismatch", "oracle", "guard"))):
        raise ValueError("native tile pair failed complete output/canary qualification")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
