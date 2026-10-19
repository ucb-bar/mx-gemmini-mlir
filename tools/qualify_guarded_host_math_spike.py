"""Qualify guarded binary32 library calls with actual RV64 link/runtime observations.

Return bits, errno and flags are compared over all BF16 words plus randomized
F32 words, in each supported rounding mode. Performance loops have matching
output stores/hashes; complete comparisons run outside the timers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from merlin.llvmlower.device_build import (
    DeviceRouting,
    admit_linked_device_elf,
    freeze_device_elf_admission,
)
from merlin.model_check.admission import service
from merlin.runtime.host_math import GuardedF32Call, host_math_recipe

from mx_gemmini_support.target_profile import load_profile, profile_sha256

DRIVER = r'''#include <stdint.h>
#include <stdio.h>
#include <errno.h>
static int native_errno;int *__errno(void){return &native_errno;}
extern float expf(float);extern float __real_expf(float);
extern float __ieee754_expf(float);
static uint32_t bits(float x){union{float f;uint32_t u;}v={.f=x};return v.u;}
static float value(uint32_t u){union{float f;uint32_t u;}v={.u=u};return v.f;}
static inline float selected(float x){return expf(x); }
static uint64_t cycle(void){uint64_t r;asm volatile("rdcycle %0":"=r"(r)::"memory");return r;}
static unsigned flags(void){unsigned r;asm volatile("frflags %0":"=r"(r)::"memory");return r;}
static void clear(void){asm volatile("fsflags zero":::"memory");}
static void mode(unsigned m){asm volatile("fsrm %0"::"r"(m):"memory");}
static float input[32768];static uint32_t reference[32768],observed[32768];
__attribute__((noinline)) static uint32_t run_original(void){uint32_t h=2166136261u;for(unsigned i=0;i<32768;++i){float r=__real_expf(input[i]);reference[i]=bits(r);h=(h^bits(r))*16777619u;}return h;}
__attribute__((noinline)) static uint32_t run_selected(void){uint32_t h=2166136261u;for(unsigned i=0;i<32768;++i){float r=selected(input[i]);observed[i]=bits(r);h=(h^bits(r))*16777619u;}return h;}
int main(void){
  unsigned mismatch=0;uint32_t random=0xaca1282au;
  for(unsigned m=0;m<5;++m){
    mode(m);
    for(unsigned i=0;i<131072;++i){
      random^=random<<13;random^=random>>17;random^=random<<5;
      uint32_t u=i<65536?i<<16:random;float x=value(u);
      errno=91;clear();float a=__real_expf(x);unsigned af=flags();int ae=errno;
      errno=91;clear();float b=selected(x);unsigned bf=flags();int be=errno;
      if(bits(a)!=bits(b)||af!=bf||ae!=be){if(mismatch<4)printf("DIFF mode=%u input=%x a=%x b=%x af=%u bf=%u ae=%d be=%d\n",m,u,bits(a),bits(b),af,bf,ae,be);++mismatch;}
    }
  }
  mode(0);printf("QUALIFY cases=655360 mismatch=%u\n",mismatch);if(mismatch)return 5;
  for(unsigned kind=0;kind<4;++kind){
    for(unsigned i=0;i<32768;++i)input[i]=kind==3?-128.0f:((kind==1 && (i&1))?value(0xff800000u):(kind==2?value(0xff800000u):-(float)(i%4096)/256.0f));
    for(unsigned rep=0;rep<3;++rep){unsigned bad=0;uint64_t t=cycle();uint32_t h=run_original();uint64_t before=cycle()-t;t=cycle();uint32_t g=run_selected();uint64_t after=cycle()-t;for(unsigned i=0;i<32768;++i)bad+=reference[i]!=observed[i];
      printf("EXP kind=%u run=%u original=%lu selected=%lu mismatch=%u hash=%u\n",kind,rep,before,after,bad,h);if(bad||h!=g)return 6;
    }
  }
  printf("DONE\n");return 0;
}'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "riscv-root", "extlib", "startup-dir", "out-dir", "guard"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--structural-elf-admission", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists() or not args.structural_elf_admission:
        parser.error("fresh output and explicit structural admission are required")
    guard = GuardedF32Call.from_dict(json.loads(args.guard.read_text()))
    recipe = host_math_recipe("guarded_f32", guarded_math=guard)
    cc = args.riscv_root.resolve() / "bin/riscv64-unknown-elf-gcc"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DBAREMETAL=1", "-mcmodel=medany", "-O2",
             "-ffreestanding", "-fno-fast-math", "-fno-common", "-fno-builtin", "-ffp-contract=off",
             "-march=rv64gc", "-mabi=lp64d"]
    library = Path(subprocess.check_output([str(cc), *flags, "-print-file-name=libm.a"], text=True).strip()).resolve(strict=True)
    if sha(library) != guard.library_sha256:
        raise ValueError("selected library differs from reviewed guarded math contract")
    startup = args.startup_dir.resolve(strict=True)
    flags += ["-I", str(startup), "-I", str(startup.parent.parent), "-I", str(startup.parent.parent / "env")]
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    source = DRIVER.replace("__ieee754_expf", "BODY_ENDPOINT").replace("__real_expf", "BASELINE_ENDPOINT")
    source = source.replace("expf(", "PUBLIC_ENDPOINT(")
    source = source.replace("BODY_ENDPOINT", guard.body_symbol).replace("BASELINE_ENDPOINT", "__real_" + guard.source_symbol)
    source = source.replace("PUBLIC_ENDPOINT", guard.source_symbol)
    (out / "driver.c").write_text(source)
    (out / "guard.c").write_text(recipe.source)
    (out / "guard.json").write_text(json.dumps(guard.to_dict(), indent=2) + "\n")
    (out / "libm.a").write_bytes(library.read_bytes())
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (out / "host_math.py").write_bytes(Path(__import__("merlin.runtime.host_math", fromlist=["__file__"]).__file__).read_bytes())
    commands, objects = [], []

    def run(command, label):
        command = list(map(str, command))
        commands.append(command)
        result = subprocess.run(command, cwd=out, capture_output=True, check=False)
        (out / (label + ".log")).write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(label + " failed")

    for i, path in enumerate((out / "driver.c", out / "guard.c", *sorted(startup.glob("*.c")), *sorted(startup.glob("*.S")))):
        obj = out / f"part{i}.o"
        run([cc, *flags, "-c", path, "-o", obj], f"compile{i}")
        objects.append(obj)
    elf = out / "section.elf"
    run([cc, "-nostdlib", "-nostartfiles", "-static", "-T", startup / "test.ld", *objects,
         *recipe.link_flags, "-Wl,--trace-symbol=" + guard.source_symbol, "-Wl,--trace-symbol=" + guard.body_symbol,
         out / "libm.a", "-lgcc", "-o", elf], "link")
    route = DeviceRouting(device="mx_gemmini", package_dir=Path(__file__).resolve().parents[1], operand_dtype="f32",
                          accum_dtype="bf16", linked_elf_admission=service("mx_gemmini"))
    admission = admit_linked_device_elf(route, freeze_device_elf_admission(route), elf=elf,
                                       linked_sha256=sha(elf), directory=out)
    environment = os.environ | {"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
                                "GEMMINI_PERF_SET": "mem.host_tracking=0", "GEMMINI_PERF_OUT": str(out / "perf.txt")}
    command = [str(args.riscv_root.resolve() / "bin/spike"), "--isa=rv64gc_zicntr",
               "--extlib=" + str(args.extlib.resolve()), "--extension=gemmini", str(elf)]
    with (out / "spike.log").open("wb") as log:
        result = subprocess.run(command, cwd=out, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    lines = (out / "spike.log").read_text().splitlines()
    qualify = [line for line in lines if line.startswith("QUALIFY ")]
    rows = [dict(part.split("=", 1) for part in line.split()[1:]) for line in lines if line.startswith("EXP ")]
    if (result.returncode or qualify != ["QUALIFY cases=655360 mismatch=0"] or lines.count("DONE") != 1
            or [(int(row["kind"]), int(row["run"])) for row in rows] != [(kind, rep) for kind in range(4) for rep in range(3)]
            or any(int(row["mismatch"]) for row in rows)):
        raise ValueError("guarded math execution or complete byte/effect comparison failed")
    report = {"schema": "mx_gemmini.guarded_math_section.v1", "scope": "host_call_sections_only",
              "guard": guard.to_dict(), "profile_sha256": profile_sha256(load_profile(args.profile)),
              "commands": commands, "spike_command": command, "exit_code": result.returncode,
              "environment": {key: environment[key] for key in ("GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET", "GEMMINI_PERF_OUT")},
              "admission": admission, "extension_sha256": sha(args.extlib), "compiler_sha256": sha(cc),
              "checks": rows, "files_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / "receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("PASS: 655360 return/errno/flag comparisons; three original/selected loop pairs per distribution")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
