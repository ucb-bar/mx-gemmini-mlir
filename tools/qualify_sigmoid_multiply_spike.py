"""Measure a resident sigmoid/multiply chain against the original VPU calls.

The oracle is the same pinned device's original materialized program. Complete
word equality proves transport preservation for these inputs, not FP32 accuracy
or whole-model quality. All calls keep their existing BF16 arithmetic policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from mx_gemmini_support.connected_sigmoid_multiply import (
    emit_connected_sigmoid_multiply,
    emit_sigmoid_with_connected_multiplies,
)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_vpu_row_softmax_spike import _run
from tools.report_stage_profile import parse_gemmini_perf


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def driver(elements, golden):
    return f'''#include <stdint.h>
#include <stdio.h>
extern const unsigned char payload[];
void chain(const float *,const float *,float *);
static float output[{elements}] __attribute__((aligned(64)));
static uint64_t cycle(void) {{uint64_t n; asm volatile("rdcycle %0":"=r"(n)::"memory");return n;}}
static uint32_t bits(float x) {{union{{float f;uint32_t u;}} w;w.f=x;return w.u;}}
static uint32_t input_hash(const float *x) {{uint32_t h=2166136261u;
  for(unsigned i=0;i<{elements * 2}u;++i)h=(h^bits(x[i]))*16777619u;return h;}}
int main(void) {{
  const float *x=(const float *)payload,*u=x+{elements};
  const uint32_t *reference=(const uint32_t *)(u+{elements});
  uint32_t original=input_hash(x);
  for(unsigned run=0;run<3;++run) {{
    uint64_t begin=cycle();chain(x,u,output);uint64_t elapsed=cycle()-begin;
    unsigned mismatches=0;uint32_t h=2166136261u;
    for(unsigned i=0;i<{elements}u;++i) {{
      uint32_t word=bits(output[i]);h=(h^word)*16777619u;
      if({int(golden)})mismatches+=word!=reference[i];
    }}
    if(input_hash(x)!=original)return 4;
    printf("SECTION %u %lu %u %u\\n",run,elapsed,mismatches,h);
    if(mismatches)return 3;
    if(!run)for(unsigned base=0;base<{elements}u;base+=128u) {{
      unsigned count={elements}u-base;if(count>128u)count=128u;
      printf("WORDS %u %u",base,count);
      for(unsigned i=0;i<count;++i)printf(" %08x",bits(output[base+i]));
      printf("\\n");
    }}
  }}
  printf("DONE\\n");return 0;
}}\n'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "profile", "riscv-root", "extlib", "startup-dir", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--structural-elf-admission", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists() or not args.structural_elf_admission:
        parser.error("fresh output and explicit structural ELF admission required")
    import numpy as np
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.device_build import (
        DeviceRouting,
        admit_linked_device_elf,
        freeze_device_elf_admission,
    )
    from merlin.model_check.admission import service
    from merlin.xdsl_dialects.lowering.dispatch_program import (
        lower_model_to_dispatch_program,
    )

    capture = json.loads((args.capture / "receipt.json").read_text())
    for name, digest in capture["artifacts"].items():
        if sha(args.capture / name) != digest:
            raise ValueError("source capture changed: " + name)
    outlined, program = lower_model_to_dispatch_program(parse_mlir_file(args.capture / "source.mlir"))
    profile = load_profile(args.profile)
    with np.load(args.capture / "inputs.npz", allow_pickle=False) as values:
        x, up = values["in0"], values["in1"]
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    (out / "source.mlir").write_text((args.capture / "source.mlir").read_text())
    (out / "outlined.mlir").write_text(str(outlined.module))
    (out / "program.json").write_text(json.dumps(program.to_dict(), indent=2) + "\n")
    route = DeviceRouting(device="mx_gemmini", package_dir=Path(__file__).resolve().parents[1],
                          operand_dtype="bf16", accum_dtype="bf16", linked_elf_admission=service("mx_gemmini"))
    selection = freeze_device_elf_admission(route)
    cc = args.riscv_root.resolve() / "bin/riscv64-unknown-elf-gcc"
    bench = args.startup_dir.resolve(strict=True)
    flags = [str(cc), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DBAREMETAL=1",
             "-mcmodel=medany", "-O2", "-fno-common", "-fno-builtin-printf",
             "-fno-tree-loop-distribute-patterns", "-march=rv64gc_zicntr",
             "-I", str(bench), "-I", str(bench.parent.parent), "-I", str(bench.parent.parent / "env")]
    results, baseline_words = {}, None
    for variant in ("materialized", "multiply_pair", "connected"):
        directory = out / variant
        directory.mkdir()
        if variant == "multiply_pair":
            generated = emit_sigmoid_with_connected_multiplies(
                outlined.module, program.to_dict(), profile, symbol="chain",
                boundary_policy="bf16_rne_each_vpu_call")
        else:
            generated = emit_connected_sigmoid_multiply(
                outlined.module, program.to_dict(), profile, symbol="chain", connect=variant == "connected",
                boundary_policy="bf16_rne_each_vpu_call")
        if list(generated.shape) != list(x.shape) or up.shape != x.shape or x.dtype != up.dtype or x.dtype != np.float32:
            raise ValueError("runtime operands differ from typed source")
        elements = x.size
        (directory / "issue.c").write_text(generated.source)
        (directory / "driver.c").write_text(driver(elements, baseline_words is not None))
        payload = x.astype("<f4").tobytes() + up.astype("<f4").tobytes()
        if baseline_words is not None:
            payload += baseline_words.tobytes()
        (directory / "payload.bin").write_bytes(payload)
        (directory / "payload.S").write_text('.section .rodata\n.balign 64\n.global payload\npayload:\n.incbin "payload.bin"\n')
        objects = []
        for index, path in enumerate((directory / "issue.c", directory / "driver.c", directory / "payload.S",
                                      *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S")))):
            obj = directory / f"part{index}.o"
            _run([*flags, "-c", str(path), "-o", str(obj)], directory, f"compile{index}.log")
            objects.append(obj)
        elf = directory / "section.elf"
        _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T", str(bench / "test.ld"),
              *map(str, objects), "-lm", "-lgcc", "-o", str(elf)], directory, "link.log")
        admission = admit_linked_device_elf(route, selection, elf=elf, linked_sha256=sha(elf), directory=directory)
        env = os.environ | {"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
                            "GEMMINI_PERF_SET": "mem.host_tracking=0", "GEMMINI_PERF_OUT": str(directory / "perf.txt")}
        command = [str(args.riscv_root.resolve() / "bin/spike"), "--isa=rv64gc_zicntr",
                   "--extlib=" + str(args.extlib.resolve()), "--extension=gemmini", str(elf)]
        with (directory / "spike.log").open("w") as stream:
            run = subprocess.run(command, cwd=directory, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, timeout=1800, check=False)
        lines = (directory / "spike.log").read_text().splitlines()
        checks, words = [], []
        for line in lines:
            parts = line.split()
            if parts and parts[0] == "SECTION":
                checks.append(list(map(int, parts[1:])))
            elif parts and parts[0] == "WORDS":
                if int(parts[1]) != len(words) or int(parts[2]) != len(parts[3:]):
                    raise ValueError("complete section words have a gap or changed extent")
                words.extend(int(word, 16) for word in parts[3:])
        if (run.returncode or lines.count("DONE") != 1 or len(checks) != 3
                or [row[0] for row in checks] != [0, 1, 2] or any(row[2] for row in checks)
                or len({row[3] for row in checks}) != 1 or len(words) != elements):
            raise ValueError("section execution or complete output qualification failed: " + variant)
        values = np.array(words, dtype="<u4")
        if baseline_words is None:
            baseline_words = values
        elif not np.array_equal(values, baseline_words):
            raise ValueError("resident sigmoid/multiply changed an output word")
        (directory / "output.bin").write_bytes(values.tobytes())
        results[variant] = {"checks": checks, "all_words_sha256": sha(directory / "output.bin"),
            "output_elements": elements, "elf_sha256": sha(elf), "console_sha256": sha(directory / "spike.log"),
            "elf_admission": admission, "perf": parse_gemmini_perf((directory / "perf.txt").read_text()),
            "removed_transfer_bytes": generated.removed_transfer_bytes, "resident_bytes": generated.resident_bytes}
        print(variant, checks, flush=True)
    report = {"schema": "mx_gemmini.sigmoid_multiply_transport.v1", "scope": "section_not_whole_model_or_independent_fp32_quality",
              "capture": capture, "results": results, "all_words_equal": True,
              "profile_sha256": profile_sha256(profile), "extension_sha256": sha(args.extlib),
              "spike_sha256": sha(args.riscv_root / "bin/spike"), "gcc_sha256": sha(cc),
              "startup_sources_sha256": {str(p): sha(p) for p in (*bench.glob("*.c"), *bench.glob("*.S"), bench / "test.ld")}}
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
