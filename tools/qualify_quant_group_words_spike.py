"""Measure exact-byte host MX packing variants inside the original call boundary."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
from pathlib import Path

from mx_gemmini_support.runtime_group_preparation import emit_runtime_group_preparation
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_vpu_row_softmax_spike import _run


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def driver(rows, cols, precision, kind, live_rows, expected_status):
    count = (live_rows if kind == "zero_rows" else rows) * cols
    codes, scales = rows * cols * precision // 8, rows * cols // 32
    function = "mx_pack_activation_zero_rows_f32" if kind == "zero_rows" else f"mx_pack_{kind}_f32"
    arguments = f"input,{live_rows},{rows},{cols},{precision}" if kind == "zero_rows" else f"input,{rows},{cols},{precision}"
    return f'''#include <stdint.h>
      #include <stdio.h>
      #include <string.h>
      #include "runtime_quant.h"
      extern const unsigned char payload[];
      static uint8_t codes[{codes + 128}] __attribute__((aligned(64)));
      static uint8_t scales[{scales + 128}] __attribute__((aligned(64)));
      static uint64_t cycle(void){{uint64_t n;asm volatile("rdcycle %0":"=r"(n)::"memory");return n;}}
      int main(void) {{
        const float *input=(const float *)payload;
        const uint8_t *expected=payload+{count * 4};
        for(unsigned run=0;run<3;++run) {{
          memset(codes,0xa5,sizeof(codes));memset(scales,0xa5,sizeof(scales));
          uint64_t begin=cycle();int status={function}({arguments},codes+64,scales+64);
          uint64_t elapsed=cycle()-begin;
          unsigned mismatch=0,guard=0;uint32_t hash=2166136261u;
          for(unsigned i=0;i<{codes};++i){{mismatch+=codes[i+64]!=expected[i];hash=(hash^codes[i+64])*16777619u;}}
          for(unsigned i=0;i<{scales};++i){{mismatch+=scales[i+64]!=expected[{codes}+i];hash=(hash^scales[i+64])*16777619u;}}
          for(unsigned i=0;i<64;++i)guard+=codes[i]!=0xa5 || codes[{codes}+64+i]!=0xa5 ||
            scales[i]!=0xa5 || scales[{scales}+64+i]!=0xa5;
          printf("PACK run=%u status=%d cycles=%lu mismatch=%u guard=%u hash=%u\\n",run,status,elapsed,mismatch,guard,hash);
          if(status!={expected_status} || mismatch || guard)return 3;
        }}
        printf("DONE\\n");return 0;
      }}\n'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "riscv-root", "extlib", "startup-dir", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--precision", type=int, choices=(4, 8), required=True)
    parser.add_argument("--kind", choices=("activation", "weight", "zero_rows"), required=True)
    parser.add_argument("--live-rows", type=int, default=0)
    parser.add_argument("--refuse-index", type=int, default=-1)
    parser.add_argument("--refuse-bits", type=lambda s: int(s, 0), default=0x7f800000)
    parser.add_argument("--structural-elf-admission", action="store_true")
    args = parser.parse_args()
    tile = 16 if args.precision == 8 else 32
    if (args.out_dir.exists() or not args.structural_elf_admission or args.rows < tile or args.cols < tile
            or args.rows % tile or args.cols % tile or args.rows * args.cols > 16777216
            or (args.rows if args.kind == "weight" else args.cols) % 32
            or (args.kind == "zero_rows" and not 1 <= args.live_rows <= args.rows)
            or not 0 <= args.refuse_bits < 1 << 32):
        parser.error("fresh output, explicit admission, valid MX extents and a bounded source are required")
    import numpy as np
    from merlin.llvmlower.device_build import (
        DeviceRouting,
        admit_linked_device_elf,
        freeze_device_elf_admission,
    )
    from merlin.model_check.admission import service

    root = Path(__file__).resolve().parents[1]
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    profile = load_profile(args.profile)
    if profile["resources"]["scale_block_elements"] != 32 or profile["geometry"]["mesh_columns"] != 16:
        raise ValueError("quantizer requires the selected MX block/mesh contract")
    source = out / "sources"
    source.mkdir()
    for name in ("runtime_quant.c", "runtime_quant.h", "runtime_quant_lut.h"):
        (source / name).write_bytes((root / "mx_gemmini_support" / name).read_bytes())
    (source / "runtime_group_prepare.h").write_text(emit_runtime_group_preparation(profile["resources"]["scale_block_elements"]))
    (source / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    reference = out / "reference.so"
    _run(["cc", "-shared", "-fPIC", "-O2", "-fno-fast-math", "-DMX_QUANT_REFERENCE",
          str(source / "runtime_quant.c"), "-o", str(reference)], out, "compile_reference.log")
    library = ctypes.CDLL(str(reference))
    function = getattr(library, "mx_pack_activation_zero_rows_f32" if args.kind == "zero_rows" else
                       f"mx_pack_{args.kind}_f32")
    function.argtypes = [ctypes.POINTER(ctypes.c_float), *([ctypes.c_uint32] * (4 if args.kind == "zero_rows" else 3)),
                        ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
    function.restype = ctypes.c_int
    shape = (args.live_rows if args.kind == "zero_rows" else args.rows, args.cols)
    rng = np.random.default_rng(2718)
    values = np.ldexp(rng.uniform(-1.99, 1.99, shape), rng.integers(-25, 26, shape)).astype("<f4")
    bits = np.array([0, 0x80000000, 1, 0x80000001, 0x00010000, 0x80010000,
                     0x3f807fff, 0x3f808000, 0x3f818000], dtype="<u4")
    values.reshape(-1)[:bits.size] = bits.view("<f4")
    if args.refuse_index >= 0:
        if args.refuse_index >= values.size:
            raise ValueError("refusal index exceeds live input")
        values.view("<u4").reshape(-1)[args.refuse_index] = args.refuse_bits
    code_count, scale_count = args.rows * args.cols * args.precision // 8, args.rows * args.cols // 32
    codes = (ctypes.c_uint8 * code_count)(*([0xa5] * code_count))
    scales = (ctypes.c_uint8 * scale_count)(*([0xa5] * scale_count))
    prefix = [args.live_rows, args.rows, args.cols] if args.kind == "zero_rows" else [args.rows, args.cols]
    status = function(values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), *prefix, args.precision, codes, scales)
    np.save(out / "input.npy", values)
    (out / "expected_codes.bin").write_bytes(bytes(codes))
    (out / "expected_scales.bin").write_bytes(bytes(scales))
    (out / "payload.bin").write_bytes(values.tobytes() + bytes(codes) + bytes(scales))
    (out / "payload.S").write_text('.section .rodata\n.balign 64\n.global payload\npayload:\n.incbin "payload.bin"\n')
    (out / "driver.c").write_text(driver(args.rows, args.cols, args.precision, args.kind, args.live_rows, status))
    cc = args.riscv_root.resolve() / "bin/riscv64-unknown-elf-gcc"
    startup = args.startup_dir.resolve(strict=True)
    flags = [str(cc), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DBAREMETAL=1", "-mcmodel=medany", "-O2",
             "-ffreestanding", "-fno-fast-math", "-fno-common", "-fno-builtin-printf",
             "-fno-tree-loop-distribute-patterns", "-march=rv64gc_zicntr", "-I", str(source), "-I", str(startup),
             "-I", str(startup.parent.parent), "-I", str(startup.parent.parent / "env")]
    common = []
    for i, path in enumerate((out / "driver.c", out / "payload.S", *sorted(startup.glob("*.c")), *sorted(startup.glob("*.S")))):
        obj = out / f"common{i}.o"
        _run([*flags, "-c", str(path), "-o", str(obj)], out, f"compile_common{i}.log")
        common.append(obj)
    route = DeviceRouting(device="mx_gemmini", package_dir=root, operand_dtype="f32", accum_dtype="bf16",
                          linked_elf_admission=service("mx_gemmini"))
    selection = freeze_device_elf_admission(route)
    results = {}
    for name, defines in (("original", []), ("words", ["-DMX_QUANT_GROUP_WORDS"]),
                          ("original_inline", ["-DMX_QUANT_INLINE_GROUP"]),
                          ("words_inline", ["-DMX_QUANT_GROUP_WORDS", "-DMX_QUANT_INLINE_GROUP"])):
        directory = out / name
        directory.mkdir()
        obj, elf = directory / "quant.o", directory / "section.elf"
        _run([*flags, *defines, "-c", str(source / "runtime_quant.c"), "-o", str(obj)], directory, "compile.log")
        _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T", str(startup / "test.ld"),
              str(obj), *map(str, common), "-lm", "-lgcc", "-o", str(elf)], directory, "link.log")
        admission = admit_linked_device_elf(route, selection, elf=elf, linked_sha256=sha(elf), directory=directory)
        command = [str(args.riscv_root.resolve() / "bin/spike"), "--isa=rv64gc_zicntr",
                   "--extlib=" + str(args.extlib.resolve()), "--extension=gemmini", str(elf)]
        environment = os.environ | {"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
                                    "GEMMINI_PERF_SET": "mem.host_tracking=0", "GEMMINI_PERF_OUT": str(directory / "perf.txt")}
        with (directory / "spike.log").open("wb") as log:
            run = subprocess.run(command, cwd=out, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=1800, check=False)
        lines = (directory / "spike.log").read_text().splitlines()
        rows = [dict(part.split("=", 1) for part in line.split()[1:]) for line in lines if line.startswith("PACK ")]
        if (run.returncode or lines.count("DONE") != 1 or len(rows) != 3
                or [int(row["run"]) for row in rows] != [0, 1, 2]
                or any(int(row["status"]) != status or int(row["mismatch"]) or int(row["guard"]) for row in rows)):
            raise ValueError("quantizer execution or byte qualification failed: " + name)
        results[name] = {"exit_code": run.returncode, "checks": rows, "elf_admission": admission,
                         "defines": defines, "command": command,
                         "environment": {k: environment[k] for k in ("GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET", "GEMMINI_PERF_OUT")},
                         "files_sha256": {p.name: sha(p) for p in (obj, elf, directory / "spike.log")}}
        print(name, rows, flush=True)
    if len({row["hash"] for result in results.values() for row in result["checks"]}) != 1:
        raise ValueError("quantizer output hashes differ")
    report = {"schema": "mx_gemmini.group_word_packing.v1", "scope": "host_packing_section_only",
              "rows": args.rows, "cols": args.cols, "live_rows": args.live_rows, "kind": args.kind,
              "precision": args.precision, "reference_status": status, "seed": 2718,
              "refuse_index": args.refuse_index, "refuse_bits": args.refuse_bits,
              "results": results, "profile_sha256": profile_sha256(profile),
              "extension_sha256": sha(args.extlib), "gcc_sha256": sha(cc), "spike_sha256": sha(args.riscv_root / "bin/spike"),
              "files_sha256": {p.name: sha(p) for p in (out / "input.npy", out / "expected_codes.bin", out / "expected_scales.bin",
                                                        out / "payload.bin", out / "payload.S", out / "driver.c")},
              "source_sha256": {p.name: sha(p) for p in source.iterdir()}}
    (out / "receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
