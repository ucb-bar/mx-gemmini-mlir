"""Compare compact BF16 packing with packing its exact F32 widening on Spike."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from mx_gemmini_support.runtime_group_preparation import emit_runtime_group_preparation
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_vpu_row_softmax_spike import _run


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def driver(m, k, precision, status, candidate_first, producer_counts=()):
    count, codes, scales = m * k, m * k * precision // 8, m * k // 32
    producer_input_bytes = sum(producer_counts) * 4
    declarations, initialize, issue, checks = "", "", "", ""
    inputs = "const float *wide=(const float *)payload;\n  const uint16_t *compact=(const uint16_t *)(payload+" + str(count * 4) + ");"
    if producer_counts:
        declarations = f'''
void pack_producer_f32(const float *,const float *,const float *,float *);
void pack_producer_bf16(const float *,const float *,const float *,uint16_t *);
static float wide_storage[{count + 32}] __attribute__((aligned(64)));
static uint16_t compact_storage[{count + 64}] __attribute__((aligned(64)));
'''
        inputs = f'''float *wide=wide_storage+16; uint16_t *compact=compact_storage+32;
  const float *a=(const float *)payload,*b=a+{producer_counts[0]},*c=b+{producer_counts[1]};
  const uint16_t *golden=(const uint16_t *)(payload+{producer_input_bytes + count * 4});'''
        initialize = "memset(wide_storage,0xa5,sizeof(wide_storage));memset(compact_storage,0xa5,sizeof(compact_storage));"
        issue = "if(variant)pack_producer_bf16(a,b,c,compact);else pack_producer_f32(a,b,c,wide);"
        checks = f'''
    for(unsigned i=0;i<{count};++i) {{
      union {{float f;uint32_t u;}} word;word.f=wide[i];
      mismatch+=variant ? compact[i]!=golden[i] : word.u!=((uint32_t)golden[i]<<16);
    }}
    const uint8_t *storage=variant ? (const uint8_t *)compact_storage : (const uint8_t *)wide_storage;
    unsigned output_bytes=variant ? {count * 2}u : {count * 4}u;
    for(unsigned i=0;i<64;++i)guard+=storage[i]!=0xa5 || storage[64+output_bytes+i]!=0xa5;
'''
    return f'''#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "runtime_quant.h"
extern const unsigned char payload[];
{declarations}
static uint8_t codes[{codes + 128}] __attribute__((aligned(64)));
static uint8_t scales[{scales + 128}] __attribute__((aligned(64)));
static uint64_t cycle(void){{uint64_t n;asm volatile("rdcycle %0":"=r"(n)::"memory");return n;}}
int main(void) {{
  {inputs}
  const uint8_t *expected=payload+{producer_input_bytes + count * 6};
  for(unsigned run=0;run<3;++run) for(unsigned order=0;order<2;++order) {{
    unsigned variant=order^(run&1u)^{int(candidate_first)}u;
    memset(codes,0xa5,sizeof(codes));memset(scales,0xa5,sizeof(scales));
    {initialize}
    uint64_t begin=cycle();
    {issue}
    int status=variant ? mx_pack_activation_bf16(compact,{m},{k},{precision},codes+64,scales+64)
                       : mx_pack_activation_f32(wide,{m},{k},{precision},codes+64,scales+64);
    uint64_t elapsed=cycle()-begin;
    unsigned mismatch=0,guard=0;uint32_t hash=2166136261u;
    {checks}
    for(unsigned i=0;i<{codes};++i){{mismatch+=codes[i+64]!=expected[i];hash=(hash^codes[i+64])*16777619u;}}
    for(unsigned i=0;i<{scales};++i){{mismatch+=scales[i+64]!=expected[{codes}+i];hash=(hash^scales[i+64])*16777619u;}}
    for(unsigned i=0;i<64;++i)guard+=codes[i]!=0xa5 || codes[{codes}+64+i]!=0xa5 ||
      scales[i]!=0xa5 || scales[{scales}+64+i]!=0xa5;
    printf("PACK run=%u variant=%u status=%d cycles=%lu mismatch=%u guard=%u hash=%u\\n",
           run,variant,status,elapsed,mismatch,guard,hash);
    if(status!={status} || mismatch || guard)return 3;
  }}
  printf("DONE\\n");return 0;
}}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "riscv-root", "extlib", "startup-dir", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--precision", type=int, choices=(4, 8), required=True)
    parser.add_argument("--refuse-index", type=int, default=-1)
    parser.add_argument("--refuse-word", type=lambda s: int(s, 0), default=0x7F80)
    parser.add_argument("--candidate-first", action="store_true")
    parser.add_argument("--producer-capture", type=Path,
                        help="also time/check a captured connected multiply producer before packing")
    parser.add_argument("--structural-elf-admission", action="store_true")
    args = parser.parse_args()
    m, k, precision = args.rows, args.cols, args.precision
    tile = 16 if precision == 8 else 32
    if (args.out_dir.exists() or not args.structural_elf_admission or m < tile or k < tile
            or m % tile or k % 32 or m * k > 16777216
            or not -1 <= args.refuse_index < m * k or not 0 <= args.refuse_word <= 65535):
        parser.error("fresh output, explicit admission and bounded padded MX extents required")
    from merlin.llvmlower.device_build import (
        DeviceRouting,
        admit_linked_device_elf,
        freeze_device_elf_admission,
    )
    from merlin.model_check.admission import service

    root, out = Path(__file__).resolve().parents[1], args.out_dir.resolve()
    out.mkdir(parents=True)
    profile = load_profile(args.profile)
    if profile["resources"]["scale_block_elements"] != 32 or profile["geometry"]["mesh_columns"] != 16:
        raise ValueError("selected MX block/mesh contract differs")
    sources = out / "sources"
    sources.mkdir()
    for name in ("runtime_quant.c", "runtime_quant.h", "runtime_quant_lut.h"):
        (sources / name).write_bytes((root / "mx_gemmini_support" / name).read_bytes())
    (sources / "runtime_group_prepare.h").write_text(emit_runtime_group_preparation(bf16_input=True))
    (sources / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    commands = []

    def run(command, log):
        commands.append(command)
        _run(command, out, log)

    reference = out / "reference.so"
    run(["cc", "-shared", "-fPIC", "-O2", "-fno-fast-math", "-DMX_QUANT_REFERENCE",
         str(sources / "runtime_quant.c"), "-o", str(reference)], "compile_reference.log")
    loaded = ctypes.CDLL(str(reference))
    function = loaded.mx_pack_activation_f32
    function.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                         ctypes.c_void_p, ctypes.c_void_p]
    function.restype = ctypes.c_int
    rng = np.random.default_rng(2718)
    values = np.ldexp(rng.uniform(-1.99, 1.99, (m, k)), rng.integers(-25, 26, (m, k))).astype("<f4")
    bits = values.view("<u4")
    words = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
    words.reshape(-1)[:9] = [0, 0x8000, 1, 0x8001, 0x7F7F, 0xFF7F, 0x3F80, 0x3F81, 0x0080]
    if args.refuse_index >= 0:
        words.reshape(-1)[args.refuse_index] = args.refuse_word
    producer_counts, producer_paths, producer = (), [], None
    producer_input = b""
    if args.producer_capture is not None:
        from math import prod

        from merlin.frontends.linalg_mlir import parse_mlir_file

        from mx_gemmini_support.merlin_connected_multiply import (
            emit_chain_kernel,
            multiply_pairs,
        )
        from mx_gemmini_support.merlin_vpu_elementwise import selected_elementwise_sites

        capture = args.producer_capture.resolve(strict=True)
        recorded = json.loads((capture / "receipt.json").read_text())
        for name in ["source.mlir", "inputs.npz", "bf16_policy_golden.npy"]:
            if sha(capture / name) != recorded["files_sha256"][name]:
                raise ValueError("producer capture changed")
            (sources / name).write_bytes((capture / name).read_bytes())
        pair, = multiply_pairs(parse_mlir_file(capture / "source.mlir"), selected_elementwise_sites(
            (capture / "source.mlir").read_text(), profile), include_broadcasts=True)
        if prod(pair["shape"]) != m * k or args.refuse_index >= 0:
            raise ValueError("producer output extent or separate refusal request differs")
        golden = np.load(capture / "bf16_policy_golden.npy", allow_pickle=False).astype("<f4").reshape(m, k)
        if not np.isfinite(golden).all() or np.any(golden.view("<u4") & 65535):
            raise ValueError("producer reference must be finite exact BF16 widenings")
        words = (golden.view("<u4") >> 16).astype("<u2")
        with np.load(capture / "inputs.npz", allow_pickle=False) as arrays:
            inputs = [arrays[f"in{i}"] for i in range(3)]
        if any(v.dtype != np.float32 or list(v.shape) != shape for v, shape in zip(inputs, pair["input_shapes"], strict=True)):
            raise ValueError("producer inputs differ from actual source signature")
        producer_counts = tuple(v.size for v in inputs)
        producer_input = b"".join(v.astype("<f4").tobytes() for v in inputs)
        for dtype in ["f32", "bf16"]:
            selected = pair | {"symbol": "pack_producer_" + dtype, "kernel_symbol": "pack_producer_" + dtype}
            text, _ = emit_chain_kernel(selected, profile, prepare_broadcasts=True, output_dtype=dtype)
            path = sources / ("producer_" + dtype + ".c")
            path.write_text(text)
            producer_paths.append(path)
        producer = {"capture_receipt": recorded, "input_shapes": pair["input_shapes"], "shape": pair["shape"],
                    "boundary_policy": "bf16_rne_each_vpu_call", "prepare_broadcasts": True}
    values = (words.astype("<u4") << 16).view("<f4")
    codes = np.full(m * k * precision // 8, 0xA5, dtype=np.uint8)
    scales = np.full(m * k // 32, 0xA5, dtype=np.uint8)
    status = function(values.ctypes.data, m, k, precision, codes.ctypes.data, scales.ctypes.data)
    np.save(out / "input_words.npy", words)
    (out / "expected_codes.bin").write_bytes(codes.tobytes())
    (out / "expected_scales.bin").write_bytes(scales.tobytes())
    (out / "payload.bin").write_bytes(producer_input + values.tobytes() + words.tobytes() + codes.tobytes() + scales.tobytes())
    (out / "payload.S").write_text('.section .rodata\n.balign 64\n.global payload\npayload:\n.incbin "payload.bin"\n')
    (out / "driver.c").write_text(driver(m, k, precision, status, args.candidate_first, producer_counts))
    cc = args.riscv_root.resolve() / "bin/riscv64-unknown-elf-gcc"
    startup = args.startup_dir.resolve(strict=True)
    flags = [str(cc), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DBAREMETAL=1", "-mcmodel=medany", "-O2",
             "-ffreestanding", "-fno-fast-math", "-fno-common", "-fno-builtin-printf",
             "-fno-tree-loop-distribute-patterns", "-march=rv64gc_zicntr", "-I", str(sources), "-I", str(startup),
             "-I", str(startup.parent.parent), "-I", str(startup.parent.parent / "env")]
    common = []
    for i, path in enumerate((out / "driver.c", out / "payload.S", *producer_paths,
                             *sorted(startup.glob("*.c")), *sorted(startup.glob("*.S")))):
        obj = out / f"common{i}.o"
        run([*flags, "-c", str(path), "-o", str(obj)], f"compile_common{i}.log")
        common.append(obj)
    obj, elf = out / "quant.o", out / "section.elf"
    defines = ["-DMX_QUANT_GROUP_WORDS", "-DMX_QUANT_INLINE_GROUP", "-DMX_QUANT_BF16_INPUT"]
    run([*flags, *defines, "-c", str(sources / "runtime_quant.c"), "-o", str(obj)], "compile_quant.log")
    run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T", str(startup / "test.ld"),
         str(obj), *map(str, common), "-lm", "-lgcc", "-o", str(elf)], "link.log")
    route = DeviceRouting(device="mx_gemmini", package_dir=root, operand_dtype="f32", accum_dtype="bf16",
                          linked_elf_admission=service("mx_gemmini"))
    admission = admit_linked_device_elf(route, freeze_device_elf_admission(route), elf=elf,
                                       linked_sha256=sha(elf), directory=out)
    command = [str(args.riscv_root.resolve() / "bin/spike"), "--isa=rv64gc_zicntr",
               "--extlib=" + str(args.extlib.resolve()), "--extension=gemmini", str(elf)]
    environment = os.environ | {"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu",
                                "GEMMINI_PERF_SET": "mem.host_tracking=0", "GEMMINI_PERF_OUT": str(out / "perf.txt")}
    with (out / "spike.log").open("wb") as log:
        result = subprocess.run(command, cwd=out, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                timeout=1800, check=False)
    lines = (out / "spike.log").read_text().splitlines()
    rows = [dict((key, int(value)) for key, value in (part.split("=", 1) for part in line.split()[1:]))
            for line in lines if line.startswith("PACK ")]
    if (result.returncode or lines.count("DONE") != 1 or len(rows) != 6
            or [(r["run"], r["variant"]) for r in rows] != [
                (rep, order ^ (rep & 1) ^ int(args.candidate_first)) for rep in range(3) for order in range(2)]
            or any(r["status"] != status or r["mismatch"] or r["guard"] for r in rows)
            or len({r["hash"] for r in rows}) != 1):
        raise ValueError("compact packing native execution or complete-byte check failed")
    report = {"schema": "mx_gemmini.bf16_activation_pack_pair.v1", "scope": (
                  "connected_producer_plus_host_packing_section_no_model_timing" if producer else
                  "host_packing_section_only_no_producer_or_model_timing"), "producer": producer,
              "rows": m, "cols": k, "precision": precision, "reference_status": status,
              "refuse_index": args.refuse_index, "refuse_word": args.refuse_word, "candidate_first": args.candidate_first,
              "checks": rows, "elf_admission": admission, "defines": defines, "command": command,
              "environment": {key: environment[key] for key in ("GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET", "GEMMINI_PERF_OUT")},
              "commands": commands, "exit_code": result.returncode, "profile_sha256": profile_sha256(profile),
              "extension_sha256": sha(args.extlib), "gcc_sha256": sha(cc), "spike_sha256": sha(args.riscv_root / "bin/spike"),
              "source_sha256": {p.name: sha(p) for p in sources.iterdir()},
              "files_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    (out / "receipt.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
