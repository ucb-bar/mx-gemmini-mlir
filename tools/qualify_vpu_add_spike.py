"""Check generated MX VPU add against explicit BF16 rounding on Spike."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vpu_add_kernel import emit_vpu_add
from tools.qualify_vpu_row_softmax_spike import _run


ROOT = Path(__file__).resolve().parents[1]
PROFILE = (ROOT / "profiles/gemmini-mx-cleanup-266c593/"
           "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _driver(rows: int, width: int, dtype: str, broadcast: str,
            symbol: str) -> str:
    scalar = "float" if dtype == "f32" else "uint16_t"
    rhs_elements = width if broadcast == "last_dimension" else rows * width
    rhs_index = "(index % WIDTH)" if broadcast == "last_dimension" else "index"
    input_store = ("a[index] = av;" if dtype == "f32" else
                   "a[index] = narrow(av);")
    rhs_store = ("b[index] = bv;" if dtype == "f32" else
                 "b[index] = narrow(bv);")
    observed = ("bits(out[index])" if dtype == "f32" else
                "((uint32_t)out[index] << 16)")
    a_value = ("widen(narrow(a[index]))" if dtype == "f32" else
               "widen(a[index])")
    b_value = (f"widen(narrow(b[{rhs_index}]))" if dtype == "f32" else
               f"widen(b[{rhs_index}])")
    return f'''#include <stdint.h>
#include <stdio.h>
#define ROWS {rows}
#define WIDTH {width}
#define ELEMENTS (ROWS * WIDTH)
void {symbol}(const {scalar} *, const {scalar} *, {scalar} *);
static {scalar} a[ELEMENTS] __attribute__((aligned(64)));
static {scalar} b[{rhs_elements}] __attribute__((aligned(64)));
static {scalar} out[ELEMENTS] __attribute__((aligned(64)));
static uint32_t bits(float value) {{ union {{float f; uint32_t u;}} w; w.f=value; return w.u; }}
static float widen(uint16_t value) {{union {{float f; uint32_t u;}} w; w.u=(uint32_t)value<<16; return w.f;}}
static uint16_t narrow(float value) {{uint32_t u=bits(value); u+=0x7fffu+((u>>16)&1u); return u>>16;}}
int main(void) {{
  for (unsigned index=0; index<ELEMENTS; ++index) {{
    float av=((int)((index*17u)%257u)-128)/64.0f;
    {input_store}
  }}
  for (unsigned index=0; index<{rhs_elements}; ++index) {{
    float bv=((int)((index*11u)%197u)-98)/64.0f;
    {rhs_store}
  }}
  {symbol}(a,b,out);
  unsigned mismatches=0;
  for (unsigned index=0; index<ELEMENTS; ++index) {{
    uint32_t want=(uint32_t)narrow({a_value}+{b_value})<<16;
    mismatches += {observed} != want;
  }}
  printf("VPU ADD %s mismatches=%u\\n", mismatches ? "FAIL" : "PASS", mismatches);
  return mismatches ? 3 : 0;
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--dtype", choices=("bf16", "f32"), required=True)
    parser.add_argument("--broadcast", choices=("none", "last_dimension"), required=True)
    for name in ("rtl-root", "riscv-root", "extlib", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.absolute()
    if out.exists():
        parser.error("output directory already exists")
    profile = load_profile(PROFILE)
    symbol = "mx_model_add"
    kernel = emit_vpu_add(profile, elements=args.rows * args.width,
                          row_width=args.width, dtype=args.dtype,
                          broadcast=args.broadcast, symbol=symbol)
    out.mkdir(parents=True)
    source = out / "mx_add.c"
    source.write_text(kernel.source)
    driver = out / "driver.c"
    driver.write_text(_driver(args.rows, args.width, args.dtype,
                              args.broadcast, symbol))
    software = args.rtl_root / "software/gemmini-rocc-tests"
    bench = software / "riscv-tests/benchmarks/common"
    if not bench.is_dir() or not (bench / "test.ld").is_file():
        raise ValueError("RTL checkout lacks Nicolas's baremetal startup")
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET",
             "-DBAREMETAL=1", "-mcmodel=medany", "-std=gnu99", "-O2",
             "-fno-common", "-fno-builtin-printf",
             "-fno-tree-loop-distribute-patterns", "-march=rv64gc",
             "-Wa,-march=rv64gc", "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"), "-I", str(software),
             "-I", str(bench)]
    objects = []
    for index, path in enumerate((source, driver, *sorted(bench.glob("*.c")),
                                  *sorted(bench.glob("*.S")))):
        obj = out / f"part_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(obj)], out,
             f"compile_{index}.log")
        objects.append(obj)
    elf = out / "add.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(obj) for obj in objects), "-lm",
          "-lgcc", "-o", str(elf)], out, "link.log")
    extlib = args.extlib.resolve(strict=True)
    command = [str(args.riscv_root / "bin/spike"), "--isa=rv64gc",
               f"--extlib={extlib}", "--extension=gemmini", str(elf)]
    result = subprocess.run(command, cwd=out, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=300)
    log = out / "spike.log"
    log.write_text(result.stdout)
    passed = result.returncode == 0 and "VPU ADD PASS mismatches=0" in result.stdout
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema": "mx_gemmini.vpu_add_spike.v1",
        "status": "spike_bit_exact_bf16_add" if passed else "failed",
        "shape": [args.rows, args.width], "dtype": args.dtype,
        "broadcast": args.broadcast,
        "elements_per_chunk": kernel.elements_per_chunk,
        "profile_sha256": profile_sha256(profile),
        "source_sha256": sha(source), "elf_sha256": sha(elf),
        "extlib_sha256": sha(extlib), "spike_log_sha256": sha(log),
        "spike_exit_code": result.returncode,
        "numeric_line": next((line for line in result.stdout.splitlines()
                              if line.startswith("VPU ADD ")), None),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                  sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    if not passed:
        raise RuntimeError(f"generated VPU add failed; see {log}")


if __name__ == "__main__":
    main()
