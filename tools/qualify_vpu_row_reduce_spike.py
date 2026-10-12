"""Qualify generated VPU BF16 row sum/max against a scalar row oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vpu_row_reduce import emit_vpu_row_reduce
from tools.qualify_vpu_row_softmax_spike import _run


ROOT = Path(__file__).resolve().parents[1]
PROFILE = (ROOT / "profiles/gemmini-mx-cleanup-266c593/"
           "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _driver(rows: int, width: int, operation: str, symbol: str,
            signed_input: bool) -> str:
    update = ("reference += value;" if operation == "rsum" else
              "if (value > reference) reference = value;")
    initial = "0.0f" if operation == "rsum" else "-1.0e20f"
    value = ("((float)((int)((index*17u)%127u)-63))/64.0f" if signed_input
             else "0.25f+(float)((index*17u)%127u)/128.0f")
    return f'''#include <stdint.h>
#include <stdio.h>
#define ROWS {rows}
#define WIDTH {width}
void {symbol}(const float *, float *);
static float input[ROWS*WIDTH] __attribute__((aligned(64)));
static float output[ROWS] __attribute__((aligned(64)));
static uint32_t bits(float value) {{ union {{float f; uint32_t u;}} w; w.f=value; return w.u; }}
static float widen(uint16_t value) {{union {{float f; uint32_t u;}} w; w.u=(uint32_t)value<<16; return w.f;}}
static uint16_t narrow(float value) {{uint32_t u=bits(value); u+=0x7fffu+((u>>16)&1u); return u>>16;}}
int main(void) {{
  for (unsigned index=0; index<ROWS*WIDTH; ++index)
    input[index]={value};
  {symbol}(input,output);
  unsigned max_relative_ppm=0, max_absolute_ppm=0, mismatches=0;
  uint32_t first_observed=bits(output[0]), first_reference=0;
  for (unsigned row=0; row<ROWS; ++row) {{
    float reference={initial};
    for (unsigned col=0; col<WIDTH; ++col) {{
      float value=widen(narrow(input[row*WIDTH+col]));
      {update}
    }}
    if (row==0) first_reference=bits(reference);
    float delta=__builtin_fabsf(output[row]-reference);
    float denominator=__builtin_fabsf(reference);
    if (denominator<1.0f) denominator=1.0f;
    unsigned relative_ppm=(unsigned)(delta/denominator*1000000.0f);
    unsigned absolute_ppm=(unsigned)(delta*1000000.0f);
    if (absolute_ppm>max_absolute_ppm) max_absolute_ppm=absolute_ppm;
    if (relative_ppm>max_relative_ppm) max_relative_ppm=relative_ppm;
    mismatches += {1 if operation == 'rmax' else 0} && bits(output[row])!=bits(reference);
  }}
  printf("VPU {operation.upper()} max_rel_ppm=%u max_abs_ppm=%u mismatches=%u first_observed=%u first_reference=%u\\n",
         max_relative_ppm,max_absolute_ppm,mismatches,first_observed,first_reference);
  return mismatches || max_relative_ppm>50000 ? 3 : 0;
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--operation", choices=("rsum", "rmax"), required=True)
    parser.add_argument("--signed-input", action="store_true")
    for name in ("rtl-root", "riscv-root", "extlib", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.absolute()
    if out.exists():
        parser.error("output directory already exists")
    profile = load_profile(PROFILE)
    symbol = "mx_model_reduce"
    kernel = emit_vpu_row_reduce(profile, logical_rows=args.rows,
                                 row_width=args.width,
                                 operation=args.operation, symbol=symbol)
    out.mkdir(parents=True)
    source = out / "mx_reduce.c"
    source.write_text(kernel.source)
    driver = out / "driver.c"
    driver.write_text(_driver(args.rows, args.width, args.operation, symbol,
                              args.signed_input))
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
    elf = out / "reduce.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(obj) for obj in objects), "-lm",
          "-lgcc", "-o", str(elf)], out, "link.log")
    extlib = args.extlib.resolve(strict=True)
    result = subprocess.run([
        str(args.riscv_root / "bin/spike"), "--isa=rv64gc",
        f"--extlib={extlib}", "--extension=gemmini", str(elf)],
        cwd=out, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=300)
    log = out / "spike.log"
    log.write_text(result.stdout)
    line = next((line for line in result.stdout.splitlines()
                 if line.startswith(f"VPU {args.operation.upper()} ")), None)
    passed = result.returncode == 0 and line is not None
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema": "mx_gemmini.vpu_row_reduce_spike.v1",
        "status": "spike_row_reduce_within_5pct" if passed else "failed",
        "shape": [args.rows, args.width], "operation": args.operation,
        "signed_input": args.signed_input,
        "padded_width": kernel.padded_width,
        "rows_per_chunk": kernel.rows_per_chunk,
        "profile_sha256": profile_sha256(profile),
        "source_sha256": sha(source), "elf_sha256": sha(elf),
        "extlib_sha256": sha(extlib), "spike_log_sha256": sha(log),
        "spike_exit_code": result.returncode, "numeric_line": line,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                  sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    if not passed:
        raise RuntimeError(f"generated VPU row reduction failed; see {log}")


if __name__ == "__main__":
    main()
