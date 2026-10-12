"""Qualify generated VPU unary BF16 arithmetic against scalar oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import prod
from pathlib import Path
import subprocess

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vpu_unary_kernel import emit_vpu_unary
from tools.qualify_vpu_row_softmax_spike import _run


ROOT = Path(__file__).resolve().parents[1]
PROFILE = (ROOT / "profiles/gemmini-mx-cleanup-266c593/"
           "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _shape(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be comma-separated integers") from exc
    if not 1 <= len(result) <= 4 or any(dim <= 0 for dim in result):
        raise argparse.ArgumentTypeError("shape needs 1..4 positive dimensions")
    return result


def _driver(shape: tuple[int, ...], dtype: str, operation: str,
            symbol: str) -> str:
    scalar = "float" if dtype == "f32" else "uint16_t"
    input_store = "input[index] = value;" if dtype == "f32" else "input[index] = narrow(value);"
    input_value = "widen(narrow(input[index]))" if dtype == "f32" else "widen(input[index])"
    output_value = "output[index]" if dtype == "f32" else "widen(output[index])"
    test_value = ("-1.0f + (float)((index*17u)%127u)/126.0f" if operation == "pow10000"
                  else "0.125f + (float)((index*17u)%127u)/32.0f" if operation in {"rsqrt", "square"}
                  else "-3.125f + (float)((index*17u)%127u)/20.0f" if operation == "neg"
                  else "(float)((int)((index*17u)%127u)-63)/20.0f")
    oracle = (f"1.0f/__builtin_sqrtf({input_value})" if operation == "rsqrt"
              else f"1.0f/(1.0f+exp_approx(-({input_value})))" if operation == "sigmoid"
              else f"2.0f/(1.0f+exp_ref(-2.0*({input_value})))-1.0f" if operation == "tanh"
              else f"({input_value})*({input_value})" if operation == "square"
              else f"exp_ref(9.210340371976184 * ({input_value}))" if operation == "pow10000"
              else (f"({input_value})/(1.0f+exp_approx(-1.595769166f*"
                    f"(({input_value})+0.044715f*({input_value})*"
                    f"({input_value})*({input_value}))))") if operation == "gelu"
              else f"-({input_value})")
    relative_limit_ppm = (50000 if operation == "pow10000" else
                          20000 if operation in {"gelu", "tanh"} else 10000)
    denominator = ("(denominator>1.0f ? denominator : 1.0f)" if operation in {"gelu", "tanh"}
                   else "(denominator>0.000001f ? denominator : 0.000001f)")
    return f'''#include <stdint.h>
#include <stdio.h>
#define ELEMENTS {prod(shape)}
void {symbol}(const {scalar} *, {scalar} *);
static {scalar} input[ELEMENTS] __attribute__((aligned(64)));
static {scalar} output[ELEMENTS] __attribute__((aligned(64)));
static uint32_t bits(float value) {{ union {{float f; uint32_t u;}} w; w.f=value; return w.u; }}
static float widen(uint16_t value) {{union {{float f; uint32_t u;}} w; w.u=(uint32_t)value<<16; return w.f;}}
static uint16_t narrow(float value) {{uint32_t u=bits(value); u+=0x7fffu+((u>>16)&1u); return u>>16;}}
static float exp_approx(float value) {{
  float term=1.0f, sum=1.0f;
  for (unsigned n=1; n<=30; ++n) {{ term*=value/(float)n; sum+=term; }}
  return sum;
}}
static float exp_ref(double value) {{
  double term=1.0, sum=1.0;
  for (unsigned n=1; n<=80; ++n) {{ term*=value/(double)n; sum+=term; }}
  return (float)sum;
}}
int main(void) {{
  for (unsigned index=0; index<ELEMENTS; ++index) {{
    float value={test_value};
    {input_store}
  }}
  {symbol}(input,output);
  unsigned max_relative_ppm=0, max_absolute_ppm=0;
  for (unsigned index=0; index<ELEMENTS; ++index) {{
    float expected={oracle};
    float observed={output_value};
    float delta=__builtin_fabsf(observed-expected);
    unsigned absolute_ppm=(unsigned)(delta*1000000.0f);
    float denominator=__builtin_fabsf(expected);
    unsigned relative_ppm=(unsigned)(delta/{denominator}*1000000.0f);
    if (absolute_ppm>max_absolute_ppm) max_absolute_ppm=absolute_ppm;
    if (relative_ppm>max_relative_ppm) max_relative_ppm=relative_ppm;
  }}
  printf("VPU {operation.upper()} max_abs_ppm=%u max_rel_ppm=%u\\n",max_absolute_ppm,max_relative_ppm);
  return (max_relative_ppm<={relative_limit_ppm} &&
          {"max_absolute_ppm<=3000" if operation == "pow10000" else "1"}) ? 0 : 3;
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_shape, required=True)
    parser.add_argument("--dtype", choices=("bf16", "f32"), required=True)
    parser.add_argument("--operation", choices=("rsqrt", "sigmoid", "tanh", "square", "neg", "gelu", "pow10000"), default="rsqrt")
    for name in ("rtl-root", "riscv-root", "extlib", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.absolute()
    if out.exists():
        parser.error("output directory already exists")
    profile = load_profile(PROFILE)
    symbol = "mx_model_unary"
    kernel = emit_vpu_unary(profile, shape=args.shape, dtype=args.dtype,
                            operation=args.operation, symbol=symbol)
    out.mkdir(parents=True)
    source = out / "mx_unary.c"
    source.write_text(kernel.source)
    driver = out / "driver.c"
    driver.write_text(_driver(args.shape, args.dtype, args.operation, symbol))
    software = args.rtl_root / "software/gemmini-rocc-tests"
    bench = software / "riscv-tests/benchmarks/common"
    if not bench.is_dir() or not (bench / "test.ld").is_file():
        raise ValueError("RTL checkout lacks Nicolas's baremetal startup")
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET",
             "-DBAREMETAL=1", "-mcmodel=medany", "-std=gnu99", "-O2",
             "-fno-common", "-fno-builtin-printf",
             "-fno-tree-loop-distribute-patterns", "-fno-math-errno", "-march=rv64gc",
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
    elf = out / "unary.elf"
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
        "schema": "mx_gemmini.vpu_unary_spike.v1",
        "status": (f"spike_{args.operation}_within_5pct_relative_0p003_absolute"
                   if args.operation == "pow10000" else
                   f"spike_{args.operation}_within_2pct_absolute_scaled"
                   if args.operation in {"gelu", "tanh"} else
                   f"spike_{args.operation}_within_1pct") if passed else "failed",
        "shape": list(args.shape), "dtype": args.dtype,
        "operation": args.operation,
        "elements_per_chunk": kernel.elements_per_chunk,
        "profile_sha256": profile_sha256(profile),
        "source_sha256": sha(source), "elf_sha256": sha(elf),
        "extlib_sha256": sha(extlib), "spike_log_sha256": sha(log),
        "spike_exit_code": result.returncode, "numeric_line": line,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                  sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    if not passed:
        raise RuntimeError(f"generated VPU {args.operation} failed; see {log}")


if __name__ == "__main__":
    main()
