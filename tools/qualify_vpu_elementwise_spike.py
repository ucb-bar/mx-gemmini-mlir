"""Qualify generated VPU binary arithmetic against an independent BF16 loop."""

from __future__ import annotations

import argparse
import hashlib
import json
from math import prod
from pathlib import Path
import subprocess

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vpu_add_kernel import emit_vpu_elementwise
from tools.qualify_vpu_row_softmax_spike import _run


ROOT = Path(__file__).resolve().parents[1]
PROFILE = (ROOT / "profiles/gemmini-mx-cleanup-266c593/"
           "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _shape(value: str) -> tuple[int, ...]:
    if value == "scalar":
        return ()
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be comma-separated integers") from exc
    if not 1 <= len(result) <= 4 or any(dim <= 0 for dim in result):
        raise argparse.ArgumentTypeError("shape needs 1..4 positive dimensions")
    return result


def _axes(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("axes must be comma-separated integers") from exc
    if not result or any(axis < 0 for axis in result):
        raise argparse.ArgumentTypeError("axes must be nonnegative integers")
    return result


def _driver(shape: tuple[int, ...], lhs_shape: tuple[int, ...],
            lhs_axes: tuple[int, ...], rhs_shape: tuple[int, ...],
            rhs_axes: tuple[int, ...], dtype: str, lhs_dtype: str,
            rhs_dtype: str, operation: str,
            symbol: str, *, include_negative_infinity: bool = False) -> str:
    scalar = "float" if dtype == "f32" else "uint16_t"
    lhs_scalar = "float" if lhs_dtype == "f32" else "uint16_t"
    rhs_scalar = "float" if rhs_dtype == "f32" else "uint16_t"
    lhs_store = "a[index] = av;" if lhs_dtype == "f32" else "a[index] = narrow(av);"
    rhs_store = "b[index] = bv;" if rhs_dtype == "f32" else "b[index] = narrow(bv);"
    lhs_value = ("widen(narrow(a[a_index]))" if lhs_dtype == "f32" else
                 "widen(a[a_index])")
    rhs_value = ("widen(narrow(b[b_index]))" if rhs_dtype == "f32" else
                 "widen(b[b_index])")
    observed = "bits(out[index])" if dtype == "f32" else "((uint32_t)out[index] << 16)"
    arithmetic = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[operation]
    lhs_test_value = ("0.25f + (float)((index*17u)%257u)/64.0f" if operation == "div"
                      else "((int)((index*17u)%257u)-128)/64.0f")
    rhs_test_value = ("0.5f + (float)((index*11u)%197u)/64.0f" if operation == "div"
                      else "((int)((index*11u)%197u)-98)/64.0f")
    if include_negative_infinity:
        rhs_test_value = f"(index % 17u == 0u ? -__builtin_inff() : {rhs_test_value})"
    numeric_check = (f'''float expected={lhs_value} / {rhs_value};
    float delta=__builtin_fabsf({"out[index]" if dtype == "f32" else "widen(out[index])"}-expected);
    unsigned relative_ppm=(unsigned)(delta/expected*1000000.0f);
    if (relative_ppm>max_relative_ppm) max_relative_ppm=relative_ppm;'''
                     if operation == "div" else
                     f'''uint32_t want=(uint32_t)narrow({lhs_value} {arithmetic} {rhs_value})<<16;
    mismatches += {observed} != want;''')
    result_line = ("max_rel_ppm=%u" if operation == "div" else
                   "%s mismatches=%u")
    result_args = ("max_relative_ppm" if operation == "div" else
                   'mismatches ? "FAIL" : "PASS", mismatches')
    exit_check = "max_relative_ppm>10000" if operation == "div" else "mismatches"
    dims = ", ".join(map(str, shape))
    rhs_dims = ", ".join(map(str, rhs_shape))
    rhs_axis_values = ", ".join(map(str, rhs_axes))
    return f'''#include <stdint.h>
#include <stdio.h>
#define ELEMENTS {prod(shape)}
#define LHS_ELEMENTS {prod(lhs_shape)}
#define RHS_ELEMENTS {prod(rhs_shape)}
#define RANK {len(shape)}
#define LHS_RANK {len(lhs_shape)}
#define RHS_RANK {len(rhs_shape)}
void {symbol}(const {lhs_scalar} *, const {rhs_scalar} *, {scalar} *);
static {lhs_scalar} a[LHS_ELEMENTS] __attribute__((aligned(64)));
static {rhs_scalar} b[RHS_ELEMENTS] __attribute__((aligned(64)));
static {scalar} out[ELEMENTS] __attribute__((aligned(64)));
static const unsigned dims[RANK] = {{{dims}}};
static const unsigned lhs_dims[LHS_RANK ? LHS_RANK : 1] = {{{', '.join(map(str, lhs_shape)) or '1'}}};
static const unsigned lhs_axes[LHS_RANK ? LHS_RANK : 1] = {{{', '.join(map(str, lhs_axes)) or '0'}}};
static const unsigned rhs_dims[RHS_RANK ? RHS_RANK : 1] = {{{rhs_dims or '1'}}};
static const unsigned rhs_axes[RHS_RANK ? RHS_RANK : 1] = {{{rhs_axis_values or '0'}}};
static uint32_t bits(float value) {{ union {{float f; uint32_t u;}} w; w.f=value; return w.u; }}
static float widen(uint16_t value) {{union {{float f; uint32_t u;}} w; w.u=(uint32_t)value<<16; return w.f;}}
static uint16_t narrow(float value) {{uint32_t u=bits(value); u+=0x7fffu+((u>>16)&1u); return u>>16;}}
static unsigned reference_lhs_index(unsigned index) {{
  unsigned coords[RANK], remainder = index;
  for (int axis=RANK-1; axis>=0; --axis) {{
    coords[axis] = remainder % dims[axis];
    remainder /= dims[axis];
  }}
  unsigned flat = 0;
  for (unsigned axis=0; axis<LHS_RANK; ++axis) {{
    unsigned coordinate = lhs_dims[axis] == 1 ? 0 : coords[lhs_axes[axis]];
    flat = flat * lhs_dims[axis] + coordinate;
  }}
  return flat;
}}
static unsigned reference_rhs_index(unsigned index) {{
  unsigned coords[RANK], remainder = index;
  for (int axis=RANK-1; axis>=0; --axis) {{
    coords[axis] = remainder % dims[axis];
    remainder /= dims[axis];
  }}
  unsigned flat = 0;
  for (unsigned axis=0; axis<RHS_RANK; ++axis) {{
    unsigned coordinate = rhs_dims[axis] == 1 ? 0 : coords[rhs_axes[axis]];
    flat = flat * rhs_dims[axis] + coordinate;
  }}
  return flat;
}}
int main(void) {{
  for (unsigned index=0; index<LHS_ELEMENTS; ++index) {{
    float av={lhs_test_value};
    {lhs_store}
  }}
  for (unsigned index=0; index<RHS_ELEMENTS; ++index) {{
    float bv={rhs_test_value};
    {rhs_store}
  }}
  {symbol}(a,b,out);
  unsigned mismatches=0, max_relative_ppm=0;
  for (unsigned index=0; index<ELEMENTS; ++index) {{
    unsigned a_index=reference_lhs_index(index);
    unsigned b_index=reference_rhs_index(index);
    {numeric_check}
  }}
  printf("VPU {operation.upper()} {result_line}\\n", {result_args});
  return {exit_check} ? 3 : 0;
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_shape, required=True)
    parser.add_argument("--lhs-shape", type=_shape)
    parser.add_argument("--lhs-axes", type=_axes)
    parser.add_argument("--rhs-shape", type=_shape, required=True)
    parser.add_argument("--rhs-axes", type=_axes)
    parser.add_argument("--dtype", choices=("bf16", "f32"), required=True)
    parser.add_argument("--lhs-dtype", choices=("bf16", "f32"))
    parser.add_argument("--rhs-dtype", choices=("bf16", "f32"))
    parser.add_argument("--operation", choices=("mul", "sub", "add", "div"), required=True)
    parser.add_argument("--include-negative-infinity", action="store_true",
                        help="include masked-attention-like negative-infinity RHS values")
    for name in ("rtl-root", "riscv-root", "extlib", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.include_negative_infinity and args.operation != "add":
        parser.error("negative-infinity fixture is only defined for addition")
    for name in ("rtl_root", "riscv_root", "extlib"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    out = args.out_dir.absolute()
    if out.exists():
        parser.error("output directory already exists")
    profile = load_profile(PROFILE)
    lhs_dtype = args.lhs_dtype or args.dtype
    rhs_dtype = args.rhs_dtype or args.dtype
    lhs_shape = args.lhs_shape if args.lhs_shape is not None else args.shape
    lhs_axes = (args.lhs_axes if args.lhs_axes is not None else
                tuple(range(len(args.shape) - len(lhs_shape), len(args.shape))))
    rhs_axes = (args.rhs_axes if args.rhs_axes is not None else
                tuple(range(len(args.shape) - len(args.rhs_shape), len(args.shape))))
    symbol = "mx_model_elementwise"
    kernel = emit_vpu_elementwise(profile, result_shape=args.shape,
                                  lhs_shape=lhs_shape, lhs_axes=lhs_axes,
                                  rhs_shape=args.rhs_shape,
                                  dtype=args.dtype, lhs_dtype=lhs_dtype,
                                  rhs_dtype=rhs_dtype,
                                  operation=args.operation,
                                  symbol=symbol, rhs_axes=rhs_axes)
    out.mkdir(parents=True)
    source = out / "mx_elementwise.c"
    source.write_text(kernel.source)
    driver = out / "driver.c"
    driver.write_text(_driver(args.shape, lhs_shape, lhs_axes,
                              args.rhs_shape, rhs_axes, args.dtype,
                              lhs_dtype, rhs_dtype,
                              args.operation, symbol,
                              include_negative_infinity=args.include_negative_infinity))
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
             "-I", str(bench), "-I", str(args.riscv_root / "include/riscv")]
    objects = []
    for index, path in enumerate((source, driver, *sorted(bench.glob("*.c")),
                                  *sorted(bench.glob("*.S")))):
        obj = out / f"part_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(obj)], out,
             f"compile_{index}.log")
        objects.append(obj)
    elf = out / "elementwise.elf"
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
    passed = (result.returncode == 0 and
              (f"VPU {args.operation.upper()} max_rel_ppm=" in result.stdout
               if args.operation == "div" else
               f"VPU {args.operation.upper()} PASS mismatches=0" in result.stdout))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema": "mx_gemmini.vpu_elementwise_spike.v1",
        "status": ("spike_div_within_1pct" if args.operation == "div" else
                   "spike_bit_exact_bf16_elementwise") if passed else "failed",
        "shape": list(args.shape), "lhs_shape": list(lhs_shape),
        "lhs_axes": list(lhs_axes), "rhs_shape": list(args.rhs_shape),
        "rhs_axes": list(rhs_axes),
        "dtype": args.dtype, "lhs_dtype": lhs_dtype,
        "rhs_dtype": rhs_dtype, "operation": args.operation,
        "include_negative_infinity": args.include_negative_infinity,
        "elements_per_chunk": kernel.elements_per_chunk,
        "profile_sha256": profile_sha256(profile),
        "source_sha256": sha(source), "elf_sha256": sha(elf),
        "extlib_sha256": sha(extlib), "spike_log_sha256": sha(log),
        "spike_exit_code": result.returncode,
        "numeric_line": next((line for line in result.stdout.splitlines()
                              if line.startswith("VPU ")), None),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                  sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    if not passed:
        raise RuntimeError(f"generated VPU {args.operation} failed; see {log}")


if __name__ == "__main__":
    main()
