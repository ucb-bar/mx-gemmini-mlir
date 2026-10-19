"""Qualify a model-shaped generated MX VPU row softmax on the reference implementation’s Spike."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vpu_row_softmax import emit_vpu_row_softmax


ROOT = Path(__file__).resolve().parents[1]
PROFILE = (ROOT / "profiles/gemmini-mx-cleanup-266c593/"
           "MxE4M3Fp4VpuGemminiRocketConfig.json")


def _run(argv: list[str], out: Path, log: str) -> None:
    result = subprocess.run(argv, cwd=out, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    (out / log).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed; see {out / log}")


def _driver(rows: int, width: int, symbol: str, *, causal_mask: bool = False,
            score_divisor: float = 8.0, score_codes: tuple[int, ...] | None = None,
            score_scale: float = 0.125) -> str:
    encoded = (", ".join(f"0x{value:04x}" for value in score_codes)
               if score_codes is not None else "0x0000")
    code_count = rows * width if score_codes is not None else 1
    return f'''#include <math.h>
#include <stdint.h>
#include <stdio.h>
void {symbol}(const float *, float *);
int *__errno(void) {{ static int value; return &value; }}
static const uint16_t source_scores[{code_count}] = {{{encoded}}};
static float input[{rows} * {width}] __attribute__((aligned(64)));
static float output[{rows} * {width}] __attribute__((aligned(64)));
int main(void) {{
  for (unsigned row = 0; row < {rows}; ++row)
    for (unsigned col = 0; col < {width}; ++col) {{
      input[row * {width} + col] =
          ((int)((row * 13 + col * 7) % 41) - 20) / {score_divisor}f;
      if ({1 if score_codes is not None else 0}) {{
        union {{ float value; uint32_t bits; }} score;
        score.bits = (uint32_t)source_scores[row * {width} + col] << 16;
        input[row * {width} + col] = score.value * {score_scale}f;
      }}
      if ({1 if causal_mask else 0} && col > row % {width})
        input[row * {width} + col] = -INFINITY;
    }}
  {symbol}(input, output);
  int max_abs_ppm = 0, max_sum_ppm = 0, max_masked_ppm = 0;
  for (unsigned row = 0; row < {rows}; ++row) {{
    float maximum = -1.0e30f, denominator = 0.0f, observed_sum = 0.0f;
    for (unsigned col = 0; col < {width}; ++col)
      if (input[row * {width} + col] > maximum)
        maximum = input[row * {width} + col];
    for (unsigned col = 0; col < {width}; ++col)
      denominator += expf(input[row * {width} + col] - maximum);
    for (unsigned col = 0; col < {width}; ++col) {{
      float got = output[row * {width} + col];
      float want = expf(input[row * {width} + col] - maximum) / denominator;
      if (!(got >= 0.0f && got <= 1.0f)) return 3;
      int err = (int)(fabsf(got - want) * 1000000.0f + 0.5f);
      if (err > max_abs_ppm) max_abs_ppm = err;
      if (input[row * {width} + col] == -INFINITY && err > max_masked_ppm)
        max_masked_ppm = err;
      observed_sum += got;
    }}
    int sum_err = (int)(fabsf(observed_sum - 1.0f) * 1000000.0f + 0.5f);
    if (sum_err > max_sum_ppm) max_sum_ppm = sum_err;
  }}
  int pass = max_abs_ppm <= 30000 && max_sum_ppm <= 50000 && max_masked_ppm <= 1000;
  printf("SOFTMAX %s max_abs_ppm=%d max_sum_ppm=%d max_masked_ppm=%d\\n",
         pass ? "PASS" : "FAIL", max_abs_ppm, max_sum_ppm, max_masked_ppm);
  return pass ? 0 : 4;
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-rows", type=int, required=True)
    parser.add_argument("--row-width", type=int, required=True)
    parser.add_argument("--causal-mask", action="store_true",
                        help="mask keys after each query position with negative infinity")
    parser.add_argument("--score-divisor", type=float, default=8.0,
                        help="synthetic score amplitude is 20 divided by this value")
    parser.add_argument("--scores-bf16", type=Path,
                        help="raw little-endian BF16 score matrix before softmax scaling")
    parser.add_argument("--score-scale", type=float, default=0.125,
                        help="scale applied to BF16 score fixture before masking")
    parser.add_argument("--rtl-root", type=Path, required=True)
    parser.add_argument("--riscv-root", type=Path, required=True)
    parser.add_argument("--extlib", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0.1 <= args.score_divisor <= 1000.0 or not 0.0 < args.score_scale <= 100.0:
        parser.error("score divisor must be within 0.1..1000 and scale within 0..100")
    out = args.out_dir.absolute()
    if out.exists():
        parser.error("output directory already exists")
    profile = load_profile(PROFILE)
    symbol = "mx_model_softmax"
    codes = None
    fixture_sha = None
    if args.scores_bf16 is not None:
        fixture = args.scores_bf16.resolve(strict=True)
        if not fixture.is_file() or fixture.is_symlink():
            parser.error("BF16 scores must be a regular file")
        raw = fixture.read_bytes()
        if len(raw) != args.logical_rows * args.row_width * 2:
            parser.error("BF16 score fixture length does not match the requested shape")
        fixture_sha = hashlib.sha256(raw).hexdigest()
        codes = tuple(int.from_bytes(raw[i:i + 2], "little")
                      for i in range(0, len(raw), 2))
    kernel = emit_vpu_row_softmax(profile, logical_rows=args.logical_rows,
                                  row_width=args.row_width, symbol=symbol)
    out.mkdir(parents=True)
    source = out / "mx_softmax.c"
    source.write_text(kernel.source)
    driver = out / "driver.c"
    driver.write_text(_driver(args.logical_rows, args.row_width, symbol,
                              causal_mask=args.causal_mask,
                              score_divisor=args.score_divisor,
                              score_codes=codes, score_scale=args.score_scale))
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
    elf = out / "softmax.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(obj) for obj in objects), "-lm",
          "-lgcc", "-o", str(elf)], out, "link.log")
    spike = args.riscv_root / "bin/spike"
    extension = args.extlib.resolve(strict=True)
    command = [str(spike), "--isa=rv64gc", f"--extlib={extension}",
               "--extension=gemmini", str(elf)]
    result = subprocess.run(command, cwd=out, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=300)
    log = out / "spike.log"
    log.write_text(result.stdout)
    passed = result.returncode == 0 and "SOFTMAX PASS " in result.stdout
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "schema": "mx_gemmini.vpu_row_softmax_spike.v1",
        "status": "spike_numeric_tolerance_pass" if passed else "failed",
        "shape": [args.logical_rows, args.row_width],
        "causal_mask": args.causal_mask,
        "score_divisor": args.score_divisor,
        "score_codes_sha256": fixture_sha,
        "score_scale": args.score_scale,
        "padded_width": kernel.padded_width,
        "rows_per_chunk": kernel.rows_per_chunk,
        "vpu_steps": list(kernel.vpu_steps),
        "profile_sha256": profile_sha256(profile),
        "source_sha256": digest(source), "driver_sha256": digest(driver),
        "elf_sha256": digest(elf),
        "extlib_sha256": digest(extension), "spike_log_sha256": digest(log),
        "spike_exit_code": result.returncode,
        "numeric_line": next((line for line in result.stdout.splitlines()
                              if line.startswith("SOFTMAX ")), None),
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                  sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    if not passed:
        raise RuntimeError(f"generated VPU softmax failed; see {log}")


if __name__ == "__main__":
    main()
