"""Run a model2MLIR small-LLaMA FP4 projection from live f32 tensors on Spike.

The RV64 image pads and MX-quantizes captured f32 operands at runtime, calls
the data-free MX issuer, widens its BF16 readout, and checks the independent
archived MX reference.  Gate and down projections are separate diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from mx_gemmini_support.physical_program import lower_bound_source
from mx_gemmini_support.dense_kernel import emit_dense_f32_kernel
from mx_gemmini_support.dense_contraction import emit_dense_contraction
from mx_gemmini_support.offline_weight_pack import pack_weight
from mx_gemmini_support.compact_runtime_issuer import emit_compact_runtime_issuer
from mx_gemmini_support.scheduled_native_issuer import emit_scheduled_native_issuer
from mx_gemmini_support.runtime_issuer import emit_runtime_issuer
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.target_profile import load_profile
from tools.compile_mx import _require_gitlink
from tools.qualify_runtime_issuer_spike import _run


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/model2mlir_small_llama_mlp_fp4_padding_e9ded36_266c593"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assembly_data(resources: dict[str, Path]) -> str:
    lines = [".section .rodata", ".balign 64"]
    for name, path in resources.items():
        escaped = str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')
        lines += [f".global {name}", f"{name}:", f'  .incbin "{escaped}"', ".balign 64"]
    return "\n".join(lines) + "\n"


def _driver(site: str, *, m: int, n: int, k: int,
            logical_m: int, logical_n: int, logical_k: int) -> str:
    return f'''#include <stdint.h>
#include <stdio.h>
#include "runtime_quant.h"
extern const float raw_a[], raw_b[];
extern const uint8_t expected_a[], expected_as[], expected_b[], expected_bs[];
extern const uint16_t golden_bf16[];
void mx_issue_encoded(const void *, const void *, const void *, const void *, const void *, const void *);
static float padded_a[{m} * {k}] __attribute__((aligned(64)));
static float padded_b[{k} * {n}] __attribute__((aligned(64)));
static uint8_t a_codes[{m} * {k} / 2] __attribute__((aligned(64)));
static uint8_t a_scales[{m} * {k} / 32] __attribute__((aligned(64)));
static uint8_t b_codes[{k} * {n} / 2] __attribute__((aligned(64)));
static uint8_t b_scales[{k} * {n} / 32] __attribute__((aligned(64)));
static uint8_t scratch[2048] __attribute__((aligned(64)));
static uint16_t output[{m} * {n}] __attribute__((aligned(64)));
static float widened[{m} * {n}] __attribute__((aligned(64)));
int main(void) {{
  for (unsigned row = 0; row < {logical_m}; ++row)
    for (unsigned col = 0; col < {logical_k}; ++col)
      padded_a[row * {k} + col] = raw_a[row * {logical_k} + col];
  for (unsigned row = 0; row < {logical_k}; ++row)
    for (unsigned col = 0; col < {logical_n}; ++col)
      padded_b[row * {n} + col] = raw_b[row * {logical_n} + col];
  if (mx_pack_activation_f32(padded_a, {m}, {k}, MX_RUNTIME_FP4, a_codes, a_scales) ||
      mx_pack_weight_f32(padded_b, {k}, {n}, MX_RUNTIME_FP4, b_codes, b_scales))
    return 2;
  unsigned pack_errors = 0, output_errors = 0;
  for (unsigned i = 0; i < sizeof a_codes; ++i) pack_errors += a_codes[i] != expected_a[i];
  for (unsigned i = 0; i < sizeof a_scales; ++i) pack_errors += a_scales[i] != expected_as[i];
  for (unsigned i = 0; i < sizeof b_codes; ++i) pack_errors += b_codes[i] != expected_b[i];
  for (unsigned i = 0; i < sizeof b_scales; ++i) pack_errors += b_scales[i] != expected_bs[i];
  if (pack_errors) return 3;
  mx_issue_encoded(a_codes, a_scales, b_codes, b_scales, output, scratch);
  if (mx_widen_bf16_f32(output, {m} * {n}, widened)) return 4;
  for (unsigned i = 0; i < {m} * {n}; ++i) {{
    output_errors += output[i] != golden_bf16[i];
    union {{ float f; uint32_t u; }} word;
    word.f = widened[i];
    output_errors += word.u != ((uint32_t)output[i] << 16);
  }}
  printf("live f32 MX {site}: %u pack mismatches, %u BF16/f32 mismatches\\n",
         pack_errors, output_errors);
  return output_errors != 0;
}}
'''


def _dense_driver(site: str, *, m: int, n: int, logical_m: int,
                  logical_n: int, time_dense: bool = False,
                  repeats: int = 1) -> str:
    clock = '''static uint64_t cycles(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
}
''' if time_dense else ""
    before = "    uint64_t before = cycles();\n" if time_dense else ""
    after = "    elapsed[iteration] = cycles() - before;\n" if time_dense else ""
    array = f"  uint64_t elapsed[{repeats}];\n" if time_dense else ""
    report = (f'''  for (unsigned iteration = 0; iteration < {repeats}; ++iteration)
    printf("DENSE_CYCLES %u %llu\\n", iteration, (unsigned long long)elapsed[iteration]);
''' if time_dense else "")
    return f'''#include <stdint.h>
#include <stdio.h>
extern const float raw_a[], raw_b[];
extern const uint16_t golden_bf16[];
void mx_dense_projection(const float *, const float *, float *);
static float output[{logical_m} * {logical_n}] __attribute__((aligned(64)));
{clock}
int main(void) {{
{array}  unsigned mismatches = 0;
  for (unsigned iteration = 0; iteration < {repeats}; ++iteration) {{
{before}    mx_dense_projection(raw_a, raw_b, output);
{after}    for (unsigned row = 0; row < {logical_m}; ++row)
      for (unsigned col = 0; col < {logical_n}; ++col) {{
        union {{ float value; uint32_t bits; }} word;
        word.value = output[row * {logical_n} + col];
        mismatches += word.bits != ((uint32_t)golden_bf16[row * {n} + col] << 16);
      }}
  }}
  printf("dense f32 MX {site}: %u output mismatches\\n", mismatches);
{report}
  return mismatches != 0;
}}
'''


def _phase_driver(site: str, *, m: int, n: int, k: int,
                  logical_m: int, logical_n: int, logical_k: int) -> str:
    """Measure live conversion and issue with the same archived operand bytes."""
    return f'''#include <stdint.h>
#include <stdio.h>
#include "runtime_quant.h"
extern const float raw_a[], raw_b[];
extern const uint8_t expected_a[], expected_as[], expected_b[], expected_bs[];
extern const uint16_t golden_bf16[];
void mx_issue_encoded(const void *, const void *, const void *, const void *, const void *, const void *);
static float padded_a[{m} * {k}] __attribute__((aligned(64)));
static float padded_b[{k} * {n}] __attribute__((aligned(64)));
static uint8_t a_codes[{m} * {k} / 2] __attribute__((aligned(64)));
static uint8_t a_scales[{m} * {k} / 32] __attribute__((aligned(64)));
static uint8_t b_codes[{k} * {n} / 2] __attribute__((aligned(64)));
static uint8_t b_scales[{k} * {n} / 32] __attribute__((aligned(64)));
static uint8_t scratch[2048] __attribute__((aligned(64)));
static uint16_t output[{m} * {n}] __attribute__((aligned(64)));
static float widened[{m} * {n}] __attribute__((aligned(64)));
static uint64_t cycles(void) {{
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
}}
int main(void) {{
  uint64_t c0 = cycles();
  for (unsigned row = 0; row < {logical_m}; ++row)
    for (unsigned col = 0; col < {logical_k}; ++col)
      padded_a[row * {k} + col] = raw_a[row * {logical_k} + col];
  for (unsigned row = 0; row < {logical_k}; ++row)
    for (unsigned col = 0; col < {logical_n}; ++col)
      padded_b[row * {n} + col] = raw_b[row * {logical_n} + col];
  uint64_t c1 = cycles();
  if (mx_pack_activation_f32(padded_a, {m}, {k}, MX_RUNTIME_FP4,
                             a_codes, a_scales)) return 2;
  uint64_t c2 = cycles();
  if (mx_pack_weight_f32(padded_b, {k}, {n}, MX_RUNTIME_FP4,
                         b_codes, b_scales)) return 3;
  uint64_t c3 = cycles();
  unsigned pack_errors = 0;
  for (unsigned i = 0; i < sizeof a_codes; ++i) pack_errors += a_codes[i] != expected_a[i];
  for (unsigned i = 0; i < sizeof a_scales; ++i) pack_errors += a_scales[i] != expected_as[i];
  for (unsigned i = 0; i < sizeof b_codes; ++i) pack_errors += b_codes[i] != expected_b[i];
  for (unsigned i = 0; i < sizeof b_scales; ++i) pack_errors += b_scales[i] != expected_bs[i];
  if (pack_errors) return 4;
  uint64_t c4 = cycles();
  mx_issue_encoded(a_codes, a_scales, b_codes, b_scales, output, scratch);
  uint64_t c5 = cycles();
  if (mx_widen_bf16_f32(output, {m} * {n}, widened)) return 5;
  uint64_t c6 = cycles();
  unsigned output_errors = 0;
  for (unsigned i = 0; i < {m} * {n}; ++i) {{
    output_errors += output[i] != golden_bf16[i];
    union {{ float f; uint32_t u; }} word;
    word.f = widened[i];
    output_errors += word.u != ((uint32_t)output[i] << 16);
  }}
  printf("live f32 MX {site}: %u pack mismatches, %u BF16/f32 mismatches\\n",
         pack_errors, output_errors);
  printf("PROFILE_CYCLES padding=%llu activation_pack=%llu weight_pack=%llu issue=%llu widen=%llu\\n",
         (unsigned long long)(c1-c0), (unsigned long long)(c2-c1),
         (unsigned long long)(c3-c2), (unsigned long long)(c5-c4),
         (unsigned long long)(c6-c5));
  return output_errors != 0;
}}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", choices=("g", "dn"), required=True)
    parser.add_argument("--dense-adapter", action="store_true",
                        help="run the generated three-pointer catalog adapter")
    parser.add_argument("--shape-planned", action="store_true",
                        help="derive the command schedule from profile and logical shape")
    parser.add_argument("--compact-issuer", action="store_true",
                        help="run the compact looped Rocket command issuer")
    parser.add_argument("--native-scheduled-issuer", action="store_true",
                        help="run the Merlin mk native DRAM-loop schedule")
    parser.add_argument("--phase-cycles", action="store_true",
                        help="measure padding, MXQuant packing, issue, and widening on Spike")
    parser.add_argument("--time-dense", action="store_true",
                        help="measure the full checked dense adapter on Spike")
    parser.add_argument("--prepacked-weight", action="store_true",
                        help="bind the archived immutable MX weight bytes to the dense adapter")
    parser.add_argument("--cached-weight", action="store_true",
                        help="pack the archived immutable RHS once per dense entry")
    parser.add_argument("--dense-repeats", type=int, default=1,
                        help="repeat the checked dense call to measure an immutable weight cache")
    for name in ("rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (args.shape_planned or args.compact_issuer or args.native_scheduled_issuer) and not (
            args.dense_adapter or args.phase_cycles):
        parser.error("selected schedule requires --dense-adapter or --phase-cycles")
    if args.phase_cycles and args.dense_adapter:
        parser.error("phase timing profiles the direct conversion path, not the opaque dense adapter")
    if args.prepacked_weight and not args.dense_adapter:
        parser.error("prepacked weight requires the dense adapter")
    if args.cached_weight and (not args.dense_adapter or args.prepacked_weight):
        parser.error("cached weight requires one dense adapter without a prepacked weight")
    if args.time_dense and not args.dense_adapter:
        parser.error("dense timing requires --dense-adapter")
    if not 1 <= args.dense_repeats <= 4 or (args.dense_repeats != 1 and not args.time_dense):
        parser.error("dense repeats must be 1..4 and require --time-dense above 1")
    if sum((args.shape_planned, args.compact_issuer, args.native_scheduled_issuer)) > 1:
        parser.error("choose one command issuer mode")
    rtl, riscv, out = (getattr(args, key).resolve()
                       for key in ("rtl_root", "riscv_root", "out_dir"))
    if out.exists():
        parser.error("output directory already exists")
    site = args.site
    archive = ARCHIVE / site
    software, extension = rtl / "software/gemmini-rocc-tests", rtl / "software/libgemmini"
    _require_gitlink(rtl, "software/gemmini-rocc-tests")
    _require_gitlink(rtl, "software/libgemmini")
    profile = load_profile(PROFILE, rtl_root=rtl)
    manifest, resources = load_bundle(archive / "bundle")
    program = lower_bound_source((archive / "projection.payload_bound.mlir").read_text(),
                                 profile, manifest, resources)
    qualified_issuer = emit_runtime_issuer(program, manifest)
    if qualified_issuer.precision != "FP4":
        raise ValueError("selected projection is not FP4")
    import numpy as np

    input_a = np.load(ARCHIVE / "inputs" / f"{site}.activation.npy", allow_pickle=False)
    input_b = np.load(ARCHIVE / "inputs" / f"{site}.weight.npy", allow_pickle=False)
    if input_a.dtype != np.float32 or input_b.dtype != np.float32:
        raise ValueError("archived model operands are not f32")
    logical_m, logical_k = input_a.shape
    second_k, logical_n = input_b.shape
    m, n, k = qualified_issuer.shape_mnk
    if (second_k != logical_k or logical_m > m or logical_n > n or logical_k > k):
        raise ValueError("archived model operand shapes differ from the MX program")
    generated = None
    if args.shape_planned:
        generated = emit_dense_contraction(
            profile, (logical_m, logical_n, logical_k), precision="FP4",
            symbol="mx_dense_projection", issuer_symbol="mx_issue_encoded")
        if (generated.shape_plan.padded_shape_mnk != qualified_issuer.shape_mnk or
                generated.shape_plan.tile_mnk != tuple(manifest["tile_mnk"]) or
                generated.issuer.source != qualified_issuer.source):
            raise ValueError("shape-planned commands differ from source-qualified projection")
    compact_issuer = None
    if args.compact_issuer:
        compact_issuer = emit_compact_runtime_issuer(
            profile, qualified_issuer.shape_mnk, tuple(manifest["tile_mnk"]),
            precision="FP4")
        if compact_issuer.plan != program.plan:
            raise ValueError("compact MX plan differs from the source-qualified projection")
    native = (emit_scheduled_native_issuer(
        profile, qualified_issuer.shape_mnk, precision="FP4")
        if args.native_scheduled_issuer else None)
    issuer = (native.issuer if native is not None else
              compact_issuer if compact_issuer is not None else
              generated.issuer if generated is not None else qualified_issuer)
    out.mkdir(parents=True)
    (out / "raw_a.bin").write_bytes(input_a.tobytes())
    (out / "raw_b.bin").write_bytes(input_b.tobytes())
    if args.prepacked_weight:
        packed_codes, packed_scales = pack_weight(
            input_b, precision="FP4", padded_shape=(k, n), workdir=out)
        if (packed_codes != (archive / "bundle/weight.bin").read_bytes()
                or packed_scales != (archive / "bundle/weight_scales.bin").read_bytes()):
            raise ValueError("offline MX weight bytes differ from the source-qualified bundle")
        (out / "prepacked_weight.bin").write_bytes(packed_codes)
        (out / "prepacked_weight_scales.bin").write_bytes(packed_scales)
    assembly = {
        "raw_a": out / "raw_a.bin", "raw_b": out / "raw_b.bin",
        "expected_a": archive / "bundle/activation.bin",
        "expected_as": archive / "bundle/activation_scales.bin",
        "expected_b": archive / "bundle/weight.bin",
        "expected_bs": archive / "bundle/weight_scales.bin",
        "golden_bf16": archive / "bundle/golden_bf16.bin",
    }
    if args.dense_adapter:
        assembly = {name: path for name, path in assembly.items()
                    if name in {"raw_a", "raw_b", "golden_bf16"}}
        if args.prepacked_weight:
            assembly.update(prepacked_weight=out / "prepacked_weight.bin",
                            prepacked_weight_scales=out / "prepacked_weight_scales.bin")
    (out / "mx_data.S").write_text(_assembly_data(assembly))
    (out / "mx_issue.c").write_text(issuer.source)
    dense_kernel = None
    if args.dense_adapter:
        dense_kernel = (generated.kernel if generated is not None and not (
                            args.prepacked_weight or args.cached_weight)
                        else emit_dense_f32_kernel(
                            issuer, (logical_m, logical_n, logical_k),
                            symbol="mx_dense_projection",
                            prepacked_weight_symbols=(
                                "prepacked_weight", "prepacked_weight_scales")
                            if args.prepacked_weight else None,
                            cache_immutable_weight=args.cached_weight))
        (out / "mx_dense.c").write_text(dense_kernel.source)
        driver = _dense_driver(site, m=m, n=n, logical_m=logical_m,
                               logical_n=logical_n, time_dense=args.time_dense,
                               repeats=args.dense_repeats)
    else:
        driver = (_phase_driver if args.phase_cycles else _driver)(
            site, m=m, n=n, k=k, logical_m=logical_m,
            logical_n=logical_n, logical_k=logical_k)
    (out / "mx_driver.c").write_text(driver)
    bench = software / "riscv-tests/benchmarks/common"
    cc = riscv / "bin/riscv64-unknown-elf-gcc"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-fno-fast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc", "-I", str(ROOT / "mx_gemmini_support"),
             "-I", str(software / "riscv-tests"), "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [out / name for name in ("mx_issue.c", "mx_driver.c", "mx_data.S")]
    if args.dense_adapter:
        sources.append(out / "mx_dense.c")
    sources += [ROOT / "mx_gemmini_support/runtime_quant.c"]
    sources += sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))
    objects = []
    for index, source in enumerate(sources):
        obj = out / f"mx_{index}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(obj)],
             out=out, log=f"compile_{index}.log")
        objects.append(obj)
    candidate = None
    if args.dense_adapter:
        candidate = out / "mx_dense_catalog_candidate.o"
        ld = riscv / "bin/riscv64-unknown-elf-ld"
        nm = riscv / "bin/riscv64-unknown-elf-nm"
        _run([str(ld), "-r", "-o", str(candidate),
              *(str(objects[index]) for index in
                ((0, 2, 3, 4) if args.prepacked_weight else (0, 3, 4)))],
             out=out, log="combine_dense.log")
        symbols = subprocess.run([str(nm), "--format=posix", "--extern-only",
                                  str(candidate)], capture_output=True, text=True, check=True)
        rows = [line.split() for line in symbols.stdout.splitlines()]
        if (sum(len(row) >= 2 and row[0] == "mx_dense_projection" and
                row[1].upper() == "T" for row in rows) != 1 or
                any(len(row) < 2 or row[1].upper() == "U" for row in rows)):
            raise ValueError("dense MX relocatable object has unresolved or missing symbols")
    elf = out / "live_f32.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T", str(bench / "test.ld"),
          *(str(obj) for obj in objects), "-lm", "-lgcc", "-o", str(elf)],
         out=out, log="link.log")
    so = out / "libgemmini.so"
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    _run(["g++", "-L", str(riscv / "lib"), f"-Wl,-rpath,{riscv / 'lib'}",
          "-shared", "-o", str(so), "-std=c++17", "-I", str(riscv / "include"),
          "-fPIC", "-O3", *(str(source) for source in extension_sources)],
         out=out, log="extension_build.log")
    environment = None
    spike_args = [str(riscv / "bin/spike")]
    if args.phase_cycles or args.time_dense:
        import os
        environment = os.environ.copy()
        environment.update(GEMMINI_MODE="both", GEMMINI_PERF_CONFIG="e4m3_vpu",
                           GEMMINI_PERF_SET="mem.host_tracking=0",
                           GEMMINI_PERF_OUT=str(out / "gemmini_perf.txt"))
        spike_args.append("--isa=rv64gc_zicntr")
    run = subprocess.run([*spike_args, f"--extlib={so}",
                          "--extension=gemmini", str(elf)], cwd=out, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env=environment)
    (out / "spike.log").write_text(run.stdout)
    expected = (f"dense f32 MX {site}: 0 output mismatches" if args.dense_adapter else
                f"live f32 MX {site}: 0 pack mismatches, 0 BF16/f32 mismatches")
    passed = run.returncode == 0 and expected in run.stdout
    receipt = {
        "schema": ("mx_gemmini.dense_f32_projection_spike.v1" if args.dense_adapter else
                   "mx_gemmini.runtime_f32_projection_spike.v1"),
        "status": ("dense_three_pointer_fp4_projection_matched" if args.dense_adapter else
                   "live_f32_to_fp4_projection_matched") if passed else "failed",
        "scope": "one model2MLIR small-LLaMA MLP projection, runtime f32 padding and MX quantization",
        "site": site, "logical_shape_mnk": [logical_m, logical_n, logical_k],
        "padded_shape_mnk": [m, n, k],
        "compared_bf16_outputs": (logical_m * logical_n * args.dense_repeats
                                  if args.dense_adapter else m * n),
        "profile_sha256": issuer.profile_sha256,
        "payload_manifest_sha256": qualified_issuer.payload_manifest_sha256,
        "issuer_sha256": _sha(out / "mx_issue.c"),
        "quantizer_source_sha256": _sha(ROOT / "mx_gemmini_support/runtime_quant.c"),
        "elf_sha256": _sha(elf), "spike_log_sha256": _sha(out / "spike.log"),
        "spike_exit_code": run.returncode,
    }
    if dense_kernel is not None:
        receipt["dense_adapter_sha256"] = _sha(out / "mx_dense.c")
        receipt["static_workspace_bytes"] = dense_kernel.workspace_bytes
        if args.cached_weight:
            receipt["cached_weight"] = {
                "scope": "archived fixed RHS; one serial cache per dense entry",
                "persistent_bytes": dense_kernel.persistent_bytes,
                "source_weight_sha256": _sha(out / "raw_b.bin"),
            }
        receipt["abi"] = {"symbol": dense_kernel.symbol,
                          "argument_order": ["lhs", "rhs", "out"],
                          "dtypes": ["f32", "f32", "f32"]}
        if args.prepacked_weight:
            receipt["prepacked_weight"] = {
                "scope": "archived fixed RHS; no runtime weight conversion",
                "codes_sha256": _sha(out / "prepacked_weight.bin"),
                "scales_sha256": _sha(out / "prepacked_weight_scales.bin"),
                "source_weight_sha256": _sha(out / "raw_b.bin"),
            }
        receipt["relocatable_object_sha256"] = _sha(candidate)
        receipt["schedule_origin"] = (
            "merlin_mk_native_dram_loop" if native is not None else
            "compact_loop_profile_and_source_plan_checked" if compact_issuer is not None else
            "shape_planned_source_parity_checked" if generated is not None else
            "payload_bound_source_qualified")
        if native is not None:
            receipt["schedule_digest"] = native.digest
            receipt["instruction_set_digest"] = native.instruction_set_digest
            receipt["native_chunks"] = list(native.chunks)
            (out / "schedule.mk").write_text(native.text)
    if args.phase_cycles:
        match = re.search(
            r"^PROFILE_CYCLES padding=(\d+) activation_pack=(\d+) "
            r"weight_pack=(\d+) issue=(\d+) widen=(\d+)$",
            run.stdout, re.MULTILINE)
        if not match:
            raise RuntimeError("Spike did not report phase cycles")
        receipt["phase_cycles"] = dict(zip(
            ("padding", "activation_pack", "weight_pack", "issue", "widen"),
            (int(value) for value in match.groups()), strict=True))
        receipt["phase_cycle_scope"] = "one archived model-derived FP4 projection; rdcycle on e4m3_vpu"
        receipt["schedule_origin"] = (
            "merlin_mk_native_dram_loop" if native else
            "compact_loop_profile_and_source_plan_checked" if compact_issuer else
            "shape_planned_source_parity_checked" if generated else
            "payload_bound_source_qualified")
    if args.time_dense:
        pairs = [(int(index), int(cycles)) for index, cycles in re.findall(
            r"^DENSE_CYCLES (\d+) (\d+)$", run.stdout, re.MULTILINE)]
        if [index for index, _ in pairs] != list(range(args.dense_repeats)):
            raise RuntimeError("Spike did not report dense adapter cycles")
        receipt["dense_cycles"] = pairs[0][1]
        if args.dense_repeats != 1:
            receipt["dense_cycles_by_call"] = [cycles for _, cycles in pairs]
        receipt["dense_cycle_scope"] = "one archived model-derived FP4 projection; rdcycle on e4m3_vpu"
    (out / "runtime_f32_projection_spike.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    if not passed:
        raise RuntimeError(f"live f32 projection mismatch; see {out / 'spike.log'}")


if __name__ == "__main__":
    main()
