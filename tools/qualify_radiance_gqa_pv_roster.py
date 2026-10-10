"""Run all 16 source Muon GQA PV operands through one compiled MX object.

An isolated Cyclotron model correction implements `ex_accumulate=0` overwrite,
as Nicolas's Spike does. It executes the original mixed Muon/MX kernel with
diagnostic stores that snapshot every P operand and PV output. Nicolas's
checked-in numerical model verifies each PV tile independently. The compiler's
generic 64^3 MX object then consumes those P/V bytes on Rocket/RoCC Spike.
"""

from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.source_attention_qk import (
    ACCUMULATOR_PRECISION, LATEST_SOURCE_REVISION, LOW_LEVEL_MODEL_SHA256,
    PRODUCT_PRECISION, SUBMODULE_REVISION, decode_e4m3)
from mx_gemmini_support.source_fp6 import _array, _bytes
from tools.compile_mx import _require_gitlink
from tools.qualify_radiance_gqa_cyclotron_pv import (
    CYCLOTRON_CONFIG_SHA256, CYCLOTRON_REVISION, PV_EVIDENCE, QK_EVIDENCE,
    SOURCE_HEADER_SHA256, SOURCE_KERNEL_SHA256, _compile, _revision,
    _scales, _sha, _simulate, _stage, _untile_p)


ROOT = Path(__file__).resolve().parents[1]
FIRST_CYCLOTRON = ROOT / "docs/evidence/radiance_gqa_cyclotron_pv_80f84ca"
EVIDENCE = ROOT / "docs/evidence/radiance_gqa_pv_roster_80f84ca"
OBJECT_DIR = ROOT / "docs/evidence/radiance_gqa_runtime_object_80f84ca/head0_object"
CYCLOTRON_OVERWRITE_SOURCE_SHA256 = "49b42427082f4bca7f0e30ceca8631563cf24376009cea2e458351fbac46cffd"
CORRECTED_BASELINE_O_SHA256 = "19693b4e65fe00a30d4137c58490c0adf9bd06e839ed4c8f698f6c8854d607f5"
P_ANCHOR = """            mu_barrier(2, wpb);

            // pack P scales"""
P_PROBE = """            mu_barrier(2, wpb);
            // Qualification only: capture every tiled P operand and E8M0 scale block.
            {
                const uint32_t tile = h * FA_NBLK_USED + j;
                const volatile __shared uint32_t *p =
                    reinterpret_cast<const volatile __shared uint32_t *>(P_BYTE[cur]);
                volatile uint32_t *p_out = reinterpret_cast<volatile uint32_t *>(P_GMEM)
                    + tile * (FA_SQ * FA_BK / 4);
                for (uint32_t i = tid; i < FA_SQ * FA_BK / 4; i += thr) p_out[i] = p[i];
                const volatile __shared uint32_t *scales =
                    reinterpret_cast<const volatile __shared uint32_t *>(SCALE_SMEM);
                volatile uint32_t *s_out = reinterpret_cast<volatile uint32_t *>(PS_GMEM)
                    + tile * (FA_GKB * FA_SQ);
                for (uint32_t i = tid; i < FA_GKB * FA_SQ; i += thr) s_out[i] = scales[i];
                mu_barrier(7, wpb);
            }

            // pack P scales"""
PV_ANCHOR = """            mu_barrier(4, wpb);

            // O_acc ="""
PV_PROBE = """            mu_barrier(4, wpb);
            // Qualification only: capture every BF16 MX PV tile before accumulation.
            {
                const uint32_t tile = h * FA_NBLK_USED + j;
                const volatile __shared uint32_t *pv =
                    reinterpret_cast<const volatile __shared uint32_t *>(PVOUT_SMEM);
                volatile uint32_t *pv_out = reinterpret_cast<volatile uint32_t *>(0x40060000)
                    + tile * (FA_SQ * FA_D / 2);
                for (uint32_t i = tid; i < FA_SQ * FA_D / 2; i += thr) pv_out[i] = pv[i];
                mu_barrier(7, wpb);
            }

            // O_acc ="""


def _instrument(source: str) -> str:
    for anchor, probe in ((P_ANCHOR, P_PROBE), (PV_ANCHOR, PV_PROBE)):
        if source.count(anchor) != 1:
            raise ValueError("GQA source no longer has the pinned PV probe point")
        source = source.replace(anchor, probe)
    return source


def _run(command: list[str], cwd: Path, log: Path) -> None:
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _source_v(header: str, kv_block: int) -> tuple[bytes, bytes]:
    v = _array(header, name="V_in", ctype="uint8_t",
               dimensions="[FA_NKV*FA_NBLK_USED*FA_BK][FA_D]",
               count=2 * 2 * 64 * 64, maximum=255)
    scales = _array(header, name="V_scales", ctype="uint8_t",
                    dimensions="[FA_NKV*FA_NBLK_USED*FA_GKB][FA_D]",
                    count=2 * 2 * 2 * 64, maximum=255)
    return (bytes(v[kv_block * 4096:(kv_block + 1) * 4096]),
            bytes(scales[kv_block * 128:(kv_block + 1) * 128]))


def _model_golden(p: bytes, ps: bytes, v: bytes, vs: bytes, *, torch, model) -> bytes:
    import math

    a = torch.tensor([decode_e4m3(code) for code in p],
                     dtype=torch.float32).reshape(64, 64)
    b = torch.tensor([decode_e4m3(code) for code in v],
                     dtype=torch.float32).reshape(64, 64)
    a_scales = torch.tensor([math.ldexp(1.0, code - 127) for code in ps],
                            dtype=torch.float32).reshape(2, 64).t()
    b_scales = torch.tensor([math.ldexp(1.0, code - 127) for code in vs],
                            dtype=torch.float32).reshape(2, 64)
    result = model.tiled_matmul_hwlike(
        a, b, a_scales, b_scales, verbose=False,
        prod_precision_list=PRODUCT_PRECISION,
        acc_precision_list=ACCUMULATOR_PRECISION)
    if not torch.isfinite(result).all():
        raise ValueError("source PV operand overflows Nicolas's reduced-precision model")
    codes, bits = model.tensor_to_custom_fp_codes(result, "bf16")
    if bits != 16:
        raise ValueError("PV numerical model changed BF16 output width")
    return _bytes(tuple(code for row in codes for code in row), 2)


def _spike(args: argparse.Namespace, cases: list[dict[str, bytes]]) -> tuple[Path, Path, Path]:
    object_manifest = json.loads((OBJECT_DIR / "object_manifest.json").read_text())
    obj = OBJECT_DIR / "mx_issue.o"
    pv_index = json.loads((PV_EVIDENCE / "index.json").read_text())
    if (object_manifest.get("status") != "rv64_rocc_issuer_object_built" or
            object_manifest.get("object_sha256") != _sha(obj.read_bytes()) or
            object_manifest.get("object_sha256") != pv_index["object_sha256"] or
            object_manifest.get("allocated_data_section_bytes") != 0 or
            object_manifest.get("shape_mnk") != [64, 64, 64]):
        raise ValueError("the compiler's source-derived 64^3 runtime object changed")
    if _revision(args.rtl_root) != pv_index["rtl_revision"]:
        raise ValueError("Nicolas MX RTL revision differs from the pinned Spike build")
    _require_gitlink(args.rtl_root, "software/gemmini-rocc-tests")
    _require_gitlink(args.rtl_root, "software/libgemmini")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    extension = args.rtl_root / "software/libgemmini"
    bench = software / "riscv-tests/benchmarks/common"
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if not (cc.is_file() and spike.is_file() and (bench / "test.ld").is_file()):
        raise ValueError("pinned Nicolas Spike or RISC-V benchmark runtime is incomplete")
    work = args.out_dir / "spike"
    work.mkdir()
    shutil.copyfile(OBJECT_DIR / "mx_issue.h", work / "mx_issue.h")
    data = work / "data"
    data.mkdir()
    names = ("activation", "activation_scales", "weight", "weight_scales", "golden_bf16")
    assembly = [".section .rodata", ""]
    for i, case in enumerate(cases):
        for name in names:
            label = f"tile{i}_{name}"
            (data / f"{label}.bin").write_bytes(case[name])
            assembly += [".p2align 6", f".globl {label}", f"{label}:",
                         f'.incbin "data/{label}.bin"', ""]
    (work / "mx_runtime_data.S").write_text("\n".join(assembly))
    externs = "".join(f"extern const uint8_t tile{i}_{name}[];\n"
                      for i in range(16) for name in names)
    pointer_arrays = "".join(
        f"static const uint8_t *const {name}[16] = {{" +
        ", ".join(f"tile{i}_{name}" for i in range(16)) + "};\n"
        for name in names)
    (work / "mx_runtime_driver.c").write_text(f'''#include <stdint.h>
#include <stdio.h>
#include "mx_issue.h"
{externs}
{pointer_arrays}
static uint8_t output[8192] __attribute__((aligned(64)));
static uint8_t scratch[2048] __attribute__((aligned(64)));

int main(void) {{
  int errors = 0;
  for (int tile = 0; tile < 16; ++tile) {{
    mx_issue(activation[tile], activation_scales[tile], output,
             scratch, weight[tile], weight_scales[tile]);
    const uint16_t *actual = (const uint16_t *)output;
    const uint16_t *expected = (const uint16_t *)golden_bf16[tile];
    for (int i = 0; i < 4096; ++i) {{
      if (actual[i] != expected[i]) {{
        if (errors < 8)
          printf("tile %d mismatch %d: got=0x%04x expected=0x%04x\\n",
                 tile, i, actual[i], expected[i]);
        ++errors;
      }}
    }}
  }}
  printf("runtime MX PV roster: %d/65536 BF16 mismatches\\n", errors);
  return errors != 0;
}}
''')
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             f"-ffile-prefix-map={work}=.",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [work / "mx_runtime_driver.c", work / "mx_runtime_data.S",
               *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S"))]
    objects = [obj]
    for i, source in enumerate(sources):
        compiled = work / f"runtime_{i}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(compiled)],
             work, work / f"compile_{i}.log")
        objects.append(compiled)
    elf = work / "mx_runtime_pv_roster.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(path) for path in objects),
          "-lm", "-lgcc", "-o", str(elf)], work, work / "link.log")
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc",
                         *sorted((extension / "perf").rglob("*.cc"))]
    so = work / "libgemmini.so"
    _run(["g++", "-L", str(args.riscv_root / "lib"),
          f"-Wl,-rpath,{args.riscv_root / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(args.riscv_root / "include"), "-fPIC", "-O3",
          *(str(path) for path in extension_sources)], work,
         work / "extension_build.log")
    spike_log = work / "spike.log"
    _run([str(spike), f"--extlib={so}", "--extension=gemmini", str(elf)],
         work, spike_log)
    if b"runtime MX PV roster: 0/65536 BF16 mismatches" not in spike_log.read_bytes():
        raise ValueError(f"Nicolas Spike PV roster did not match: {spike_log}")
    return elf, so, spike_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "radiance-lib-root", "cyclotron-root", "llvm-muon",
                 "riscv-root", "rtl-root", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--baseline-index", type=Path)
    args = parser.parse_args()
    for name in ("source_root", "radiance_lib_root", "cyclotron_root", "llvm_muon",
                 "riscv_root", "rtl_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    first_index = json.loads((FIRST_CYCLOTRON / "index.json").read_text())
    correction = EVIDENCE / "cyclotron_overwrite.patch"
    correction_bytes = subprocess.check_output(
        ["git", "diff", "-U0", "--", "src/muon/mxgemmini/mod.rs"],
        cwd=args.cyclotron_root)
    if (_revision(args.source_root) != LATEST_SOURCE_REVISION or
            _revision(args.source_root / "lib/mxgemmini") != SUBMODULE_REVISION or
            _revision(args.cyclotron_root) != CYCLOTRON_REVISION or
            _sha((args.cyclotron_root / "config.toml").read_bytes()) !=
            CYCLOTRON_CONFIG_SHA256 or
            _sha((args.cyclotron_root / "src/muon/mxgemmini/mod.rs").read_bytes()) !=
            CYCLOTRON_OVERWRITE_SOURCE_SHA256 or
            correction_bytes != correction.read_bytes() or
            _sha((args.radiance_lib_root / "libmuonrt.a").read_bytes()) !=
            first_index["muon_runtime_archive_sha256"]):
        raise ValueError("source or corrected Cyclotron model differs from pinned PV roster")
    corrected_o = gzip.decompress((EVIDENCE / "corrected_baseline_o.bin.gz").read_bytes())
    if _sha(corrected_o) != CORRECTED_BASELINE_O_SHA256:
        raise ValueError("unmodified source output on corrected Cyclotron changed")
    original = (args.source_root / "kernels/flash_attention_mx_gqa/kernel.cpp").read_bytes()
    header = gzip.decompress((QK_EVIDENCE / "fa_data.h.gz").read_bytes())
    if _sha(original) != SOURCE_KERNEL_SHA256 or _sha(header) != SOURCE_HEADER_SHA256:
        raise ValueError("source kernel or generated data differs from pinned GQA fixture")
    args.out_dir.mkdir(parents=True)
    probe = _instrument(original.decode())
    (args.out_dir / "probe.patch").write_text("".join(difflib.unified_diff(
        original.decode().splitlines(keepends=True), probe.splitlines(keepends=True),
        fromfile="kernel.cpp", tofile="kernel.cpp.all_pv_probe", n=0)))
    kernel_dir = _stage(args.source_root, args.out_dir / "probe", probe, header)
    elf = _compile(kernel_dir, args, args.out_dir / "build_probe.log")
    p_raw = _simulate(args, elf, 0x40010000, 65536,
                      args.out_dir / "p_tiles.bin", args.out_dir / "p_tiles.log")
    ps_raw = _simulate(args, elf, 0x40020000, 8192,
                       args.out_dir / "p_scales.bin", args.out_dir / "p_scales.log")
    pv_raw = _simulate(args, elf, 0x40060000, 131072,
                       args.out_dir / "pv_tiles.bin", args.out_dir / "pv_tiles.log")
    final_o = _simulate(args, elf, 0x40040000, 65536,
                        args.out_dir / "probe_o.bin", args.out_dir / "probe_o.log")
    if final_o != corrected_o:
        raise ValueError("all-tile probe changed the source attention output")
    header_text = header.decode("ascii")
    source_o_codes = _array(
        header_text, name="O_gold", ctype="uint16_t",
        dimensions="[FA_NQ*FA_SQ][FA_D]", count=8 * 64 * 64, maximum=65535)
    source_o = _bytes(source_o_codes, 2)
    source_o_mismatches = sum(final_o[i:i + 2] != source_o[i:i + 2]
                              for i in range(0, len(final_o), 2))
    if source_o_mismatches != 31852:
        raise ValueError("source O_gold diagnostic changed for the pinned GQA fixture")
    import sys
    source_model = args.source_root / "lib/mxgemmini/fp8_matmul_model.py"
    if _sha(source_model.read_bytes()) != LOW_LEVEL_MODEL_SHA256:
        raise ValueError("Nicolas reduced-precision numerical model changed")
    sys.path.insert(0, str(source_model.parent))
    import torch
    import fp8_matmul_model as model
    if Path(model.__file__).resolve() != source_model.resolve():
        raise ValueError("PV numerical model imported from another checkout")
    cases = []
    case_index = []
    for tile in range(16):
        h, j = divmod(tile, 2)
        kv_block = (h // 4) * 2 + j
        p = _untile_p(p_raw[tile * 4096:(tile + 1) * 4096])
        ps = _scales(ps_raw[tile * 512:(tile + 1) * 512])
        v, vs = _source_v(header_text, kv_block)
        pv = pv_raw[tile * 8192:(tile + 1) * 8192]
        golden = _model_golden(p, ps, v, vs, torch=torch, model=model)
        cyclotron_mismatches = sum(pv[i:i + 2] != golden[i:i + 2]
                                   for i in range(0, 8192, 2))
        if cyclotron_mismatches:
            raise ValueError(f"corrected Cyclotron PV tile {tile} differs from Nicolas model")
        cases.append({"activation": p, "activation_scales": ps,
                      "weight": v, "weight_scales": vs, "golden_bf16": golden})
        case_index.append({"head": h, "block": j, "kv_block": kv_block,
                           "cyclotron_pv_bf16_sha256": _sha(pv),
                           "cyclotron_pv_mismatches": cyclotron_mismatches,
                           **{f"{name}_sha256": _sha(value)
                              for name, value in cases[-1].items()}})
    pv_index = json.loads((PV_EVIDENCE / "index.json").read_text())
    if (case_index[0]["activation_sha256"] != pv_index["activation_sha256"] or
            case_index[0]["activation_scales_sha256"] != pv_index["activation_scales_sha256"] or
            case_index[0]["golden_bf16_sha256"] != pv_index["golden_bf16_sha256"] or
            case_index[0]["cyclotron_pv_mismatches"] != 0):
        raise ValueError("first tile no longer matches the independently checked PV proxy")
    spike_elf, extension, spike_log = _spike(args, cases)
    receipt = {
        "schema": "mx_gemmini.gqa_pv_runtime_roster_spike.v1",
        "status": "all_source_muon_pv_operands_match_compiled_object_on_spike",
        "scope": "16 source-produced Muon P/PV tiles on isolated corrected Cyclotron; Nicolas numerical and Spike parity; no RTL SFU claim",
        "source_revision": LATEST_SOURCE_REVISION,
        "cyclotron_revision": CYCLOTRON_REVISION,
        "cyclotron_model_source_sha256": CYCLOTRON_OVERWRITE_SOURCE_SHA256,
        "cyclotron_overwrite_patch_sha256": _sha(correction.read_bytes()),
        "cyclotron_binary_sha256": _sha((args.cyclotron_root / "target/release/cyclotron").read_bytes()),
        "muon_runtime_archive_sha256": _sha((args.radiance_lib_root / "libmuonrt.a").read_bytes()),
        "unmodified_corrected_o_sha256": _sha(corrected_o),
        "rtl_revision": _revision(args.rtl_root),
        "source_header_sha256": _sha(header),
        "source_kernel_sha256": _sha(original),
        "numerical_model_sha256": _sha(source_model.read_bytes()),
        "probe_patch_sha256": _sha((args.out_dir / "probe.patch").read_bytes()),
        "probe_elf_sha256": _sha(elf.read_bytes()),
        "p_tiles_sha256": _sha(p_raw), "p_scales_sha256": _sha(ps_raw),
        "pv_tiles_sha256": _sha(pv_raw), "final_o_sha256": _sha(final_o),
        "source_o_gold_sha256": _sha(source_o),
        "source_o_gold_bf16_mismatches": source_o_mismatches,
        "object_sha256": _sha((OBJECT_DIR / "mx_issue.o").read_bytes()),
        "object_manifest_sha256": _sha((OBJECT_DIR / "object_manifest.json").read_bytes()),
        "first_cyclotron_index_sha256": _sha((FIRST_CYCLOTRON / "index.json").read_bytes()),
        "spike_elf_sha256": _sha(spike_elf.read_bytes()),
        "spike_extension_sha256": _sha(extension.read_bytes()),
        "spike_log_sha256": _sha(spike_log.read_bytes()),
        "spike_binary_sha256": _sha((args.riscv_root / "bin/spike").read_bytes()),
        "riscv_gcc_sha256": _sha((args.riscv_root / "bin/riscv64-unknown-elf-gcc").read_bytes()),
        "spike_exit_code": 0,
        "compared_bf16_outputs": 16 * 4096,
        "cases": case_index,
    }
    if args.baseline_index and receipt != json.loads(args.baseline_index.read_text()):
        raise ValueError("all-tile Cyclotron-to-Spike result differs from archived baseline")
    (args.out_dir / "index.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("all 16 Muon-produced PV operands match one compiled MX object on Nicolas Spike: 0/65536 BF16 mismatches")


if __name__ == "__main__":
    main()
