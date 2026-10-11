"""Compile a source-bound dense MX+VPU attention shape and check it on Spike."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.nicolas_attention_io import ATTENTION_BUFFERS
from mx_gemmini_support.nicolas_attention_program import render_dense_attention_program
from mx_gemmini_support.nicolas_attention_shape import DenseAttentionShape
from mx_gemmini_support.nicolas_attention_vector import render_dense_attention_vector
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _source_closure


ROOT = Path(__file__).resolve().parents[1]
BASELINES = {
    "base": ("nicolas_attn_vpu_source_baseline_e9ded36_0344bcb",
             "attn_vpu.h.gz", "attn_vpu_expect.h", "npu_source_sha256"),
    "mid": ("nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb",
            "attn_vpu_mid.h.gz", "attn_vpu_mid_expect.h", "source_core_sha256"),
    "fa": ("nicolas_attn_vpu_fa_source_baseline_a042643_0344bcb",
           "attn_vpu_fa.h.gz", "attn_vpu_fa_expect.h", "source_core_sha256"),
}
EXPECTED_SHAPES = {"base": (32, 64, 64), "mid": (64, 128, 64),
                   "fa": (64, 256, 128)}
CONTRACT = ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, cwd: Path, log: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def replace_source_kernel(source: str, *, variant: str) -> str:
    """Retain Nicolas's data/checker but call one compiler-issued MX object."""
    if variant not in BASELINES:
        raise ValueError("dense attention source variant is unsupported")
    prototype = ("void mx_issue(" +
                 ", ".join("const void *" + name for name in ATTENTION_BUFFERS) +
                 ");\n")
    first = source.index("static void mvout_rows")
    last = source.index("\nint *__errno", first)
    source = source[:first] + prototype + source[last:]
    first = source.index("static void attention(int check) {")
    last = source.index("\nstatic void rel_errs", first)
    call = ", ".join(ATTENTION_BUFFERS)
    body = ("static void attention(int check) {\n"
            "  (void)check;\n"
            "  mark(0);\n"
            f"  mx_issue({call});\n"
            "  gemmini_fence();\n"
            "  st[6] = read_cycles();\n"
            "}\n")
    source = source[:first] + body + source[last:]
    scale_calls = (
        "  gemmini_mx_load_scales((uint64_t)Q_SCALES, sizeof(Q_SCALES), 0);                    // act half 0\n"
        "  gemmini_mx_load_scales((uint64_t)KT_SCALES, sizeof(KT_SCALES), 1);                  // wgt half 0\n"
        "  gemmini_mx_load_scales_2d((uint64_t)V_SCALES, D, SK / 32, D, 4096, 1);              // wgt half 1")
    if source.count(scale_calls) != 1:
        raise ValueError("dense attention source scale call sequence differs")
    source = source.replace(scale_calls, "")
    if any(token in source for token in ("gemmini_extended_mvin(",
                                         "gemmini_extended_mvout(",
                                         "gemmini_vpu_", "gemmini_spad_requant(",
                                         "gemmini_loop_ws_spad(",
                                         "gemmini_mx_load_scales(",
                                         "gemmini_mx_load_scales_2d(",
                                         "gemmini_mxquant_config_mvout(")):
        raise ValueError("source checker retained a handwritten kernel command")
    if variant in ("mid", "fa"):
        source = "#define ATTN_HOST_SOFTMAX 0\n#define ATTN_ACC_ROWS 16\n" + source
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(BASELINES), required=True)
    for name in ("npu-root", "profile", "rtl-root", "riscv-root",
                 "spike-extension", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("npu_root", "profile", "rtl_root", "riscv_root",
                 "spike_extension", "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    evidence_name, header_name, expect_name, source_key = BASELINES[args.variant]
    evidence = ROOT / "docs/evidence" / evidence_name
    baseline = json.loads((evidence / "receipt.json").read_text())
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    source_path = args.npu_root / "baremetal/mxgemmini/src/attn_vpu.c"
    source = source_path.read_text()
    header = gzip.decompress((evidence / header_name).read_bytes())
    frontend = (evidence / "attention.model2mlir.mlir").read_text()
    if (_sha(source_path) != baseline[source_key] or
            hashlib.sha256(header).hexdigest() !=
            baseline.get("source_header_sha256", baseline.get("generated_header_sha256")) or
            _sha(evidence / "attention.model2mlir.mlir") !=
            baseline["frontend_mlir_sha256"] or
            profile_sha256(profile) != baseline["profile_sha256"]):
        raise ValueError("pinned source, frontend, header, or profile differs")
    shape = DenseAttentionShape.from_header(header.decode())
    if (shape.sq, shape.sk, shape.d) != EXPECTED_SHAPES[args.variant]:
        raise ValueError("dense attention source variant shape differs")
    if args.variant in ("mid", "fa"):
        wrapper = args.npu_root / f"baremetal/mxgemmini/src/attn_vpu_{args.variant}.c"
        if _sha(wrapper) != baseline["source_wrapper_sha256"]:
            raise ValueError("dense attention source wrapper differs")
    vector, binding = render_dense_attention_vector(
        frontend, source, header.decode(), profile, CONTRACT.read_bytes())
    program = render_dense_attention_program(vector, profile, binding)
    out = args.out_dir
    out.mkdir(parents=True)
    (out / "attention.vector_bound.mlir").write_text(vector)
    (out / "vector_binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n")
    (out / "attention.program_bound.mlir").write_text(program)
    object_dir = out / "object"
    _run([sys.executable, "-m", "tools.compile_object",
          "--mlir", str(out / "attention.program_bound.mlir"),
          "--precursor-mlir", str(out / "attention.vector_bound.mlir"),
          "--binding-json", str(out / "vector_binding.json"),
          "--profile", str(args.profile), "--rtl-root", str(args.rtl_root),
          "--riscv-root", str(args.riscv_root), "--mx-opt", str(args.mx_opt),
          "--out-dir", str(object_dir)], cwd=ROOT, log=out / "object_compile.log")
    obj_manifest = json.loads((object_dir / "object_manifest.json").read_text())
    if (obj_manifest["allocated_data_section_bytes"] or
            obj_manifest["embedded_operand_bytes"] or
            obj_manifest["embedded_golden_bytes"] or
            len(obj_manifest["buffer_abi"]) != len(ATTENTION_BUFFERS)):
        raise ValueError("dense attention object contains data or ABI differs")
    npu_src = args.npu_root / "baremetal/mxgemmini"
    software = args.rtl_root / "software/gemmini-rocc-tests"
    bench = software / "riscv-tests/benchmarks/common"
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    driver = out / "attn_vpu_typed_program.c"
    driver.write_text(replace_source_kernel(source, variant=args.variant))
    (out / "attn_vpu.h").write_bytes(header)
    shutil.copyfile(evidence / expect_name, out / "attn_vpu_expect.h")
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             "-I", str(npu_src), "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"), "-I", str(software),
             "-I", str(bench)]
    sources = [driver, *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S"))]
    objects = [object_dir / "mx_issue.o"]
    for index, path in enumerate(sources):
        obj = out / f"source_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(obj)],
             cwd=out, log=out / f"source_{index}.log")
        objects.append(obj)
    elf = out / "attn_vpu_typed_program.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(obj) for obj in objects),
          "-lm", "-lgcc", "-o", str(elf)], cwd=out, log=out / "link.log")
    logs = []
    for run in (1, 2):
        log = out / f"spike_{run}.log"
        _run([str(spike), f"--extlib={args.spike_extension}",
              "--extension=gemmini", str(elf)], cwd=out, log=log)
        logs.append(log.read_text())
    hashes = baseline["source_stage_hashes"]
    expected = "check hashes: " + "  ".join(
        f"{key} {hashes[key]}" for key in ("S", "P", "Pq", "O"))
    if (logs[0] != logs[1] or expected not in logs[0] or
            f"check S vs mesh golden: 0/{shape.sq * shape.sk // 4} 4-element words differ"
            not in logs[0] or
            "check hashes vs Spike: match (0 of 4 differ)" not in logs[0] or
            "attn_vpu PASSED" not in logs[0]):
        raise ValueError("compiler-issued dense attention differs from source")
    receipt = {
        "schema": "mx_gemmini.nicolas_dense_attention_shape_qualification.v1",
        "status": "typed_attention_object_matches_source_on_spike",
        "variant": args.variant, "shape_sq_sk_d": [shape.sq, shape.sk, shape.d],
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "npu_revision": _git_revision(args.npu_root),
        "rtl_revision": _git_revision(args.rtl_root),
        "model2mlir_revision": baseline["model2mlir_revision"],
        "source_baseline_sha256": _sha(evidence / "receipt.json"),
        "source_sha256": _sha(source_path),
        "frontend_mlir_sha256": _sha(evidence / "attention.model2mlir.mlir"),
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "profile_sha256": profile_sha256(profile),
        "typed_mlir_sha256": _sha(out / "attention.program_bound.mlir"),
        "vector_mlir_sha256": _sha(out / "attention.vector_bound.mlir"),
        "binding_sha256": _sha(out / "vector_binding.json"),
        "object_sha256": _sha(object_dir / "mx_issue.o"),
        "object_manifest_sha256": _sha(object_dir / "object_manifest.json"),
        "compile_manifest_sha256": _sha(object_dir / "compile_manifest.json"),
        "driver_sha256": _sha(driver), "elf_sha256": _sha(elf),
        "spike_log_sha256": _sha(out / "spike_1.log"),
        "spike_reproduced_runs": 2,
        "stage_hashes": hashes,
        "riscv_gcc_sha256": _sha(cc), "spike_sha256": _sha(spike),
        "spike_extension_sha256": _sha(args.spike_extension),
        "native_verifier_sha256": _sha(args.mx_opt),
        "command_count": obj_manifest["command_count"],
        "fence_count": obj_manifest["fence_count"],
        "allocated_data_section_bytes": obj_manifest["allocated_data_section_bytes"],
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"typed {args.variant} attention matched all source hashes on Spike")


if __name__ == "__main__":
    main()
