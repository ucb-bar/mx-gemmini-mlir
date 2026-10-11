"""Run a model-derived TinyLlama QK score through the typed MX VPU softmax."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from mx_gemmini_support.model2mlir_worklist import build_model2mlir_worklist
from mx_gemmini_support.source_payload import load_bundle
from mx_gemmini_support.source_softmax import (audit_source, render_softmax_bound,
                                               replace_source_program)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_mx import _git_revision, _run, _sha, _source_closure


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
MODEL2MLIR_SOURCE_CLOSURE = "6eb6648cf2eebd72cd481dff084b0e154e034fbe50b7bce31935091b1a237e39"
MODEL2MLIR_REVISION = "a042643e31366724ca0482389c4ee85a9f88e343"


def _score_words(qk_bf16: bytes, pre, probability):
    import numpy as np
    import torch

    qk = (np.frombuffer(qk_bf16, dtype="<u2").astype(np.uint32) << 16).view(
        np.float32).reshape(16, 32)
    if (pre.shape != (8, 8) or probability.shape != (8, 8) or
            not np.isfinite(qk).all() or not np.isfinite(probability).all()):
        raise ValueError("TinyLlama attention QK or softmax capture has invalid values")
    mask = np.triu(np.ones((8, 8), dtype=bool), 1)
    score = np.full((16, 32), -np.inf, dtype=np.float32)
    score[:8, :8] = qk[:8, :8] * np.float32(0.125)
    score[:8, :8][mask] = -np.inf
    score[8:, :] = 0
    bits = torch.from_numpy(score).to(torch.bfloat16).view(torch.int16).numpy().astype(
        "<u2").copy()
    if (not np.all(bits[:8, :8][mask] == 0xff80) or
            not np.all(bits[:8, 8:] == 0xff80) or
            not np.all(bits[8:] == 0)):
        raise ValueError("TinyLlama VPU score mask or padding changed")
    quality = {
        "score_mean_absolute_error_vs_source_f32": float(np.abs(
            score[:8, :8][~mask] - pre[~mask]).mean()),
        "source_masked_values": int(mask.sum()),
    }
    return bits, quality


def _driver(source: str, words) -> str:
    transformed = replace_source_program(source)
    initialization = (
        "  for (int r = 0; r < ROWS; r++)\n"
        "    for (int l = 0; l < VPU_LANES; l++)\n"
        "      S[r][l] = vpu_f_to_bf16(((float)(int)(rnd() % 2001) - 1000.0f) / 125.0f);"
        "   // [-8, 8]")
    if transformed.count(initialization) != 1:
        raise ValueError("Nicolas softmax input initializer changed")
    rows = ",\n".join("    {" + ", ".join(f"0x{int(word):04x}" for word in row) + "}"
                       for row in words.reshape(64, 8))
    replacement = ("  static const uint16_t model_score[ROWS][VPU_LANES] = {\n" +
                   rows + "\n  };\n  memcpy(S, model_score, sizeof(S));")
    transformed = transformed.replace(initialization, replacement, 1)
    accuracy_start = "  // accuracy vs FP32 softmax (integer-scaled print: no printf %f on bare metal)"
    status_start = '  printf("vpu_softmax %s\\n", bad ? "FAILED" : "PASSED");'
    if transformed.count(accuracy_start) != 1 or transformed.count(status_start) != 1:
        raise ValueError("Nicolas softmax checker layout changed")
    start, end = transformed.index(accuracy_start), transformed.index(status_start)
    result = (
        '  printf("model attention softmax 16x32: %d mismatches vs ref\\n", bad);\n'
        '  printf("probability_head0_bf16:");\n'
        '  for (int m = 0; m < 8; m++)\n'
        '    for (int c = 0; c < 8; c++)\n'
        '      printf(" %04x", P_hw[m * RL + c / VPU_LANES][c % VPU_LANES]);\n'
        '  printf("\\n");\n')
    return transformed[:start] + result + transformed[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full-model-mlir", "qk-bundle", "softmax-inputs", "model2mlir-root",
                 "profile", "rtl-root", "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("full_model_mlir", "qk_bundle", "softmax_inputs", "model2mlir_root",
                 "profile", "rtl_root", "riscv_root", "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    if _source_closure(args.model2mlir_root,
                       list((args.model2mlir_root / "m2m").rglob("*.py"))) != MODEL2MLIR_SOURCE_CLOSURE:
        raise ValueError("selected model2MLIR source closure differs")
    frontend_index = json.loads((ROOT / "docs/evidence/"
        "full_model_compile_preflight_a042643_748b984/index.json").read_text())
    if (frontend_index.get("model2mlir_revision") != MODEL2MLIR_REVISION or
            frontend_index.get("model2mlir_source_closure_sha256") !=
            MODEL2MLIR_SOURCE_CLOSURE):
        raise ValueError("archived model2MLIR revision and source closure differ")
    raw = args.full_model_mlir.read_bytes()
    full_graph = (gzip.decompress(raw) if args.full_model_mlir.name.endswith(".gz")
                  else raw).decode()
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    worklist = build_model2mlir_worklist(full_graph, profile)
    qk_manifest, qk_resources = load_bundle(args.qk_bundle)
    softmax_receipt = json.loads((args.softmax_inputs / "receipt.json").read_text())
    source_site = qk_manifest.get("model2mlir_attention_qk", {})
    if (qk_manifest.get("shape_mnk") != [16, 32, 64] or
            qk_manifest.get("profile_sha256") != profile_sha256(profile) or
            source_site.get("source_mlir_sha256") != worklist["source_mlir_sha256"] or
            source_site.get("region_id") != "matmul_4" or
            source_site.get("batch_index") != 0 or
            softmax_receipt.get("schema") !=
            "mx_gemmini.tinyllama_attention_softmax_capture.v1" or
            softmax_receipt.get("full_model_mlir_sha256") !=
            worklist["source_mlir_sha256"] or
            softmax_receipt.get("qk_input_sha256") != source_site.get("input_sha256") or
            softmax_receipt.get("qk_weight_sha256") != source_site.get("weight_sha256") or
            softmax_receipt.get("attention_scale") != 0.125 or
            softmax_receipt.get("mask") !=
            "strict_causal_upper_triangle_negative_infinity"):
        raise ValueError("QK, softmax, and source graph do not describe one TinyLlama head")
    import numpy as np
    import torch
    pre_path = args.softmax_inputs / "pre_softmax.npy"
    prob_path = args.softmax_inputs / "probability.npy"
    if (_sha(pre_path) != softmax_receipt["pre_softmax_npy_sha256"] or
            _sha(prob_path) != softmax_receipt["probability_npy_sha256"]):
        raise ValueError("captured model softmax arrays differ")
    pre = np.load(pre_path, allow_pickle=False)
    probability = np.load(prob_path, allow_pickle=False)
    if (hashlib.sha256(pre.tobytes()).hexdigest() !=
            softmax_receipt["pre_softmax_sha256"] or
            hashlib.sha256(probability.tobytes()).hexdigest() !=
            softmax_receipt["probability_sha256"]):
        raise ValueError("captured model softmax tensor bytes differ")
    score, quality = _score_words(qk_resources["golden_bf16"], pre, probability)
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "score_bf16.bin").write_bytes(score.tobytes())
    sys.path.insert(0, str(args.model2mlir_root))
    import m2m
    from m2m.coverage import opaque_report
    if Path(m2m.__file__).resolve().parents[1] != args.model2mlir_root:
        raise ValueError("model2MLIR imported from another source tree")

    class Softmax(torch.nn.Module):
        def forward(self, scores: torch.Tensor) -> torch.Tensor:
            return torch.softmax(scores, dim=-1)

    capture = m2m.convert(Softmax().eval(),
                          (torch.zeros((16, 32), dtype=torch.bfloat16),),
                          backend="fx_importer")
    if not capture.ok or opaque_report(capture.mlir_text):
        raise ValueError("model2MLIR did not capture a portable BF16 softmax")
    source_path = (args.rtl_root / "software/gemmini-rocc-tests/bareMetalC/"
                   "vpu_softmax.c")
    source = source_path.read_text()
    audit_source(source)
    bound, binding = render_softmax_bound(
        capture.mlir_text, source, profile, CONTRACT.read_bytes())
    verify_ir(bound, profile)
    (args.out_dir / "softmax.model2mlir.mlir").write_text(capture.mlir_text)
    mlir_path = args.out_dir / "softmax.profile_bound.mlir"
    mlir_path.write_text(bound)
    (args.out_dir / "binding_manifest.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n")
    _run([str(args.mx_opt), str(mlir_path), "-o", "/dev/null"],
         cwd=args.out_dir, log=args.out_dir / "native_verify.log")
    abi = args.out_dir / "softmax_buffer_abi.json"
    abi.write_text(json.dumps({
        "schema": "mx_gemmini.vpu_softmax_buffer_map.v1",
        "inputs": {"score": "score"}, "outputs": {"output": "output"},
    }, indent=2, sort_keys=True) + "\n")
    obj = args.out_dir / "object"
    _run([sys.executable, "-m", "tools.compile_object", "--mlir", str(mlir_path),
          "--profile", str(args.profile), "--rtl-root", str(args.rtl_root),
          "--riscv-root", str(args.riscv_root), "--abi-json", str(abi),
          "--mx-opt", str(args.mx_opt), "--out-dir", str(obj)],
         cwd=ROOT, log=args.out_dir / "object_compile.log")
    object_manifest = json.loads((obj / "object_manifest.json").read_text())
    if (object_manifest.get("allocated_data_section_bytes") != 0 or
            object_manifest.get("command_count") != 17):
        raise ValueError("typed VPU softmax object is not data-free or complete")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    extension = args.rtl_root / "software/libgemmini"
    bench = software / "riscv-tests/benchmarks/common"
    build = args.out_dir / "build"
    build.mkdir()
    driver = build / "mx_driver.c"
    driver.write_text(_driver(source, score))
    riscv_cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             f"-ffile-prefix-map={build}=.",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [driver, *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S"))]
    objects = [obj / "mx_issue.o"]
    for index, path in enumerate(sources):
        compiled = build / f"mx_{index}.o"
        _run([str(riscv_cc), *flags, "-c", str(path), "-o", str(compiled)],
             cwd=build, log=build / f"compile_{index}.log")
        objects.append(compiled)
    elf = build / "mx_program.elf"
    _run([str(riscv_cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(path) for path in objects), "-lm", "-lgcc",
          "-o", str(elf)], cwd=build, log=build / "link.log")
    so = build / "libgemmini.so"
    ext_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc",
                   *sorted((extension / "perf").rglob("*.cc"))]
    _run(["g++", "-L", str(args.riscv_root / "lib"),
          f"-Wl,-rpath,{args.riscv_root / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(args.riscv_root / "include"), "-fPIC", "-O3",
          *(str(path) for path in ext_sources)],
         cwd=build, log=build / "extension_build.log")
    spike = args.riscv_root / "bin/spike"
    result = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini", str(elf)],
                            cwd=build, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    (build / "spike.log").write_text(result.stdout)
    line = re.search(r"^probability_head0_bf16:((?: [0-9a-f]{4}){64})$",
                     result.stdout, re.MULTILINE)
    passed = (result.returncode == 0 and
              "model attention softmax 16x32: 0 mismatches vs ref" in result.stdout and
              "vpu_softmax PASSED" in result.stdout and line is not None)
    output = None
    if passed:
        output = np.asarray([int(code, 16) for code in line.group(1).split()],
                            dtype="<u2").reshape(8, 8)
        (args.out_dir / "probability_head0_bf16.bin").write_bytes(output.tobytes())
        f32 = (output.astype(np.uint32) << 16).view(np.float32)
        quality["probability_mean_absolute_error_vs_source_f32"] = float(
            np.abs(f32 - probability).mean())
        quality["probability_max_absolute_error_vs_source_f32"] = float(
            np.abs(f32 - probability).max())
        quality["probability_masked_nonzero_count"] = int(np.count_nonzero(
            f32[np.triu(np.ones((8, 8), dtype=bool), 1)]))
        quality["probability_row_sum_max_absolute_error"] = float(np.abs(
            f32.sum(axis=1) - 1).max())
        if not np.isfinite(f32).all() or quality["probability_masked_nonzero_count"]:
            raise ValueError("VPU softmax emitted nonfinite or unmasked causal output")
    receipt = {
        "schema": "mx_gemmini.tinyllama_attention_softmax_spike.v1",
        "status": ("model_qk_to_vpu_softmax_matched_on_pinned_spike" if passed else
                   "model_qk_to_vpu_softmax_failed_on_pinned_spike"),
        "scope": "model-derived QK BF16 scores are host scaled and causally masked, then compiled VPU softmax runs on Spike; PV remains separate",
        "full_model_mlir_sha256": worklist["source_mlir_sha256"],
        "source_softmax_region_id": softmax_receipt["softmax_region_id"],
        "qk_golden_bf16_sha256": hashlib.sha256(qk_resources["golden_bf16"]).hexdigest(),
        "captured_softmax_receipt_sha256": _sha(args.softmax_inputs / "receipt.json"),
        "score_bf16_sha256": _sha(args.out_dir / "score_bf16.bin"),
        "probability_head0_bf16_sha256": (_sha(args.out_dir / "probability_head0_bf16.bin")
                                           if passed else None),
        "model2mlir_revision": MODEL2MLIR_REVISION,
        "model2mlir_source_closure_sha256": MODEL2MLIR_SOURCE_CLOSURE,
        "profile_sha256": profile_sha256(profile),
        "rtl_revision": _git_revision(args.rtl_root),
        "source_softmax_sha256": _sha(source_path),
        "bound_mlir_sha256": _sha(mlir_path),
        "object_sha256": object_manifest["object_sha256"],
        "elf_sha256": _sha(elf),
        "spike_log_sha256": _sha(build / "spike.log"),
        "spike_exit_code": result.returncode,
        "compared_bf16_values": 512 if passed else 0,
        "quality_diagnostics": quality,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
