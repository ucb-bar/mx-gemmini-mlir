"""Replay compiler-issued QK, VPU, requant, and PV compute on Nicolas's Spike.

The pinned source driver still loads operands/scales and reads scratchpad
results. This tool verifies the latest model2MLIR/source binding, builds three
data-free command objects, links the source checker, and compares all four
source stage hashes. It does not use a bitstream.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.command_ir import emit_c
from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.nicolas_attention_matrix import lower_dense_attention_matrix
from mx_gemmini_support.nicolas_attention_io import (
    ATTENTION_BUFFERS, lower_dense_attention_operands,
    lower_dense_attention_program, lower_dense_attention_readout,
    lower_dense_attention_scales)
from mx_gemmini_support.nicolas_attention_vector import render_dense_attention_vector
from mx_gemmini_support.nicolas_attention_program import render_dense_attention_program
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _source_closure
from tools.emit_resident_pair_object import _compile_object


ROOT = Path(__file__).resolve().parents[1]
SOURCE_EVIDENCE = ROOT / "docs/evidence/nicolas_attn_vpu_source_baseline_e9ded36_0344bcb"
VECTOR_EVIDENCE = ROOT / "docs/evidence/nicolas_attn_vpu_generated_middle_e9ded36_0344bcb"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, cwd: Path, log: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _driver(middle: str) -> str:
    marker = "// spad-resident matmul"
    after = "\nint *__errno"
    qk = "mm(SQ, D, SK, SP_Q, SP_KT_ARG, SP_S, 0);"
    pv = "mm(SQ, SK, D, SP_P, SP_V_ARG, SP_O, 1);"
    if (middle.count(marker) != 1 or middle.count(after) != 1 or
            middle.count(qk) != 1 or middle.count(pv) != 1 or
            middle.count("mx_issue(p_scales);") != 1):
        raise ValueError("dense attention matrix or vector source seam differs")
    first = middle.index(marker)
    last = middle.index(after, first)
    middle = (middle[:first] +
              "void mx_issue_qk(const void *scale_sink);\n"
              "void mx_issue_pv(const void *scale_sink);\n" + middle[last:])
    return (middle.replace(qk, "mx_issue_qk(scale_sink);")
                  .replace(pv, "mx_issue_pv(scale_sink);"))


def _driver_with_io(compute: str) -> str:
    """Replace source matrix transfers and result readouts with generated calls."""
    first = compute.index("static void mvout_rows")
    last = compute.index("void mx_issue_qk", first)
    prototypes = "\n".join(
        "void mx_issue_" + name + "(" +
        ", ".join("const void *" + buffer for buffer in buffers) + ");"
        for name, buffers in (
            ("scales", ("Q_SCALES", "KT_SCALES", "V_SCALES")),
            ("operands", ("Q_IN", "KT_IN", "V_IN")),
            ("score", ("S_hw",)), ("probability", ("P_hw",)),
            ("probability_codes", ("p_codes_hw",)), ("output", ("O_hw",))))
    compute = compute[:first] + prototypes + "\n" + compute[last:]
    replacements = (
        ("  mvin_A((const uint8_t *)Q_IN, SQ, D, SP_Q);\n"
         "  mvin_B((const uint8_t *)KT_IN, D, SK, SP_KT);\n"
         "  mvin_B((const uint8_t *)V_IN, SK, D, SP_V);",
         "  mx_issue_operands(Q_IN, KT_IN, V_IN);"),
        ("  gemmini_mx_load_scales((uint64_t)Q_SCALES, sizeof(Q_SCALES), 0);                    // act half 0\n"
         "  gemmini_mx_load_scales((uint64_t)KT_SCALES, sizeof(KT_SCALES), 1);                  // wgt half 0\n"
         "  gemmini_mx_load_scales_2d((uint64_t)V_SCALES, D, SK / 32, D, 4096, 1);              // wgt half 1",
         "  mx_issue_scales(Q_SCALES, KT_SCALES, V_SCALES);"),
        ("mvout_rows(S_hw, SP_S, ROWS16(SQ, SK));", "mx_issue_score(S_hw);"),
        ("mvout_rows(O_hw, SP_O, ROWS16(SQ, D));", "mx_issue_output(O_hw);"),
        ("mvout_rows(P_hw, SP_S, ROWS16(SQ, SK));", "mx_issue_probability(P_hw);"),
        ("mvout_rows(p_codes_hw, SP_P, ROWS8(SQ, SK));",
         "mx_issue_probability_codes(p_codes_hw);"),
    )
    for old, new in replacements:
        if compute.count(old) != 1:
            raise ValueError("dense attention source transfer or readout seam differs")
        compute = compute.replace(old, new)
    if any(token in compute for token in (
            "gemmini_extended_mvin(", "gemmini_extended_mvout(",
            "gemmini_mx_load_scales(", "gemmini_mx_load_scales_2d(")):
        raise ValueError("dense attention driver retained handwritten transfer commands")
    return compute


def _driver_one_object(io_driver: str) -> str:
    """Call one compiler issuer for the whole checked attention command stream."""
    first = io_driver.index("void mx_issue_scales(")
    last = io_driver.index("\nint *__errno", first)
    prototype = ("void mx_issue(" +
                 ", ".join("const void *" + name for name in ATTENTION_BUFFERS) +
                 ");\n")
    io_driver = io_driver[:first] + prototype + io_driver[last:]
    old_vector_prototype = "void mx_issue(const void *p_scales);\n"
    if io_driver.count(old_vector_prototype) != 1:
        raise ValueError("dense attention vector source declaration differs")
    io_driver = io_driver.replace(old_vector_prototype, "")
    first = io_driver.index("static void attention(int check) {")
    last = io_driver.index("\nstatic void rel_errs", first)
    call = ", ".join(ATTENTION_BUFFERS)
    body = ("static void attention(int check) {\n"
            "  (void)check;\n"
            "  mark(0);\n"
            f"  mx_issue({call});\n"
            "  gemmini_fence();\n"
            "  st[6] = read_cycles();\n"
            "}\n")
    io_driver = io_driver[:first] + body + io_driver[last:]
    scale_call = "  mx_issue_scales(Q_SCALES, KT_SCALES, V_SCALES);"
    if io_driver.count(scale_call) != 1:
        raise ValueError("dense attention source scale setup differs")
    return io_driver.replace(scale_call, "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("npu-root", "profile", "rtl-root", "riscv-root",
                 "spike-extension", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--mx-opt", type=Path)
    parser.add_argument("--generated-io", action="store_true",
                        help="also replace source scale, operand, and readout commands")
    parser.add_argument("--single-object", action="store_true",
                        help="link one data-free issuer for the entire attention command stream")
    parser.add_argument("--typed-program", action="store_true",
                        help="compile the full typed attention graph with the public object CLI")
    args = parser.parse_args()
    if args.typed_program:
        args.single_object = True
    if args.single_object:
        args.generated_io = True
    for name in ("npu_root", "profile", "rtl_root", "riscv_root",
                 "spike_extension", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    source_path = args.npu_root / "baremetal/mxgemmini/src/attn_vpu.c"
    npu_src = args.npu_root / "baremetal/mxgemmini"
    baseline = json.loads((SOURCE_EVIDENCE / "receipt.json").read_text())
    source = source_path.read_text()
    header = gzip.decompress((SOURCE_EVIDENCE / "attn_vpu.h.gz").read_bytes())
    frontend = (SOURCE_EVIDENCE / "attention.model2mlir.mlir").read_text()
    bound, binding = render_dense_attention_vector(
        frontend, source, header.decode(), profile,
        (ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml").read_bytes())
    archived_binding = json.loads((VECTOR_EVIDENCE / "vector_binding.json").read_text())
    if (hashlib.sha256(source.encode()).hexdigest() != baseline["npu_source_sha256"] or
            hashlib.sha256(header).hexdigest() != baseline["generated_header_sha256"] or
            bound != (VECTOR_EVIDENCE / "attention.vector_bound.mlir").read_text() or
            binding != archived_binding):
        raise ValueError("current source, header, or frontend differs from bound attention")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    bench = software / "riscv-tests/benchmarks/common"
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    for path in (source_path, cc, spike, args.spike_extension, bench / "test.ld"):
        if not path.is_file():
            raise ValueError(f"dense attention replay input is absent: {path}")
    out = args.out_dir
    out.mkdir(parents=True)
    vector = out / "vector"
    cmd = [sys.executable, "-m", "tools.compile_object",
           "--mlir", str(VECTOR_EVIDENCE / "attention.vector_bound.mlir"),
           "--binding-json", str(VECTOR_EVIDENCE / "vector_binding.json"),
           "--profile", str(args.profile), "--rtl-root", str(args.rtl_root),
           "--riscv-root", str(args.riscv_root), "--out-dir", str(vector)]
    if args.mx_opt:
        cmd.extend(("--mx-opt", str(args.mx_opt.resolve())))
    _run(cmd, cwd=ROOT, log=out / "vector_compile.log")
    if _sha(vector / "mx_issue.o") != _sha(VECTOR_EVIDENCE / "mx_issue.o"):
        raise ValueError("public vector object differs from Spike-qualified middle")
    objects = [vector / "mx_issue.o"]
    for stage in ("qk", "pv"):
        directory = out / stage
        directory.mkdir()
        commands = lower_dense_attention_matrix(profile, stage)
        (directory / "mx_issue.c").write_text(
            emit_c(list(commands), transport="rocket_rocc",
                   buffers=("scale_sink",), symbol=f"mx_issue_{stage}"))
        obj, data_bytes = _compile_object(directory, args.riscv_root,
                                          symbol=f"mx_issue_{stage}")
        if data_bytes:
            raise ValueError(f"dense attention {stage} issuer embeds source data")
        objects.append(obj)
    io_objects = {}
    if args.generated_io:
        io_plans = [
            ("scales", lower_dense_attention_scales(profile),
             ("Q_SCALES", "KT_SCALES", "V_SCALES")),
            ("operands", lower_dense_attention_operands(profile),
             ("Q_IN", "KT_IN", "V_IN")),
        ]
        for name, buffer in (("score", "S_hw"), ("probability", "P_hw"),
                             ("probability_codes", "p_codes_hw"),
                             ("output", "O_hw")):
            io_plans.append((name, lower_dense_attention_readout(profile, name),
                             (buffer,)))
        for name, commands, buffers in io_plans:
            directory = out / name
            directory.mkdir()
            (directory / "mx_issue.c").write_text(
                emit_c(list(commands), transport="rocket_rocc", buffers=buffers,
                       symbol=f"mx_issue_{name}"))
            obj, data_bytes = _compile_object(directory, args.riscv_root,
                                              symbol=f"mx_issue_{name}")
            if data_bytes:
                raise ValueError(f"dense attention {name} issuer embeds source data")
            io_objects[name] = obj
            objects.append(obj)
    single_object = None
    if args.single_object:
        full = lower_dense_attention_program(bound, profile, binding)
        directory = out / "single"
        directory.mkdir()
        (directory / "mx_issue.c").write_text(
            emit_c(list(full), transport="rocket_rocc", buffers=ATTENTION_BUFFERS))
        single_object, data_bytes = _compile_object(directory, args.riscv_root)
        if data_bytes:
            raise ValueError("single dense attention issuer embeds source data")
        (directory / "physical_program.json").write_text(json.dumps({
            "schema": "mx_gemmini.nicolas_attention_single_physical.v1",
            "profile_sha256": profile_sha256(profile),
            "bound_vector_mlir_sha256": hashlib.sha256(bound.encode()).hexdigest(),
            "buffers": ATTENTION_BUFFERS,
            "commands": [asdict(item) if isinstance(item, Command) else
                         {"kind": "fence"} for item in full],
        }, indent=2, sort_keys=True) + "\n")
        objects = [single_object]
    typed_object = None
    typed_mlir = None
    if args.typed_program:
        typed_mlir = out / "attention.program_bound.mlir"
        typed_mlir.write_text(render_dense_attention_program(bound, profile, binding))
        directory = out / "typed"
        command = [sys.executable, "-m", "tools.compile_object",
                   "--mlir", str(typed_mlir),
                   "--precursor-mlir", str(VECTOR_EVIDENCE / "attention.vector_bound.mlir"),
                   "--binding-json", str(VECTOR_EVIDENCE / "vector_binding.json"),
                   "--profile", str(args.profile), "--rtl-root", str(args.rtl_root),
                   "--riscv-root", str(args.riscv_root), "--out-dir", str(directory)]
        if args.mx_opt:
            command.extend(("--mx-opt", str(args.mx_opt.resolve())))
        _run(command, cwd=ROOT, log=out / "typed_compile.log")
        typed_object = directory / "mx_issue.o"
        if _sha(typed_object) != _sha(single_object):
            raise ValueError("typed attention object differs from Spike-qualified physical stream")
        objects = [typed_object]
    driver = out / ("attn_vpu_typed_program.c" if args.typed_program else
                    "attn_vpu_single_object.c" if args.single_object else
                    "attn_vpu_generated_io.c" if args.generated_io else
                    "attn_vpu_generated_compute.c")
    compute_driver = _driver(
        (VECTOR_EVIDENCE / "attn_vpu_generated_middle.c").read_text())
    staged_driver = _driver_with_io(compute_driver) if args.generated_io else compute_driver
    driver.write_text(_driver_one_object(staged_driver) if args.single_object else
                      staged_driver)
    (out / "attn_vpu.h").write_bytes(header)
    shutil.copyfile(SOURCE_EVIDENCE / "attn_vpu_expect.h",
                    out / "attn_vpu_expect.h")
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             "-I", str(npu_src), "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [driver, *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S"))]
    for index, path in enumerate(sources):
        obj = out / f"source_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(obj)],
             cwd=out, log=out / f"source_{index}.log")
        objects.append(obj)
    elf = out / ("attn_vpu_typed_program.elf" if args.typed_program else
                 "attn_vpu_single_object.elf" if args.single_object else
                 "attn_vpu_generated_io.elf" if args.generated_io else
                 "attn_vpu_generated_compute.elf")
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(obj) for obj in objects),
          "-lm", "-lgcc", "-o", str(elf)], cwd=out, log=out / "link.log")
    logs = []
    for run in (1, 2):
        command = [str(spike), f"--extlib={args.spike_extension}",
                   "--extension=gemmini", str(elf)]
        log = out / f"spike_{run}.log"
        _run(command, cwd=out, log=log)
        logs.append(log.read_text())
    expected_hashes = baseline["source_stage_hashes"]
    hashes = "check hashes: " + "  ".join(
        f"{name} {expected_hashes[name]}" for name in ("S", "P", "Pq", "O"))
    if (logs[0] != logs[1] or hashes not in logs[0] or
            "check S vs mesh golden: 0/512 4-element words differ" not in logs[0] or
            "check hashes vs Spike: match (0 of 4 differ)" not in logs[0] or
            "attn_vpu PASSED" not in logs[0]):
        raise ValueError("compiler-issued attention compute differs from source on Spike")
    receipt = {
        "schema": ("mx_gemmini.nicolas_attention_typed_program.v1" if args.typed_program
                   else "mx_gemmini.nicolas_attention_single_object.v1" if args.single_object
                   else "mx_gemmini.nicolas_attention_generated_io.v1" if args.generated_io
                   else "mx_gemmini.nicolas_attention_generated_compute.v1"),
        "status": ("typed_attention_object_matches_source_on_spike" if args.typed_program
                   else "single_attention_object_matches_source_on_spike" if args.single_object
                   else "generated_attention_io_and_compute_matches_source_on_spike"
                   if args.generated_io else
                   "generated_qk_vpu_requant_pv_compute_matches_source_on_spike"),
        "scope": ("one public-CLI RV64 object from typed QK/VPU/requant/PV and I/O ops "
                  "issues all attention kernel commands; source checker still flushes/fences"
                  if args.typed_program else
                  "one data-free object issues all MX attention kernel commands; "
                  "source checker still flushes/fences; QK/PV and I/O are source-specific physical lowering"
                  if args.single_object else
                  "all scale loads, operand transfers, QK, VPU, requant, PV, "
                  "and readouts generated; source checker still flushes/fences"
                  if args.generated_io else
                  "all QK, VPU, resident requant, and PV compute commands generated; "
                  "source C still issues operand/scale transfers and readouts"),
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "npu_revision": _git_revision(args.npu_root),
        "source_sha256": _sha(source_path),
        "header_sha256": hashlib.sha256(header).hexdigest(),
        "frontend_mlir_sha256": _sha(SOURCE_EVIDENCE / "attention.model2mlir.mlir"),
        "profile_sha256": profile_sha256(profile),
        "rtl_revision": _git_revision(args.rtl_root),
        "spike_sha256": _sha(spike), "spike_extension_sha256": _sha(args.spike_extension),
        "riscv_gcc_sha256": _sha(cc),
        "driver_sha256": _sha(driver),
        "matrix_object_sha256": {stage: _sha(out / stage / "mx_issue.o")
                                 for stage in ("qk", "pv")},
        "vector_object_sha256": _sha(vector / "mx_issue.o"),
        "elf_sha256": _sha(elf),
        "spike_log_sha256": _sha(out / "spike_1.log"),
        "spike_reproduced_runs": 2,
        "stage_hashes": expected_hashes,
    }
    if args.generated_io:
        receipt["io_object_sha256"] = {name: _sha(path)
                                       for name, path in io_objects.items()}
    if single_object is not None:
        receipt["single_object_sha256"] = _sha(single_object)
        receipt["single_physical_program_sha256"] = _sha(
            out / "single/physical_program.json")
    if typed_object is not None:
        receipt["typed_mlir_sha256"] = _sha(typed_mlir)
        receipt["typed_object_sha256"] = _sha(typed_object)
        receipt["typed_compile_manifest_sha256"] = _sha(
            out / "typed/compile_manifest.json")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("compiler-issued attention " + ("typed object" if args.typed_program else
                                           "single object" if args.single_object else
                                           "I/O and compute" if args.generated_io
                                           else "compute") +
          " matched all source hashes on Spike")


if __name__ == "__main__":
    main()
