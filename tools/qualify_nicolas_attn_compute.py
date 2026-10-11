"""Replay compiler-issued QK, VPU, requant, and PV compute on Nicolas's Spike.

The pinned source driver still loads operands/scales and reads scratchpad
results. This tool verifies the latest model2MLIR/source binding, builds three
data-free command objects, links the source checker, and compares all four
source stage hashes. It does not use a bitstream.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.command_ir import emit_c
from mx_gemmini_support.nicolas_attention_matrix import lower_dense_attention_matrix
from mx_gemmini_support.nicolas_attention_vector import render_dense_attention_vector
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("npu-root", "profile", "rtl-root", "riscv-root",
                 "spike-extension", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--mx-opt", type=Path)
    args = parser.parse_args()
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
    driver = out / "attn_vpu_generated_compute.c"
    driver.write_text(_driver(
        (VECTOR_EVIDENCE / "attn_vpu_generated_middle.c").read_text()))
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
    elf = out / "attn_vpu_generated_compute.elf"
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
        "schema": "mx_gemmini.nicolas_attention_generated_compute.v1",
        "status": "generated_qk_vpu_requant_pv_compute_matches_source_on_spike",
        "scope": ("all QK, VPU, resident requant, and PV compute commands generated; "
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
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("compiler-issued QK/VPU/requant/PV compute matched all source hashes on Spike")


if __name__ == "__main__":
    main()
