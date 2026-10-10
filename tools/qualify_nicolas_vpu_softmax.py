"""Capture BF16 softmax with model2MLIR and run compiler-issued VPU ops on Spike.

The copied Nicolas source retains input generation, transfers, and its bit-exact
reference checker. Its six VPU calls are replaced by six commands lowered from
the profile-bound MX MLIR; this qualifies the vector compute seam only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.source_softmax import (render_softmax_bound,
                                               replace_source_commands)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.vector_lowering import lower_vector_c
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "profile", "rtl-root", "riscv-root",
                 "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("model2mlir_root", "profile", "rtl_root", "riscv_root",
                 "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    for path in (args.mx_opt, args.riscv_root / "bin/riscv64-unknown-elf-gcc",
                 args.riscv_root / "bin/spike"):
        if not path.is_file():
            parser.error(f"required executable is absent: {path}")
    if shutil.which("g++") is None:
        parser.error("host g++ is required to build Nicolas's Spike extension")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    software = args.rtl_root / "software/gemmini-rocc-tests"
    extension = args.rtl_root / "software/libgemmini"
    _require_gitlink(args.rtl_root, "software/gemmini-rocc-tests")
    _require_gitlink(args.rtl_root, "software/libgemmini")
    source_path = software / "bareMetalC/vpu_softmax.c"
    source = source_path.read_text()
    sys.path.insert(0, str(args.model2mlir_root.resolve()))
    import m2m
    import torch
    from m2m.coverage import opaque_report

    if Path(m2m.__file__).resolve().parents[1] != args.model2mlir_root.resolve():
        raise RuntimeError("model2MLIR resolved to a different checkout")

    class Softmax(torch.nn.Module):
        def forward(self, scores: torch.Tensor) -> torch.Tensor:
            return torch.softmax(scores, dim=-1)

    scores = torch.zeros((16, 32), dtype=torch.bfloat16)
    captured = m2m.convert(Softmax().eval(), (scores,), backend="fx_importer")
    if not captured.ok or opaque_report(captured.mlir_text):
        raise ValueError(f"model2MLIR left an opaque or invalid softmax: {captured.diagnostics}")
    mlir, binding = render_softmax_bound(
        captured.mlir_text, source, profile, CONTRACT.read_bytes())
    transformed_source = replace_source_commands(source)
    issuer = lower_vector_c(mlir, profile)
    if issuer.count(".insn r 0x7b, 3, 33") != 6:
        raise ValueError("compiler did not emit the six VPU commands")
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    frontend_path = out / "softmax.model2mlir.mlir"
    frontend_path.write_text(captured.mlir_text)
    bound_path = out / "softmax.profile_bound.mlir"
    bound_path.write_text(mlir)
    (out / "binding_manifest.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True) + "\n")
    _run([str(args.mx_opt.resolve()), str(bound_path), "-o", "/dev/null"],
         cwd=out, log=out / "native_verify.log")
    build = out / "build"
    build.mkdir()
    (build / "mx_issue.c").write_text(issuer)
    (build / "mx_driver.c").write_text(transformed_source)
    riscv_cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    bench = software / "riscv-tests/benchmarks/common"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             f"-ffile-prefix-map={build}=.",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [build / "mx_issue.c", build / "mx_driver.c"]
    sources += sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))
    objects = []
    for index, path in enumerate(sources):
        obj = build / f"mx_{index}.o"
        _run([str(riscv_cc), *flags, "-c", str(path), "-o", str(obj)],
             cwd=build, log=build / f"compile_{index}.log")
        objects.append(obj)
    elf = build / "mx_program.elf"
    _run([str(riscv_cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(path) for path in objects), "-lm", "-lgcc",
          "-o", str(elf)], cwd=build, log=build / "link.log")
    ext_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    ext_sources += sorted((extension / "perf").rglob("*.cc"))
    so = build / "libgemmini.so"
    _run(["g++", "-L", str(args.riscv_root / "lib"),
          f"-Wl,-rpath,{args.riscv_root / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(args.riscv_root / "include"), "-fPIC", "-O3",
          *(str(path) for path in ext_sources)],
         cwd=build, log=build / "extension_build.log")
    result = subprocess.run([str(spike), f"--extlib={so}", "--extension=gemmini", str(elf)],
                            cwd=build, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log = build / "spike.log"
    log.write_text(result.stdout)
    passed = (result.returncode == 0 and
              "softmax 16x32: 0 mismatches vs ref" in result.stdout and
              "vpu_softmax PASSED" in result.stdout)
    receipt = {
        "schema": "mx_gemmini.nicolas_vpu_softmax_spike.v1",
        "status": "source_vpu_softmax_matched_on_pinned_spike" if passed else
                  "source_vpu_softmax_failed_on_pinned_spike",
        "scope": "model2MLIR BF16 softmax capture; six compiler-issued VPU compute commands; Nicolas source transfers and reference checker",
        "source_revision": _git_revision(software),
        "source_sha256": _sha(source_path),
        "model2mlir_revision": _git_revision(args.model2mlir_root),
        "rtl_revision": _git_revision(args.rtl_root),
        "extension_revision": _git_revision(extension),
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "profile_sha256": profile_sha256(profile),
        "frontend_mlir_sha256": _sha(frontend_path),
        "bound_mlir_sha256": _sha(bound_path),
        "binding_manifest_sha256": _sha(out / "binding_manifest.json"),
        "files_sha256": {path.name: _sha(path) for path in
                         (build / "mx_issue.c", build / "mx_driver.c")},
        "object_sha256": {path.name: _sha(path) for path in objects},
        "riscv_gcc_sha256": _sha(riscv_cc), "spike_sha256": _sha(spike),
        "elf_sha256": _sha(elf), "extension_sha256": _sha(so),
        "spike_log_sha256": _sha(log), "spike_exit_code": result.returncode,
        "compared_bf16_values": 512,
    }
    (out / "artifact_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{receipt['status']}: {out / 'artifact_manifest.json'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
