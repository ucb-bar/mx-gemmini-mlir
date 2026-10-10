"""Capture, compile, and qualify a source-derived wider MX/VPU resident pair.

The 64x64 MM1 and VPU use Nicolas's checked source bytes. MM2 extends the
source B2 matrix with sign-flipped source columns, then uses Nicolas's pinned
FP8 mesh model for every new output code and scale. The added width is an
explicitly derived candidate, not an unchanged Nicolas source kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.connected_fp8_model import MODEL_SHA256
from mx_gemmini_support.narrow_vpu_chain import render_connected_fp8_vpu_chain
from mx_gemmini_support.narrow_vpu_source import derive_wide_vpu_resources
from mx_gemmini_support.resident_pair_graph import INPUTS
from mx_gemmini_support.resident_vpu_graph import OUTPUTS
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run
from tools.qualify_narrow_vpu_object import MARKER, _driver


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "profiles/gemmini-mx-cleanup-266c593"
RTL_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
M2M_REVISION = "e9ded36eb85abf2d9097ac4dc11457c825853388"
MXQ_REVISION = "b4af5430bac147f4a16126931cc0177367cc3982"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _invoke(command: list[str], log: Path) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "rtl-root", "riscv-root",
                 "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", choices=(
        "MxE4M3Fp4VpuGemminiRocketConfig", "MxE4M3VpuGemminiRocketConfig"),
        default="MxE4M3Fp4VpuGemminiRocketConfig")
    parser.add_argument("--second-width", type=int, choices=(96, 128), required=True)
    args = parser.parse_args()
    for name in ("model2mlir_root", "mxq_root", "rtl_root", "riscv_root",
                 "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    for root, revision in ((args.model2mlir_root, M2M_REVISION),
                           (args.mxq_root, MXQ_REVISION),
                           (args.rtl_root, RTL_REVISION)):
        if _git_revision(root) != revision:
            parser.error(f"selected source revision differs: {root}")
    if not args.mx_opt.is_file():
        parser.error("native MX dialect verifier is absent")
    profile_path = PROFILE_DIR / f"{args.profile}.json"
    profile = load_profile(profile_path, rtl_root=args.rtl_root)
    software = args.rtl_root / "software/gemmini-rocc-tests"
    extension = args.rtl_root / "software/libgemmini"
    _require_gitlink(args.rtl_root, "software/gemmini-rocc-tests")
    _require_gitlink(args.rtl_root, "software/libgemmini")
    source = software / "bareMetalC/chain_vpu_spad_requant.c"
    header = software / "include/matmul_fp8_64x64_chain.h"
    first_source = software / "bareMetalC/matmul_tiled_fp8_64x64_chain.c"
    args.out_dir.mkdir(parents=True)
    capture = args.out_dir / "capture"
    _invoke([
        sys.executable, "-m", "tools.capture_nicolas_chain",
        "--model2mlir-root", str(args.model2mlir_root),
        "--mxq-root", str(args.mxq_root),
        "--rtl-root", str(args.rtl_root),
        "--profile", str(profile_path), "--mx-opt", str(args.mx_opt),
        "--matrix-dim", "64", "--second-width", str(args.second_width),
        "--out-dir", str(capture),
    ], args.out_dir / "capture.log")
    captured = json.loads((capture / "receipt.json").read_text())
    frontend = capture / "nicolas_chain.profile_bound.mlir"
    manifest_path = capture / "quantization_manifest.json"
    if (captured.get("schema") !=
            "mx_gemmini.nicolas_derived_wide_vpu_chain_model2mlir_capture.v1" or
            captured.get("source_sha256") != _sha(source) or
            captured.get("header_sha256") != _sha(header) or
            captured.get("profile_sha256") != profile_sha256(profile) or
            captured.get("bound_mlir_sha256") != _sha(frontend) or
            captured.get("manifest_sha256") != _sha(manifest_path) or
            captured.get("second_shape_mnk") != [64, args.second_width, 64] or
            captured.get("opaque_calls")):
        raise ValueError("wide VPU model2MLIR capture differs from pinned source")
    resources, facts = derive_wide_vpu_resources(
        source, header, first_source, profile, second_width=args.second_width)
    bound = args.out_dir / "connected.mlir"
    bound.write_text(render_connected_fp8_vpu_chain(
        frontend.read_text(), json.loads(manifest_path.read_text()),
        profile, resources, facts))
    data = args.out_dir / "resources"
    data.mkdir()
    for name, payload in sorted(resources.items()):
        (data / f"{name}.bin").write_bytes(payload)
    (data / "facts.json").write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    abi = args.out_dir / "abi.json"
    abi.write_text(json.dumps({
        "schema": "mx_gemmini.resident_vpu_buffer_map.v1",
        "inputs": {name: name for name in INPUTS},
        "outputs": {name: name for name in OUTPUTS},
    }, indent=2, sort_keys=True) + "\n")
    obj = args.out_dir / "object"
    _invoke([
        sys.executable, "-m", "tools.compile_object",
        "--mlir", str(bound), "--profile", str(profile_path),
        "--rtl-root", str(args.rtl_root), "--riscv-root", str(args.riscv_root),
        "--resources-dir", str(data), "--abi-json", str(abi),
        "--mx-opt", str(args.mx_opt), "--out-dir", str(obj),
    ], args.out_dir / "compile.log")
    compiled = json.loads((obj / "compile_manifest.json").read_text())
    object_manifest = json.loads((obj / "object_manifest.json").read_text())
    if (compiled["lowering_family"] != "resident_vpu_pair" or
            object_manifest["shape_mnk"] != [64, args.second_width, 64] or
            object_manifest["input_sha256"] != {
                name: hashlib.sha256(resources[name]).hexdigest() for name in INPUTS} or
            object_manifest["allocated_data_section_bytes"] != 0 or
            object_manifest["embedded_operand_bytes"] != 0 or
            object_manifest["embedded_golden_bytes"] != 0):
        raise ValueError("wide VPU object differs from bound runtime resources")
    build = args.out_dir / "build"
    build.mkdir()
    assembly = [".section .rodata", ".balign 64"]
    for name, payload in sorted(resources.items()):
        (build / f"{name}.bin").write_bytes(payload)
        assembly.extend((f".globl {name}", f"{name}:",
                         f'.incbin "{name}.bin"', ".balign 64"))
    assembly.append('.section .note.GNU-stack,"",@progbits')
    (build / "mx_data.S").write_text("\n".join(assembly) + "\n")
    names = tuple(entry["name"] for entry in object_manifest["buffer_abi"])
    (build / "mx_driver.c").write_text(_driver(names, args.second_width))
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if not cc.is_file() or not spike.is_file() or shutil.which("g++") is None:
        parser.error("RISC-V GCC, Spike, or host g++ is absent")
    bench = software / "riscv-tests/benchmarks/common"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET",
             "-DBAREMETAL=1", "-mcmodel=medany", "-std=gnu99", "-O2",
             "-ffast-math", "-fno-common", "-fno-builtin-printf",
             "-fno-tree-loop-distribute-patterns", "-march=rv64gc",
             "-Wa,-march=rv64gc", f"-ffile-prefix-map={build.resolve()}=.",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench)]
    sources = [build / "mx_driver.c", build / "mx_data.S"]
    sources += sorted(bench.glob("*.c")) + sorted(bench.glob("*.S"))
    objects = []
    for index, path in enumerate(sources):
        target = build / f"mx_{index}.o"
        _run([str(cc), *flags, "-c", str(path), "-o", str(target)],
             cwd=build, log=build / f"compile_{index}.log")
        objects.append(target)
    elf = build / "mx_program.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), str(obj / "mx_issue.o"),
          *(str(path) for path in objects), "-lm", "-lgcc", "-o", str(elf)],
         cwd=build, log=build / "link.log")
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc"]
    extension_sources += sorted((extension / "perf").rglob("*.cc"))
    so = build / "libgemmini.so"
    _run(["g++", "-L", str(args.riscv_root / "lib"),
          f"-Wl,-rpath,{args.riscv_root / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(args.riscv_root / "include"),
          "-fPIC", "-O3", *(str(path) for path in extension_sources)],
         cwd=build, log=build / "extension_build.log")
    result = subprocess.run(
        [str(spike), f"--extlib={so}", "--extension=gemmini", str(elf)],
        cwd=build, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    log = build / "spike.log"
    log.write_text(result.stdout)
    passed = result.returncode == 0 and MARKER in result.stdout
    receipt = {
        "schema": "mx_gemmini.derived_wide_vpu_pair_spike.v1",
        "status": ("source_derived_wide_vpu_pair_matched_on_pinned_spike" if passed else
                   "source_derived_wide_vpu_pair_failed_on_pinned_spike"),
        "source_scope": facts["source_scope"],
        "second_width": args.second_width,
        "first_shape_mnk": [64, 64, 64],
        "second_shape_mnk": [64, args.second_width, 64],
        "compared_c1_bf16_values": 4096,
        "compared_c1_fp8_codes": 4096,
        "compared_c1_e8m0_scales": 128,
        "compared_c2_fp8_codes": 64 * args.second_width,
        "compared_c2_e8m0_scales": 2 * args.second_width,
        "model_sha256": MODEL_SHA256,
        "model2mlir_revision": _git_revision(args.model2mlir_root),
        "mxq_revision": _git_revision(args.mxq_root),
        "rtl_revision": _git_revision(args.rtl_root),
        "compiler_revision": _git_revision(ROOT),
        "profile_sha256": profile_sha256(profile),
        "source_sha256": _sha(source),
        "header_sha256": _sha(header),
        "capture_receipt_sha256": _sha(capture / "receipt.json"),
        "bound_mlir_sha256": _sha(bound),
        "object_manifest_sha256": _sha(obj / "object_manifest.json"),
        "object_sha256": _sha(obj / "mx_issue.o"),
        "elf_sha256": _sha(elf),
        "spike_log_sha256": _sha(log),
        "spike_exit_code": result.returncode,
        "resources_sha256": facts["resource_sha256"],
    }
    (args.out_dir / "index.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"{receipt['status']}: {args.out_dir}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
