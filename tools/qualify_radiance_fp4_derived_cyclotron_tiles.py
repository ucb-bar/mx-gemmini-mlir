"""Run four source-derived FP4 single-tile Muon kernels on Cyclotron.

The pinned Radiance shared GEMM helper executes one output tile even when a
driver passes larger M/N dimensions. This qualifier slices the generated
256x256 fixture into four 128x128 source-compatible headers, builds one
existing Radiance single-tile driver per header, and reassembles their actual
Muon/MX outputs. It does not claim that the unmodified 256x256 driver works.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.quant_reference import exact_bf16_x2
from mx_gemmini_support.source_payload import load_bundle


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/radiance_fp4_generated_tilewise_vpu_266c593"
RAW_DRIVER = ("fixture/kernels/gemm_mxgemmini/"
              "mxgemm.fp4.m256n256k256.tm128tn128tk128.fullout.cpp")
RAW_HEADER = "fixture/kernels/gemm_mxgemmini/mxgemm.data.fp4.m256n256k256.h.gz"
SOURCE_REVISION = "80f84caedbabc663a7433c1da4455b936cca41f3"
MX_SOFTWARE_REVISION = "6fc8ec79bc828b33628ec951bfa1d0f65278a338"
CYCLOTRON_REVISION = "2d6adad4ad94d1621fdff9c9a1e5eac871048f24"
SOURCE_FILES = {
    "kernels/common.mk": "fa889bdf2587f3c5e6a2216a0babb030be4f2b6d1e377398845b958593802b07",
    "kernels/gemm_mxgemmini/mxgemm.fp4.m128n128k256.tm128tn128tk128.fullout.cpp":
        "33accc38f62228e978b6bf82342dbb0ff6f35812081fdab1371f640390042e9d",
    "lib/mxgemm/mxgemm_lib.hpp":
        "003302c019de637a02b285a90ca40f114a53341f6ee22bec774fd20310a7177d",
}
CYCLOTRON_MODEL_SHA256 = "49b42427082f4bca7f0e30ceca8631563cf24376009cea2e458351fbac46cffd"
CYCLOTRON_CONFIG_SHA256 = "70ef96c088c01caf3093bdff649fdb1926ea60ee68c4fcbd98af5edbf6dbf508"
CYCLOTRON_BINARY_SHA256 = "8f87b4bdf9392304cc7d7c87e793ab6873405f46e5ce756a6ca523300bf3751d"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path,
                                   text=True).strip()


def _array(name: str, ctype: str, rows: int, columns: int,
           values: bytes | list[int]) -> str:
    if len(values) != rows * columns:
        raise ValueError(f"{name} has the wrong number of elements")
    width = 2 if ctype == "uint8_t" else 4
    lines = [f"static const {ctype} {name}[{rows}][{columns}] = {{"]
    for row in range(rows):
        segment = values[row * columns:(row + 1) * columns]
        lines.append("    { " + ", ".join(f"0x{value:0{width}x}" for value in segment) + " },")
    return "\n".join(lines + ["};", ""])


def _slice_rows(data: bytes, row_width: int, row_start: int,
                rows: int, col_start: int, columns: int) -> bytes:
    return b"".join(data[r * row_width + col_start:r * row_width + col_start + columns]
                    for r in range(row_start, row_start + rows))


def _tile_header(resources: dict[str, bytes], tile_m: int, tile_n: int) -> tuple[str, bytes]:
    if tile_m not in (0, 1) or tile_n not in (0, 1):
        raise ValueError("only the four 128x128 output tiles are supported")
    a = _slice_rows(resources["activation"], 256, tile_m * 64, 64, 0, 256)
    b = _slice_rows(resources["weight"], 128, 0, 256, tile_n * 64, 64)
    a_scales = _slice_rows(resources["activation_scales"], 256, 0, 8,
                           tile_m * 128, 128)
    b_scales = _slice_rows(resources["weight_scales"], 256, 0, 8,
                           tile_n * 128, 128)
    golden_bytes = _slice_rows(resources["golden_bf16"], 256 * 2,
                               tile_m * 128, 128, tile_n * 128 * 2, 128 * 2)
    golden_words = [int.from_bytes(golden_bytes[i:i + 2], "little")
                    for i in range(0, len(golden_bytes), 2)]
    header = ("// Derived from the pinned Radiance 256x256 FP4 generator output.\n"
              "#include <stdint.h>\n"
              "#define MATMUL_M 128\n#define MATMUL_N 128\n#define MATMUL_K 256\n"
              "#define MATMUL_GK 8\n#define MATMUL_GN 4\n")
    for name, ctype, rows, columns, values in (
        ("A_in_hw", "uint8_t", 64, 256, a),
        ("B_in", "uint8_t", 256, 64, b),
        ("A_scales_row", "uint8_t", 8, 128, a_scales),
        ("B_scales_col", "uint8_t", 8, 128, b_scales),
        ("C_out_bf16", "uint16_t", 128, 128, golden_words),
    ):
        header += _array(name, ctype, rows, columns, values)
    return header, golden_bytes


def _run(command: list[str], *, cwd: Path, log: Path,
         env: dict[str, str] | None = None) -> None:
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _build(args: argparse.Namespace, kernel_dir: Path, name: str, log: Path,
           env: dict[str, str]) -> Path:
    sysroot = args.riscv_root / "riscv64-unknown-elf"
    _run(["make", f"{name}.radiance.elf", f"MU_SRCS={name}.cpp",
          f"MU_CXX={args.llvm_muon / 'bin/clang++'} -stdlib=libc++",
          f"LLVM_MUON={args.llvm_muon}",
          f"RADIANCE_LIB_PATH={args.radiance_lib_root}",
          f"GEMMINI_SW_PATH={args.mx_software_root}",
          f"RISCV_TOOLCHAIN_PATH={args.riscv_root}",
          f"RISCV_SYSROOT={sysroot}",
          f"MU_LIBC_INCLUDE={sysroot / 'include'}",
          "EXTRA_MU_CFLAGS=-nostdinc++"],
         cwd=kernel_dir, log=log, env=env)
    return kernel_dir / f"{name}.radiance.elf"


def _simulate(cyclotron: Path, binary: Path, elf: Path, output: Path,
              log: Path, length: int) -> bytes:
    env = os.environ.copy()
    env["CYCLOTRON_MXGEMMINI"] = "1"
    env["CYCLOTRON_DUMP_GMEM"] = f"0x40000000:{length}:{output}"
    _run([str(binary), "config.toml", "--binary-path", str(elf)],
         cwd=cyclotron, log=log, env=env)
    data = output.read_bytes()
    if len(data) != length or b"simulation finished" not in log.read_bytes():
        raise ValueError(f"Cyclotron did not finish or dump {length} bytes: {log}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "radiance-lib-root", "mx-software-root",
                 "cyclotron-root", "llvm-muon", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("source_root", "radiance_lib_root", "mx_software_root",
                 "cyclotron_root", "llvm_muon", "riscv_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    out = args.out_dir
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    if (_revision(args.source_root) != SOURCE_REVISION or
            _revision(args.radiance_lib_root.parent) != SOURCE_REVISION or
            _revision(args.mx_software_root) != MX_SOFTWARE_REVISION or
            _revision(args.cyclotron_root) != CYCLOTRON_REVISION or
            any(_file_sha(args.source_root / name) != digest
                for name, digest in SOURCE_FILES.items())):
        parser.error("Radiance source, software, or Cyclotron differs from pinned revision")
    cyclotron = args.cyclotron_root
    model = cyclotron / "src/muon/mxgemmini/mod.rs"
    binary = cyclotron / "target/release/cyclotron"
    if (_file_sha(model) != CYCLOTRON_MODEL_SHA256 or
            _file_sha(cyclotron / "config.toml") != CYCLOTRON_CONFIG_SHA256 or
            _file_sha(binary) != CYCLOTRON_BINARY_SHA256):
        parser.error("Cyclotron must use the qualified accumulator-overwrite model")
    manifest, resources = load_bundle(EVIDENCE / "bundle")
    if (manifest.get("shape_mnk") != [256, 256, 256] or
            manifest.get("tile_mnk") != [128, 128, 128] or
            manifest.get("precision") != "FP4"):
        parser.error("archived generated FP4 fixture differs")
    base = (args.source_root / next(name for name in SOURCE_FILES if name.endswith(
        "mxgemm.fp4.m128n128k256.tm128tn128tk128.fullout.cpp"))).read_text()
    old_header = 'mxgemm.data.fp4.m128n128k256.h'
    if base.count(old_header) != 1 or base.count(
            'static const uint8_t *A_in = &A_in_hw[0][0];') != 1:
        parser.error("Radiance FP4 single-tile template changed")
    kernel_dir = out / "kernels/gemm_mxgemmini"
    kernel_dir.mkdir(parents=True)
    shutil.copyfile(args.source_root / "kernels/common.mk", out / "kernels/common.mk")
    (kernel_dir / "Makefile").write_text("PROJECT = mxgemm\nMU_SRCS =\ninclude ../common.mk\n")
    env = os.environ.copy()
    env["CPLUS_INCLUDE_PATH"] = str(args.llvm_muon / "include/c++/v1")
    cases = []
    full_output = bytearray(256 * 256 * 2)
    for tile_m in range(2):
        for tile_n in range(2):
            name = f"tile_{tile_m}{tile_n}"
            header_name = f"{name}.h"
            header, golden = _tile_header(resources, tile_m, tile_n)
            (kernel_dir / header_name).write_text(header)
            driver = kernel_dir / f"{name}.cpp"
            driver.write_text(base.replace(old_header, header_name))
            build_log = out / f"{name}.build.log"
            elf = _build(args, kernel_dir, name, build_log, env)
            output = out / f"{name}.bf16.bin"
            sim_log = out / f"{name}.cyclotron.log"
            actual = _simulate(cyclotron, binary, elf, output, sim_log, 32768)
            if actual != golden:
                raise ValueError(f"source-derived Muon/MX {name} differs from Radiance golden")
            for row in range(128):
                dst = ((tile_m * 128 + row) * 256 + tile_n * 128) * 2
                src = row * 128 * 2
                full_output[dst:dst + 128 * 2] = actual[src:src + 128 * 2]
            cases.append({"tile": [tile_m, tile_n],
                          "header_sha256": _file_sha(kernel_dir / header_name),
                          "driver_sha256": _file_sha(driver),
                          "elf_sha256": _file_sha(elf),
                          "build_log_sha256": _file_sha(build_log),
                          "cyclotron_log_sha256": _file_sha(sim_log),
                          "output_sha256": _sha(actual),
                          "matched_bf16": 16384})
    full = bytes(full_output)
    (out / "assembled_bf16.bin").write_bytes(full)
    scaled = exact_bf16_x2(full)
    (out / "assembled_x2_bf16.bin").write_bytes(scaled)
    compiler = json.loads((EVIDENCE / "build/artifact_manifest.json").read_text())
    if (compiler.get("status") != "derived_vpu_golden_matched_on_pinned_spike" or
            compiler.get("compared_bf16_outputs") != 65536 or
            full != resources["golden_bf16"] or
            _sha(scaled) != compiler["files_sha256"]["derived_expected_bf16.bin"]):
        raise ValueError("four executed Muon tiles differ from compiler's VPU x2 reference")
    raw_driver = (EVIDENCE / RAW_DRIVER).read_bytes()
    raw_header = gzip.decompress((EVIDENCE / RAW_HEADER).read_bytes())
    if (_sha(raw_driver) != manifest["source_driver_sha256"] or
            _sha(raw_header) != manifest["source_header_sha256"]):
        raise ValueError("raw generated driver or header differs from source manifest")
    (kernel_dir / "raw_256x256.cpp").write_bytes(raw_driver)
    (kernel_dir / "mxgemm.data.fp4.m256n256k256.h").write_bytes(raw_header)
    raw_build_log = out / "raw_256x256.build.log"
    raw_elf = _build(args, kernel_dir, "raw_256x256", raw_build_log, env)
    raw_output = out / "raw_256x256.bf16.bin"
    raw_sim_log = out / "raw_256x256.cyclotron.log"
    raw = _simulate(cyclotron, binary, raw_elf, raw_output, raw_sim_log, 131072)
    mismatches = sum(raw[i:i + 2] != full[i:i + 2] for i in range(0, len(raw), 2))
    if mismatches == 0:
        raise ValueError("raw 256x256 source driver unexpectedly matches all four tiles")
    receipt = {
        "schema": "mx_gemmini.radiance_generated_fp4_four_source_tiles_cyclotron.v1",
        "status": "four_derived_muon_mx_tiles_match_compiler_vpu_reference",
        "scope": "four separately built source-derived single-tile kernels; not the "
                 "unmodified 256x256 Radiance driver, RTL, or FPGA",
        "source_revision": SOURCE_REVISION,
        "mx_software_revision": MX_SOFTWARE_REVISION,
        "compiler_revision": _revision(ROOT),
        "cyclotron_revision": CYCLOTRON_REVISION,
        "cyclotron_model_sha256": CYCLOTRON_MODEL_SHA256,
        "cyclotron_binary_sha256": CYCLOTRON_BINARY_SHA256,
        "source_files_sha256": SOURCE_FILES,
        "muon_clang_sha256": _file_sha(args.llvm_muon / "bin/clang"),
        "riscv_gcc_sha256": _file_sha(args.riscv_root / "bin/riscv64-unknown-elf-gcc"),
        "muon_runtime_archive_sha256": _file_sha(args.radiance_lib_root / "libmuonrt.a"),
        "source_header_sha256": manifest["source_header_sha256"],
        "source_payload_manifest_sha256": _file_sha(EVIDENCE / "bundle/manifest.json"),
        "source_golden_bf16_sha256": _sha(resources["golden_bf16"]),
        "assembled_bf16_sha256": _sha(full),
        "assembled_x2_bf16_sha256": _sha(scaled),
        "compiler_derived_x2_bf16_sha256": compiler["files_sha256"]["derived_expected_bf16.bin"],
        "compared_bf16": len(full) // 2,
        "raw_driver": {
            "status": "builds_but_does_not_compute_four_output_tiles",
            "mismatched_bf16": mismatches,
            "driver_sha256": _sha(raw_driver),
            "header_sha256": _sha(raw_header),
            "elf_sha256": _file_sha(raw_elf),
            "output_sha256": _sha(raw),
            "build_log_sha256": _file_sha(raw_build_log),
            "cyclotron_log_sha256": _file_sha(raw_sim_log),
        },
        "cases": cases,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("four source-derived FP4 Muon/MX tiles: 65,536/65,536 BF16 matched Cyclotron")


if __name__ == "__main__":
    main()
