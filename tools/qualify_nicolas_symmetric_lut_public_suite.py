"""Replay Nicolas's three same-format LUT matrices through public RV64 objects."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_plain_matrix_object import (
    MODEL2MLIR_REVISION, MXQ_REVISION, ROOT, RTL_REVISION, _revision,
)


SCHEMA = "mx_gemmini.nicolas_symmetric_lut_public_suite.v1"


@dataclass(frozen=True)
class Case:
    profile: str
    source: str
    header: str
    source_sha256: str
    header_sha256: str
    mesh_dim: int = 16
    source_shape: str = "64x64"


CASES = {
    "e2m3": Case(
        "MxE2M3OnlyGemminiRocketConfig",
        "matmul_tiled_fp6_e2m3_lut_64x64.c",
        "matmul_data_mx_lut_e2m3_64x64.h",
        "a142f11d56ffca02baf620d7bc94e125721618f1e41b806625d3403a49269e15",
        "78e21f8aa147979d91fb70358163b83164505eb38565f256ff4d03156a773cd0"),
    "e4m3": Case(
        "MxE4M3LutGemminiRocketConfig",
        "matmul_tiled_fp8_e4m3_lut_64x64.c",
        "matmul_data_mx_lut_e4m3_64x64.h",
        "4fee567ca604c0245dd690d7fd9b55234f135364bc7699144e00c810f6418351",
        "35b5dad4bdc4a5421a96f38abe1cea1c587305081dd7002213f582cba66b5e5c"),
    "e5m2": Case(
        "MxE5M2GemminiRocketConfig",
        "matmul_tiled_fp8_e5m2_64x64.c",
        "matmul_data_mx_lut_e5m2_64x64.h",
        "b3d523976586bddd742f1856bf060a71f75c804c0fe178a6090e1065d40864f0",
        "45469edcf03546c70772f3eef8fc90b679c2728f4207116d27e1e685070c454c"),
    "e4m3_dim32_64": Case(
        "MxDim32AllGemminiRocketConfig",
        "matmul_tiled_fp8_e4m3_lut_64x64_dim32.c",
        "matmul_data_mx_lut_e4m3_64x64_dim32.h",
        "c1ca615dfc976af1a2ee55a0aaf8460180106d1bef6bc709b52a5f0d87b1f6c4",
        "cd913cb3a83a9946833363eb9bee31a33d952362af7b4aa8f437e5fa9f5bc3f2",
        32),
    "e4m3_dim8_64": Case(
        "MxDim8AllGemminiRocketConfig",
        "matmul_tiled_fp8_e4m3_lut_64x64_nonrequant_dim8.c",
        "matmul_data_mx_lut_e4m3_64x64_dim8.h",
        "a871838208da00e9d4b55b653d195730092f5098984764c8beaa80f43d43617a",
        "742d91da250434c15ea8ca9aa66f6bf2fed8b14047d0621b77d10e174125d569",
        8),
    "e4m3_dim8_128": Case(
        "MxDim8AllGemminiRocketConfig",
        "matmul_tiled_fp8_e4m3_lut_128x128_nonrequant_dim8.c",
        "matmul_data_mx_lut_e4m3_128x128_dim8.h",
        "b64b94df6ebe1b3642f57a1ff506943c099bcd418b2523ee85b65bb69ab5e31f",
        "05ab49d9b8d9ce68d64c5aaca404f037df285fb0de509b984eff8b4b472bdea8",
        8, "128x128"),
}
DEFAULT_CASES = ("e2m3", "e4m3", "e5m2")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(key: str, directory: Path, profile_path: Path,
            rtl_root: Path, compiler_revision: str) -> dict:
    case = CASES[key]
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    dispatch = directory / "object/compile_manifest.json"
    manifest_path = directory / "object/object_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    obj = directory / "object/mx_issue.o"
    elf = directory / "physical/asymmetric_program.elf"
    log = directory / "physical/spike.log"
    physical = directory / "object/physical_program.json"
    profile = load_profile(profile_path, rtl_root=rtl_root)
    side = int(case.source_shape.split("x")[0])
    expected_outputs = side * side
    if (receipt.get("status") != "source_golden_matched_on_pinned_spike" or
            receipt.get("spike_exit_code") != 0 or
            receipt.get("compared_bf16_outputs") != expected_outputs or
            receipt.get("source_driver_sha256") != case.source_sha256 or
            receipt.get("source_header_sha256") != case.header_sha256 or
            receipt.get("compiler_revision") != compiler_revision or
            receipt.get("rtl_revision") != RTL_REVISION or
            receipt.get("model2mlir_revision") != MODEL2MLIR_REVISION or
            receipt.get("mxq_revision") != MXQ_REVISION or
            receipt.get("profile_name") != case.profile or
            receipt.get("mesh_dim") != case.mesh_dim or
            receipt.get("profile_sha256") != profile_sha256(profile) or
            receipt.get("public_object_sha256") != _sha(obj) or
            receipt.get("public_object_dispatch_sha256") != _sha(dispatch) or
            receipt.get("public_object_manifest_sha256") != _sha(manifest_path) or
            receipt.get("physical_program_sha256") != _sha(physical) or
            receipt.get("elf_sha256") != _sha(elf) or
            receipt.get("spike_log_sha256") != _sha(log) or
            manifest.get("allocated_data_section_bytes") != 0 or
            manifest.get("embedded_operand_bytes") != 0 or
            manifest.get("embedded_golden_bytes") != 0 or
            "0 BF16 mismatches" not in log.read_text()):
        raise ValueError(f"Nicolas {key} LUT source lacks a complete public-object replay")
    return {
        "case": key,
        "source_program": case.source.removesuffix(".c"),
        "source_driver_sha256": case.source_sha256,
        "source_header_sha256": case.header_sha256,
        "profile_name": case.profile,
        "profile_sha256": receipt["profile_sha256"],
        "compared_bf16_outputs": expected_outputs,
        "receipt": f"{key}/receipt.json", "receipt_sha256": _sha(receipt_path),
        "object_sha256": _sha(obj), "elf_sha256": _sha(elf),
        "spike_log_sha256": _sha(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES,
                        help="source case to replay; default is the three DIM16 modes")
    parser.add_argument("--jobs", type=int, default=1)
    for name in ("model2mlir-root", "mxq-root", "rtl-root", "riscv-root",
                 "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    selected = args.case or list(DEFAULT_CASES)
    if len(selected) != len(set(selected)):
        parser.error("each Nicolas LUT source case must be selected once")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    rtl_root = args.rtl_root.resolve()
    model2mlir_root = args.model2mlir_root.resolve()
    mxq_root = args.mxq_root.resolve()
    if (_revision(rtl_root) != RTL_REVISION or
            _revision(model2mlir_root) != MODEL2MLIR_REVISION or
            _revision(mxq_root) != MXQ_REVISION):
        raise ValueError("Nicolas LUT suite needs pinned RTL, model2MLIR, and MXQuant")
    software = rtl_root / "software/gemmini-rocc-tests"
    for key in selected:
        case = CASES[key]
        if (_sha(software / "bareMetalC" / case.source) != case.source_sha256 or
                _sha(software / "include" / case.header) != case.header_sha256):
            raise ValueError(f"Nicolas LUT source/header differs: {case.source}")
    compiler_revision = _revision(ROOT)
    args.out_dir.mkdir(parents=True)
    index = {
        "schema": SCHEMA, "status": "running",
        "scope": "selected pinned same-format LUT C numerical outputs via fresh model2MLIR and public RV64 objects",
        "compiler_revision": compiler_revision, "rtl_revision": RTL_REVISION,
        "model2mlir_revision": MODEL2MLIR_REVISION, "mxq_revision": MXQ_REVISION,
        "selected_cases": selected, "cases": [],
    }

    def write_index() -> None:
        (args.out_dir / "index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n")

    def replay(key: str) -> tuple[str, int]:
        case = CASES[key]
        profile = ROOT / "profiles/gemmini-mx-cleanup-266c593" / f"{case.profile}.json"
        command = [
            sys.executable, "-m", "tools.qualify_nicolas_asym",
            "--public-object", "--symmetric-lut", key.split("_")[0],
            "--mesh-dim", str(case.mesh_dim),
            "--source-shape", case.source_shape,
            "--model2mlir-root", str(model2mlir_root),
            "--mxq-root", str(mxq_root), "--rtl-root", str(rtl_root),
            "--profile", str(profile), "--riscv-root", str(args.riscv_root.resolve()),
            "--mx-opt", str(args.mx_opt.resolve()),
            "--out-dir", str(args.out_dir / key),
        ]
        run = subprocess.run(command, cwd=ROOT, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             check=False)
        (args.out_dir / f"{key}.log").write_text(run.stdout)
        return key, run.returncode

    write_index()
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for key, returncode in executor.map(replay, selected):
            if returncode:
                index["status"], index["failed_case"] = "failed", key
                write_index()
                raise SystemExit(f"{key} failed; see {args.out_dir / (key + '.log')}")
            case = CASES[key]
            profile = ROOT / "profiles/gemmini-mx-cleanup-266c593" / f"{case.profile}.json"
            try:
                record = _record(key, args.out_dir / key, profile, rtl_root,
                                 compiler_revision)
            except (ValueError, KeyError, FileNotFoundError):
                index["status"], index["failed_case"] = "failed", key
                write_index()
                raise
            index["cases"].append(record)
            write_index()
            print(f"matched {key}: {record['compared_bf16_outputs']} BF16 outputs", flush=True)
    index["status"] = ("all_three_source_goldens_matched_on_pinned_spike"
                       if selected == list(DEFAULT_CASES) else
                       "all_selected_source_goldens_matched_on_pinned_spike")
    index["matched_sources"] = len(index["cases"])
    index["total_bf16_outputs_checked"] = sum(
        row["compared_bf16_outputs"] for row in index["cases"])
    write_index()
    print(f"matched {index['matched_sources']} LUT sources and "
          f"{index['total_bf16_outputs_checked']} BF16 outputs", flush=True)


if __name__ == "__main__":
    main()
