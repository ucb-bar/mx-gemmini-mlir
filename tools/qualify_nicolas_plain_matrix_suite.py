"""Replay every qualified direct Nicolas matrix case through the public object CLI.

This covers the checked-in source matrix results named in CASES. It does not
compile performance instrumentation or the other Nicolas C programs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_nicolas_plain_matrix_object import (
    CASES, MODEL2MLIR_REVISION, MXQ_REVISION, ROOT, RTL_REVISION, _revision,
)


SCHEMA = "mx_gemmini.nicolas_plain_matrix_object_suite.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(case_key: str, directory: Path, profile_path: Path,
            rtl_root: Path, compiler_revision: str) -> dict:
    case = CASES[case_key]
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    profile = load_profile(profile_path, rtl_root=rtl_root)
    m, n, _ = case.shape
    code_bytes = m * n // 2 if case.precision in {"FP4", "FP6"} else m * n
    expected_outputs = (code_bytes + m * n // 32 if case.quant_output else m * n)
    if (receipt.get("case") != case.key or
            receipt.get("status") != "source_golden_matched_on_pinned_spike" or
            receipt.get("mismatches") != 0 or
            receipt.get("outputs_checked") != expected_outputs or
            receipt.get("source_driver_sha256") != case.source_sha256 or
            receipt.get("source_header_sha256") != case.header_sha256 or
            receipt.get("compiler_revision") != compiler_revision or
            receipt.get("model2mlir_revision") != MODEL2MLIR_REVISION or
            receipt.get("mxq_revision") != MXQ_REVISION or
            receipt.get("rtl_revision") != RTL_REVISION or
            receipt.get("profile_name") != case.profile_name or
            receipt.get("profile_sha256") != profile_sha256(profile) or
            receipt.get("mesh_dim") != profile["geometry"]["mesh_columns"] or
            receipt.get("object_sha256") != _sha(directory / "object/mx_issue.o") or
            receipt.get("elf_sha256") != _sha(directory / "run/mx_program.elf") or
            receipt.get("spike_log_sha256") != _sha(directory / "run/spike.log")):
        raise ValueError(f"Nicolas suite case {case.key} lacks a complete checked replay")
    if case.quant_output:
        expected_packed = code_bytes if case.precision in {"FP4", "FP6"} else None
        expected_codes = code_bytes if case.precision == "FP8" else None
        if (receipt.get("packed_bytes_checked") != expected_packed or
                receipt.get("codes_checked") != expected_codes or
                receipt.get("scales_checked") != m * n // 32):
            raise ValueError(f"Nicolas suite case {case.key} checked the wrong quantized extent")
    return {
        "case": case.key,
        "precision": case.precision,
        "shape_mnk": list(case.shape),
        "quant_output": case.quant_output,
        "mesh_dim": receipt["mesh_dim"],
        "profile_name": case.profile_name,
        "profile_sha256": receipt["profile_sha256"],
        "source_driver_sha256": case.source_sha256,
        "source_header_sha256": case.header_sha256,
        "outputs_checked": expected_outputs,
        "receipt": f"{case.key}/receipt.json",
        "receipt_sha256": _sha(receipt_path),
        "object_sha256": receipt["object_sha256"],
        "elf_sha256": receipt["elf_sha256"],
        "spike_log_sha256": receipt["spike_log_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES,
                        help="case to replay; omit to run every direct case")
    parser.add_argument("--jobs", type=int, default=1,
                        help="independent source cases to replay concurrently")
    for name in ("model2mlir-root", "mxq-root", "rtl-root", "riscv-root",
                 "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    selected = args.case or list(CASES)
    if len(selected) != len(set(selected)):
        parser.error("each Nicolas source case must be selected once")
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
        raise ValueError("Nicolas suite needs pinned RTL, model2MLIR, and MXQuant revisions")
    compiler_revision = _revision(ROOT)
    args.out_dir.mkdir(parents=True)
    index = {
        "schema": SCHEMA,
        "status": "running",
        "scope": "direct Nicolas matrix numerical outputs through public typed object compiler",
        "compiler_revision": compiler_revision,
        "rtl_revision": RTL_REVISION,
        "model2mlir_revision": MODEL2MLIR_REVISION,
        "mxq_revision": MXQ_REVISION,
        "selected_cases": selected,
        "cases": [],
    }

    def write_index() -> None:
        (args.out_dir / "index.json").write_text(json.dumps(index, indent=2,
                                                          sort_keys=True) + "\n")

    write_index()

    def replay(key: str) -> tuple[str, int]:
        case = CASES[key]
        profile = ROOT / "profiles/gemmini-mx-cleanup-266c593" / f"{case.profile_name}.json"
        directory = args.out_dir / key
        command = [
            sys.executable, "-m", "tools.qualify_nicolas_plain_matrix_object",
            "--case", key,
            "--model2mlir-root", str(model2mlir_root),
            "--mxq-root", str(mxq_root),
            "--rtl-root", str(rtl_root),
            "--profile", str(profile),
            "--riscv-root", str(args.riscv_root.resolve()),
            "--mx-opt", str(args.mx_opt.resolve()),
            "--out-dir", str(directory),
        ]
        run = subprocess.run(command, cwd=ROOT, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             check=False)
        (args.out_dir / f"{key}.log").write_text(run.stdout)
        return key, run.returncode

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for key, returncode in executor.map(replay, selected):
            case = CASES[key]
            profile = ROOT / "profiles/gemmini-mx-cleanup-266c593" / f"{case.profile_name}.json"
            directory = args.out_dir / key
            if returncode:
                index["status"] = "failed"
                index["failed_case"] = key
                write_index()
                raise SystemExit(f"{key} failed; see {args.out_dir / (key + '.log')}")
            try:
                record = _record(key, directory, profile, rtl_root, compiler_revision)
            except (ValueError, KeyError, FileNotFoundError):
                index["status"] = "failed"
                index["failed_case"] = key
                write_index()
                raise
            index["cases"].append(record)
            write_index()
            print(f"matched {key}: {record['outputs_checked']} outputs", flush=True)
    index["status"] = "all_selected_source_goldens_matched_on_pinned_spike"
    index["total_outputs_checked"] = sum(row["outputs_checked"] for row in index["cases"])
    write_index()
    print(f"matched {len(selected)} cases and {index['total_outputs_checked']} outputs", flush=True)


if __name__ == "__main__":
    main()
