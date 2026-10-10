"""Build and run every captured Radiance MX GEMM source case on pinned Spike.

The input is the complete frontend index from recapture_radiance_roster.py.
Each case compiles its own source-bound MLIR and compares the full output to
the source header through the existing standalone compiler and checker.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.recapture_radiance_roster import PROFILE_BY_PRECISION, PROFILE_DIR, ROOT


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-root", "source-root", "rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 4:
        parser.error("--jobs must be between 1 and 4")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    capture_root = args.capture_root.resolve()
    source_root = args.source_root.resolve()
    index_path = capture_root / "index.json"
    frontend = json.loads(index_path.read_text())
    rows = frontend.get("rows")
    if (frontend.get("schema") != "mx_gemmini.radiance_frontend_roster.v1" or
            not isinstance(rows, list) or len(rows) != frontend.get("captured_drivers") or
            frontend.get("source_revision") != _revision(source_root) or
            frontend.get("rtl_revision") != _revision(args.rtl_root)):
        raise ValueError("frontend index differs from selected source or RTL")
    profiles = {precision: load_profile(PROFILE_DIR / name, rtl_root=args.rtl_root)
                for precision, name in PROFILE_BY_PRECISION.items()}
    args.out_dir.mkdir(parents=True)

    def qualify(row: dict) -> dict:
        name = row["driver"]
        driver = source_root / name
        kernel = read_source_gemm(driver)
        case = args.out_dir / driver.stem
        captured = capture_root / driver.stem / "mx_gemm.profile_bound.mlir"
        if (not kernel.data_header_present or
                row["driver_sha256"] != _sha(driver) or
                row["header_sha256"] != _sha(kernel.data_header) or
                row["shape_mnk"] != list(kernel.shape) or
                row["tile_mnk"] != list(kernel.tile) or
                row["precision"] != kernel.datatype or
                row["quant_output"] != kernel.quant_output or
                row["bound_mlir_sha256"] != _sha(captured) or
                row["profile_sha256"] != profile_sha256(profiles[kernel.datatype])):
            raise ValueError(f"captured frontend differs from selected source: {name}")
        command = [
            sys.executable, "-m", "tools.qualify_source_mx",
            "--mlir", str(captured), "--driver", str(driver),
            "--profile", str(PROFILE_DIR / PROFILE_BY_PRECISION[kernel.datatype]),
            "--rtl-root", str(args.rtl_root.resolve()),
            "--riscv-root", str(args.riscv_root.resolve()),
            "--out-dir", str(case),
        ]
        if kernel.quant_output:
            command.append("--source-header-quantized")
        run = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
        case.mkdir(parents=True, exist_ok=True)
        (case / "qualification.log").write_text(run.stdout)
        if run.returncode:
            raise RuntimeError(f"MX qualification failed for {name}; see {case / 'qualification.log'}")
        receipt_path = case / "build/artifact_manifest.json"
        receipt = json.loads(receipt_path.read_text())
        expected_status = ("radiance_header_matched_on_pinned_spike" if kernel.quant_output
                           else "source_golden_matched_on_pinned_spike")
        comparison = ("compared_source_fp6_packed_bytes" if
                      kernel.datatype == "FP6" and kernel.quant_output else
                      "compared_source_fp8_codes" if kernel.quant_output else
                      "compared_bf16_outputs")
        m, n, _ = kernel.shape
        expected_count = m * n // (2 if kernel.datatype == "FP6" and
                                    kernel.quant_output else 1)
        if (receipt.get("status") != expected_status or
                receipt.get("spike_exit_code") != 0 or
                receipt.get("compiler_revision") != _revision(ROOT) or
                receipt.get("rtl_revision") != frontend["rtl_revision"] or
                receipt.get("source_driver_sha256") != row["driver_sha256"] or
                receipt.get("source_header_sha256") != row["header_sha256"] or
                receipt.get("profile_sha256") != row["profile_sha256"] or
                receipt.get("shape_mnk") != row["shape_mnk"] or
                receipt.get(comparison) != expected_count or
                receipt.get("elf_sha256") != _sha(case / "build/mx_program.elf") or
                receipt.get("spike_log_sha256") != _sha(case / "build/spike.log")):
            raise ValueError(f"MX numerical receipt differs from selected source: {name}")
        if kernel.quant_output and receipt.get("compared_source_e8m0_scales") != m * n // 32:
            raise ValueError(f"MX output scale comparison is incomplete: {name}")
        return {
            "driver": name, "precision": kernel.datatype,
            "shape_mnk": list(kernel.shape), "quant_output": kernel.quant_output,
            "source_driver_sha256": row["driver_sha256"],
            "source_header_sha256": row["header_sha256"],
            "frontend_receipt_sha256": row["receipt_sha256"],
            "profile_bound_mlir_sha256": row["bound_mlir_sha256"],
            "payload_bound_mlir_sha256": receipt["bound_mlir_sha256"],
            "receipt": str(receipt_path.relative_to(args.out_dir)),
            "receipt_sha256": _sha(receipt_path),
            "comparison": comparison, "compared_count": expected_count,
            **({"compared_source_e8m0_scales": m * n // 32}
               if kernel.quant_output else {}),
            "elf_sha256": receipt["elf_sha256"],
            "spike_log_sha256": receipt["spike_log_sha256"],
            "status": expected_status,
        }

    completed = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(qualify, row): row["driver"] for row in rows}
        for future in as_completed(futures):
            result = future.result()
            completed[result["driver"]] = result
            print(f"qualified {len(completed)}/{len(rows)} {result['driver']}", flush=True)
    output = {
        "schema": "mx_gemmini.radiance_mx_gemm_source_parity.v2",
        "source_revision": frontend["source_revision"],
        "model2mlir_revision": frontend["model2mlir_revision"],
        "rtl_revision": frontend["rtl_revision"],
        "compiler_revision": _revision(ROOT),
        "frontend_index_sha256": _sha(index_path),
        "covered_drivers": len(rows),
        "fullout_drivers": sum(not row["quant_output"] for row in rows),
        "requant_drivers": sum(row["quant_output"] for row in rows),
        "rows": [completed[row["driver"]] for row in rows],
    }
    (args.out_dir / "index.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"qualified {len(rows)} MX GEMM drivers -> {args.out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
