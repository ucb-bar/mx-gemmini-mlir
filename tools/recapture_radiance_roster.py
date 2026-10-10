"""Recapture every Radiance MX GEMM driver with a selected model2MLIR checkout.

Each case uses tests/capture_radiance_mx_gemm.py, which captures PyTorch,
checks the one selected contraction, and verifies the MX and Radiance dialect
handoffs. This command records frontend coverage; numerical execution is a
separate qualification step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "profiles/gemmini-mx-cleanup-266c593"
PROFILE_BY_PRECISION = {
    "FP8": "MxE4M3Fp4VpuGemminiRocketConfig.json",
    "FP4": "MxE4M3Fp4VpuGemminiRocketConfig.json",
    "FP6": "MxE3M2OnlyGemminiRocketConfig.json",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "source-root", "rtl-root",
                 "mx-opt", "radiance-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    source = args.source_root.resolve()
    drivers = sorted((source / "kernels/gemm_mxgemmini").glob("mxgemm.fp*.cpp"))
    if not drivers:
        parser.error("selected radiance-kernels checkout has no MX GEMM drivers")
    profiles = {precision: load_profile(PROFILE_DIR / name, rtl_root=args.rtl_root)
                for precision, name in PROFILE_BY_PRECISION.items()}
    args.out_dir.mkdir(parents=True)
    rows = []
    for driver in drivers:
        kernel = read_source_gemm(driver)
        if not kernel.data_header_present:
            raise ValueError(f"source header is missing for {driver.name}")
        profile_path = PROFILE_DIR / PROFILE_BY_PRECISION[kernel.datatype]
        case = args.out_dir / driver.stem
        command = [
            sys.executable, str(ROOT / "tests/capture_radiance_mx_gemm.py"),
            "--model2mlir-root", str(args.model2mlir_root.resolve()),
            "--mxq-root", str(args.mxq_root.resolve()),
            "--source-root", str(source),
            "--driver", str(driver.relative_to(source)),
            "--out", str(case),
            "--mx-opt", str(args.mx_opt.resolve()),
            "--radiance-opt", str(args.radiance_opt.resolve()),
            "--profile", str(profile_path),
            "--rtl-root", str(args.rtl_root.resolve()),
        ]
        if kernel.datatype == "FP6":
            command += ["--policy", str(ROOT / "examples/fp6-source-line0-policy.yaml")]
        result = subprocess.run(command, cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                check=False)
        if result.returncode:
            case.mkdir(parents=True, exist_ok=True)
            (case / "capture_failure.log").write_text(result.stdout)
            raise RuntimeError(f"frontend capture failed for {driver.name}; see capture_failure.log")
        receipt_path = case / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        target = receipt.get("target_binding") or {}
        if (receipt.get("status") != "source_shape_frontend_handoff_only" or
                receipt.get("source_revision") != _revision(source) or
                receipt.get("model2mlir_revision") != _revision(args.model2mlir_root) or
                receipt.get("source_driver_sha256") != _sha(driver) or
                receipt.get("source_data_header_sha256") != _sha(kernel.data_header) or
                receipt.get("opaque_calls") or
                target.get("profile_sha256") != profile_sha256(profiles[kernel.datatype]) or
                target.get("bound_mlir_sha256") != _sha(case / "mx_gemm.profile_bound.mlir")):
            raise ValueError(f"frontend capture receipt differs for {driver.name}")
        rows.append({
            "driver": str(driver.relative_to(source)),
            "driver_sha256": _sha(driver),
            "header_sha256": _sha(kernel.data_header),
            "shape_mnk": list(kernel.shape),
            "tile_mnk": list(kernel.tile),
            "precision": kernel.datatype,
            "quant_output": kernel.quant_output,
            "profile_sha256": target["profile_sha256"],
            "receipt": str(receipt_path.relative_to(args.out_dir)),
            "receipt_sha256": _sha(receipt_path),
            "source_mlir_sha256": _sha(case / "mx_gemm.model2mlir.mlir"),
            "bound_mlir_sha256": target["bound_mlir_sha256"],
            "site_id": receipt["selected_site"]["site_id"],
            "status": receipt["status"],
        })
        print(f"captured {len(rows)}/{len(drivers)} {driver.name}", flush=True)
    index = {
        "schema": "mx_gemmini.radiance_frontend_roster.v1",
        "source_revision": _revision(source),
        "model2mlir_revision": _revision(args.model2mlir_root),
        "mxq_revision": _revision(args.mxq_root),
        "rtl_revision": _revision(args.rtl_root),
        "compiler_revision": _revision(ROOT),
        "captured_drivers": len(rows),
        "fullout_drivers": sum(not row["quant_output"] for row in rows),
        "requant_drivers": sum(row["quant_output"] for row in rows),
        "scope": "PyTorch/model2MLIR structural captures; packed source operands and numerical Spike results are qualified separately",
        "rows": rows,
    }
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"captured {len(rows)} MX GEMM drivers -> {args.out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
