"""Reproduce all 31 Radiance MX GEMM source goldens from one command.

This invokes the source header materializer, current model2MLIR capture,
physical MX lowering, RV64 build, and Nicolas Spike comparison. The result
must match every archived frontend, physical, and simulator artifact digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "source-root", "rtl-root",
                 "riscv-root", "mx-opt", "radiance-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--compatible-source-revision", action="store_true",
                        help="allow a different Radiance revision only with identical driver, "
                             "header, frontend, physical, ELF, and Spike artifacts")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4")
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    baseline = args.baseline.resolve()
    expected_front = _read(baseline / "frontend/index.json")
    expected_spike = _read(baseline / "spike/index.json")
    if (expected_front.get("captured_drivers") != 31 or
            expected_spike.get("covered_drivers") != 31 or
            len(expected_front.get("rows", [])) != 31 or
            len(expected_spike.get("rows", [])) != 31 or
            expected_front["source_revision"] != expected_spike["source_revision"] or
            expected_front["rtl_revision"] != expected_spike["rtl_revision"]):
        raise ValueError("baseline is not the complete Radiance MX GEMM roster")
    for path, revision in (
            (args.model2mlir_root, expected_front["model2mlir_revision"]),
            (args.mxq_root, expected_front["mxq_revision"]),
            (args.rtl_root, expected_front["rtl_revision"])):
        if _revision(path) != revision:
            raise ValueError(f"selected source revision differs from baseline: {path}")
    if (_revision(args.source_root) != expected_front["source_revision"] and
            not args.compatible_source_revision):
        raise ValueError("selected Radiance revision differs from baseline")
    for path in (args.mx_opt, args.radiance_opt,
                 args.riscv_root / "bin/spike"):
        if not path.is_file():
            raise ValueError(f"required executable is absent: {path}")
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    source = args.source_root.resolve()
    rtl = args.rtl_root.resolve()
    print("building Radiance golden tool", flush=True)
    subprocess.run(["make", "-C", str(source / "lib/golden"), "mx_golden"],
                   check=True)
    print("materializing and checking 31 source driver headers", flush=True)
    materialize = [
        sys.executable, "-m", "tools.materialize_radiance_roster_headers",
        "--source-root", str(source),
        "--expected-index", str(baseline / "frontend/index.json"),
        "--out", str(out / "headers.json"),
    ]
    if args.compatible_source_revision:
        materialize.append("--compatible-source-revision")
    subprocess.run(materialize, cwd=ROOT, check=True)
    print("capturing 31 PyTorch/model2MLIR contractions", flush=True)
    subprocess.run([
        sys.executable, "-m", "tools.recapture_radiance_roster",
        "--model2mlir-root", str(args.model2mlir_root.resolve()),
        "--mxq-root", str(args.mxq_root.resolve()),
        "--source-root", str(source), "--rtl-root", str(rtl),
        "--mx-opt", str(args.mx_opt.resolve()),
        "--radiance-opt", str(args.radiance_opt.resolve()),
        "--out-dir", str(out / "frontend"),
    ], cwd=ROOT, check=True)
    print("compiling and comparing all source outputs on Spike", flush=True)
    subprocess.run([
        sys.executable, "-m", "tools.requalify_radiance_roster",
        "--capture-root", str(out / "frontend"),
        "--source-root", str(source), "--rtl-root", str(rtl),
        "--riscv-root", str(args.riscv_root.resolve()),
        "--out-dir", str(out / "spike"), "--jobs", str(args.jobs),
    ], cwd=ROOT, check=True)
    front = _read(out / "frontend/index.json")
    spike = _read(out / "spike/index.json")
    if (front["captured_drivers"] != 31 or spike["covered_drivers"] != 31 or
            len(front["rows"]) != 31 or len(spike["rows"]) != 31 or
            spike["frontend_index_sha256"] != _sha(out / "frontend/index.json")):
        raise ValueError("new frontend and Spike indices are incomplete")
    for selected, expected in zip(front["rows"], expected_front["rows"]):
        if selected["driver"] != expected["driver"]:
            raise ValueError("source driver roster order differs")
        for key in ("driver_sha256", "header_sha256", "shape_mnk", "tile_mnk",
                    "precision", "quant_output", "profile_sha256",
                    "source_mlir_sha256", "bound_mlir_sha256"):
            if selected[key] != expected[key]:
                raise ValueError(f"frontend artifact drift: {selected['driver']}: {key}")
        current_capture = _read(out / "frontend" / selected["receipt"])
        baseline_capture = _read(baseline / "frontend" / expected["receipt"])
        for key in ("source_generator_sha256", "source_layout", "source_plan_error"):
            if current_capture[key] != baseline_capture[key]:
                raise ValueError(f"source planning drift: {selected['driver']}: {key}")
    for selected, expected in zip(spike["rows"], expected_spike["rows"]):
        if selected["driver"] != expected["driver"]:
            raise ValueError("Spike driver roster order differs")
        for key in ("source_driver_sha256", "source_header_sha256",
                    "profile_bound_mlir_sha256", "payload_bound_mlir_sha256",
                    "elf_sha256", "spike_log_sha256", "comparison",
                    "compared_count", "status"):
            if selected[key] != expected[key]:
                raise ValueError(f"numerical artifact drift: {selected['driver']}: {key}")
        actual_receipt = _read(out / "spike" / selected["receipt"])
        old_receipt = _read(baseline / "spike" / expected["receipt"])
        for key in ("files_sha256", "object_sha256", "extension_sha256",
                    "elf_sha256", "spike_log_sha256", "bound_mlir_sha256"):
            if actual_receipt[key] != old_receipt[key]:
                raise ValueError(f"compiled artifact drift: {selected['driver']}: {key}")
    result = {
        "schema": "mx_gemmini.radiance_mx_gemm_reproduction.v1",
        "status": "all_31_source_goldens_reproduced_on_pinned_spike",
        "baseline_frontend_index_sha256": _sha(baseline / "frontend/index.json"),
        "baseline_spike_index_sha256": _sha(baseline / "spike/index.json"),
        "frontend_index_sha256": _sha(out / "frontend/index.json"),
        "spike_index_sha256": _sha(out / "spike/index.json"),
        "compiler_revision": _revision(ROOT),
        "model2mlir_revision": front["model2mlir_revision"],
        "source_revision": front["source_revision"],
        "baseline_source_revision": expected_front["source_revision"],
        "compatible_source_revision": (
            front["source_revision"] != expected_front["source_revision"]),
        "rtl_revision": front["rtl_revision"],
        "covered_drivers": 31,
        "fullout_drivers": 23,
        "requant_drivers": 8,
    }
    (out / "reproduction.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"reproduced all 31 Radiance MX GEMM source goldens -> {out / 'reproduction.json'}")


if __name__ == "__main__":
    main()
