"""Capture four Radiance batched decode GEMMs and qualify full output on Spike."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.materialize_radiance_batched_gemv_data import CASES, materialize
from tools.qualify_radiance_ws_roster import (
    ROOT, PROFILE, MODEL2MLIR_REVISION, MXQ_REVISION, RTL_REVISION,
    _prove_weight_read_once, _read, _revision, _run, _semantic_manifest_sha, _sha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "source-root", "rtl-root",
                 "riscv-root", "mx-opt", "radiance-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--baseline-index", type=Path,
                        help="require the new capture, commands, object, ELF, and Spike log to match")
    args = parser.parse_args()
    for name in ("model2mlir_root", "mxq_root", "source_root", "rtl_root",
                 "riscv_root", "mx_opt", "radiance_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    for path, expected in ((args.model2mlir_root, MODEL2MLIR_REVISION),
                           (args.mxq_root, MXQ_REVISION),
                           (args.rtl_root, RTL_REVISION)):
        if _revision(path) != expected:
            raise ValueError(f"selected source revision differs from pinned MX profile: {path}")
    for path in (args.mx_opt, args.radiance_opt,
                 args.riscv_root / "bin/spike",
                 args.riscv_root / "bin/riscv64-unknown-elf-gcc"):
        if not path.is_file():
            raise ValueError(f"required executable is absent: {path}")
    source_report = materialize(args.source_root)
    profile_digest = profile_sha256(load_profile(PROFILE, rtl_root=args.rtl_root))
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "data_materialization.json").write_text(
        json.dumps(source_report, indent=2, sort_keys=True) + "\n")
    rows = []
    for directory, precision, m, driver_sha, data_sha in CASES:
        case = args.out_dir / directory
        case.mkdir()
        driver = args.source_root / "kernels" / directory / "kernel.cpp"
        kernel = read_source_gemm(driver)
        if (_sha(driver), _sha(kernel.data_header), kernel.shape,
                kernel.tile, kernel.datatype.lower()) != (
                    driver_sha, data_sha, (m, 128, 2048), (m, 128, 128), precision):
            raise ValueError(f"source data differs from pinned batched GEMV: {directory}")
        frontend = case / "frontend"
        _run([sys.executable, str(ROOT / "tests/capture_radiance_mx_gemm.py"),
              "--model2mlir-root", str(args.model2mlir_root),
              "--mxq-root", str(args.mxq_root),
              "--source-root", str(args.source_root),
              "--driver", str(driver.relative_to(args.source_root)),
              "--out", str(frontend), "--mx-opt", str(args.mx_opt),
              "--radiance-opt", str(args.radiance_opt),
              "--profile", str(PROFILE), "--rtl-root", str(args.rtl_root)],
             cwd=ROOT, log=case / "capture.log")
        capture = _read(frontend / "receipt.json")
        bound = frontend / "mx_gemm.profile_bound.mlir"
        if (capture["status"] != "source_shape_frontend_handoff_only" or
                capture["source_driver_sha256"] != driver_sha or
                capture["source_data_header_sha256"] != data_sha or
                capture["source_revision"] != source_report["source_revision"] or
                capture["model2mlir_revision"] != MODEL2MLIR_REVISION or
                capture["target_binding"]["profile_sha256"] != profile_digest or
                capture["target_binding"]["bound_mlir_sha256"] != _sha(bound) or
                capture["opaque_calls"]):
            raise ValueError(f"frontend capture differs from pinned source: {directory}")
        spike = case / "spike"
        _run([sys.executable, "-m", "tools.qualify_source_mx",
              "--mlir", str(bound), "--driver", str(driver),
              "--profile", str(PROFILE), "--rtl-root", str(args.rtl_root),
              "--riscv-root", str(args.riscv_root), "--out-dir", str(spike)],
             cwd=ROOT, log=case / "qualification.log")
        manifest = _read(spike / "build/artifact_manifest.json")
        program_path = spike / "build/physical_program.json"
        traffic = _prove_weight_read_once(
            _read(program_path), m=m, n=128, k=2048, tile_k=128,
            precision=precision)
        if (manifest["status"] != "source_golden_matched_on_pinned_spike" or
                manifest["spike_exit_code"] != 0 or
                manifest["compared_bf16_outputs"] != m * 128 or
                manifest["source_driver_sha256"] != driver_sha or
                manifest["source_header_sha256"] != data_sha or
                manifest["profile_sha256"] != profile_digest or
                manifest["rtl_revision"] != RTL_REVISION):
            raise ValueError(f"Spike comparison differs from pinned source: {directory}")
        rows.append({"directory": directory, "precision": precision,
                     "shape_mnk": [m, 128, 2048], "tile_mnk": list(kernel.tile),
                     "driver_sha256": driver_sha, "data_sha256": data_sha,
                     "frontend_mlir_sha256": _sha(frontend / "mx_gemm.model2mlir.mlir"),
                     "bound_mlir_sha256": _sha(bound),
                     "frontend_receipt_sha256": _sha(frontend / "receipt.json"),
                     "payload_bound_mlir_sha256": _sha(spike / "payload_bound.mlir"),
                     "physical_program_sha256": _sha(program_path),
                     "files_sha256": manifest["files_sha256"],
                     "object_sha256": manifest["object_sha256"],
                     "elf_sha256": manifest["elf_sha256"],
                     "extension_sha256": manifest["extension_sha256"],
                     "spike_log_sha256": manifest["spike_log_sha256"],
                     "spike_manifest_semantic_sha256": _semantic_manifest_sha(manifest),
                     "compared_bf16_outputs": manifest["compared_bf16_outputs"],
                     **traffic})
        print(f"qualified {directory}: {m * 128} BF16 outputs, "
              f"{traffic['weight_tile_reads']} unique K waves", flush=True)
    index = {"schema": "mx_gemmini.radiance_batched_gemv_roster.v1",
             "status": "four_source_goldens_matched_on_pinned_spike",
             "source_revision": source_report["source_revision"],
             "model2mlir_revision": MODEL2MLIR_REVISION,
             "mxq_revision": MXQ_REVISION, "rtl_revision": RTL_REVISION,
             "compiler_revision": _revision(ROOT), "profile_sha256": profile_digest,
             "data_materialization_sha256": _sha(args.out_dir / "data_materialization.json"),
             "rows": rows}
    if args.baseline_index:
        baseline = _read(args.baseline_index)
        if baseline != index:
            raise ValueError("batched GEMV source or compiled artifact drift")
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"qualified {len(rows)} Radiance batched GEMVs -> {args.out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
