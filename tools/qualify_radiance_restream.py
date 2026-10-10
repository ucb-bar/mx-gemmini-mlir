"""Qualify Radiance's committed four-M-block MX re-stream baseline on Spike."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.qualify_radiance_ws_roster import (
    MODEL2MLIR_REVISION, MXQ_REVISION, PROFILE, ROOT, RTL_REVISION,
    _read, _revision, _run, _semantic_manifest_sha, _sha,
)
from tools.materialize_radiance_ws_data import SOURCE_REVISION


DRIVER_SHA256 = "02a902eff587326a883c77313bf670d1a5fafdc72f4e3aac66f76717a4836a05"
DATA_SHA256 = "d3f6e85821d863af2b58ae5aaf60d983bcb4e81689aff0dd9df9630bb2e85ba0"


def prove_restream_weight_reads(program: dict) -> dict:
    """Prove four complete B reads, one per M=64 output tile."""
    if (program["shape_mnk"] != [256, 64, 2048] or
            [tile["m_start"] for tile in program["plan"].get("output_tiles", [])]
            != [0, 64, 128, 192]):
        raise ValueError("physical re-stream output tiling differs from source")
    transfers: list[int] = []
    for step in program["steps"]:
        command = step["command"]
        operand = command.get("rs1") or {}
        if operand.get("buffer") != "weight":
            continue
        if step["phase"] != "move_weight" or command.get("funct") != 2:
            raise ValueError("re-stream weight transfer is not a DMA command")
        rs2 = command["rs2"]["immediate"]
        if ((rs2 >> 48 & 0xffff), (rs2 >> 32 & 0xffff)) != (16, 16):
            raise ValueError("re-stream weight transfer differs from 16x16 byte tile")
        transfers.append(operand["byte_offset"])
    expected = {row * 64 + col for row in range(0, 2048, 16)
                for col in range(0, 64, 16)}
    if (len(expected) != 512 or len(transfers) != 4 * len(expected) or
            any(set(transfers[tile * 512:(tile + 1) * 512]) != expected
                for tile in range(4)) or
            set(Counter(transfers).values()) != {4}):
        raise ValueError("physical re-stream B reads differ from four complete passes")
    return {"m_tiles": 4, "k_waves_per_m_tile": 32,
            "weight_tile_reads": 128,
            "weight_dma_commands": len(transfers),
            "weight_bytes_per_pass": 2048 * 64,
            "weight_bytes_transferred": 4 * 2048 * 64}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "source-root", "rtl-root",
                 "riscv-root", "mx-opt", "radiance-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--baseline-index", type=Path)
    args = parser.parse_args()
    for name in ("model2mlir_root", "mxq_root", "source_root", "rtl_root",
                 "riscv_root", "mx_opt", "radiance_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    for path, expected in ((args.source_root, SOURCE_REVISION),
                           (args.model2mlir_root, MODEL2MLIR_REVISION),
                           (args.mxq_root, MXQ_REVISION),
                           (args.rtl_root, RTL_REVISION)):
        if _revision(path) != expected:
            raise ValueError(f"selected source differs from pinned re-stream qualification: {path}")
    for path in (args.mx_opt, args.radiance_opt,
                 args.riscv_root / "bin/spike",
                 args.riscv_root / "bin/riscv64-unknown-elf-gcc"):
        if not path.is_file():
            raise ValueError(f"required executable is absent: {path}")
    profile = load_profile(PROFILE, rtl_root=args.rtl_root)
    profile_digest = profile_sha256(profile)
    driver = args.source_root / "kernels/gemm_mxgemmini_ws_restream/kernel.cpp"
    kernel = read_source_gemm(driver)
    if (_sha(driver), _sha(kernel.data_header), kernel.shape, kernel.tile,
            kernel.datatype) != (DRIVER_SHA256, DATA_SHA256,
                                (256, 64, 2048), (64, 64, 64), "FP8"):
        raise ValueError("re-stream source driver or committed data differs")
    args.out_dir.mkdir(parents=True)
    frontend = args.out_dir / "frontend"
    _run([sys.executable, str(ROOT / "tests/capture_radiance_mx_gemm.py"),
          "--model2mlir-root", str(args.model2mlir_root),
          "--mxq-root", str(args.mxq_root),
          "--source-root", str(args.source_root),
          "--driver", str(driver.relative_to(args.source_root)),
          "--out", str(frontend), "--mx-opt", str(args.mx_opt),
          "--radiance-opt", str(args.radiance_opt),
          "--profile", str(PROFILE), "--rtl-root", str(args.rtl_root)],
         cwd=ROOT, log=args.out_dir / "capture.log")
    capture = _read(frontend / "receipt.json")
    bound = frontend / "mx_gemm.profile_bound.mlir"
    if (capture["status"] != "source_shape_frontend_handoff_only" or
            capture["source_driver_sha256"] != DRIVER_SHA256 or
            capture["source_data_header_sha256"] != DATA_SHA256 or
            capture["target_binding"]["profile_sha256"] != profile_digest or
            capture["target_binding"]["bound_mlir_sha256"] != _sha(bound) or
            capture["opaque_calls"]):
        raise ValueError("re-stream model2MLIR capture differs from source")
    spike = args.out_dir / "spike"
    _run([sys.executable, "-m", "tools.qualify_source_mx",
          "--mlir", str(bound), "--driver", str(driver),
          "--profile", str(PROFILE), "--rtl-root", str(args.rtl_root),
          "--riscv-root", str(args.riscv_root), "--out-dir", str(spike)],
         cwd=ROOT, log=args.out_dir / "qualification.log")
    manifest = _read(spike / "build/artifact_manifest.json")
    program_path = spike / "build/physical_program.json"
    traffic = prove_restream_weight_reads(_read(program_path))
    if (manifest["status"] != "source_golden_matched_on_pinned_spike" or
            manifest["spike_exit_code"] != 0 or
            manifest["compared_bf16_outputs"] != 16384 or
            manifest["source_driver_sha256"] != DRIVER_SHA256 or
            manifest["source_header_sha256"] != DATA_SHA256 or
            manifest["profile_sha256"] != profile_digest or
            manifest["rtl_revision"] != RTL_REVISION):
        raise ValueError("re-stream numerical result differs from source golden")
    index = {"schema": "mx_gemmini.radiance_restream_source_parity.v1",
             "status": "four_m_tile_restream_source_golden_matched_on_pinned_spike",
             "source_revision": SOURCE_REVISION,
             "model2mlir_revision": MODEL2MLIR_REVISION,
             "mxq_revision": MXQ_REVISION, "rtl_revision": RTL_REVISION,
             "compiler_revision": _revision(ROOT), "profile_sha256": profile_digest,
             "driver_sha256": DRIVER_SHA256, "data_sha256": DATA_SHA256,
             "shape_mnk": [256, 64, 2048], "tile_mnk": [64, 64, 64],
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
             "compared_bf16_outputs": 16384, **traffic}
    if args.baseline_index and index != _read(args.baseline_index):
        raise ValueError("re-stream frontend, physical program, or Spike artifact drift")
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"qualified re-stream MX source: 16384 BF16 outputs, "
          f"{traffic['weight_tile_reads']} B tile reads -> {args.out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
