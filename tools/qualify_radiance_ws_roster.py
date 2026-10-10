"""Capture and qualify Radiance's two read-once MX GEMMs on Nicolas's Spike."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.materialize_radiance_ws_data import CASES, materialize


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
MODEL2MLIR_REVISION = "e9ded36eb85abf2d9097ac4dc11457c825853388"
MXQ_REVISION = "b4af5430bac147f4a16126931cc0177367cc3982"
RTL_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _semantic_manifest_sha(manifest: dict) -> str:
    # Link logs include the chosen output directory. The issuer, objects, ELF,
    # extension, numerical log, and all source digests remain byte identical.
    stable = dict(manifest)
    stable.pop("build_log_sha256", None)
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def _run(command: list[str], *, cwd: Path, log: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _prove_weight_read_once(program: dict, *, m: int, n: int, k: int,
                            tile_k: int, precision: str) -> dict:
    """Check that the physical DMA trace covers every packed B block once."""
    if program["shape_mnk"] != [m, n, k] or m != 256 or n != 64 or tile_k != 64:
        raise ValueError("physical program differs from pinned read-once source shape")
    row_stride = n if precision == "fp8" else n // 2
    if row_stride % 16 or k % tile_k:
        raise ValueError("read-once B layout is not tiled into 16-byte columns")
    transfers: dict[int, list[int]] = {}
    for step in program["steps"]:
        command = step["command"]
        operand = command.get("rs1") or {}
        if operand.get("buffer") != "weight":
            continue
        if step["phase"] != "move_weight" or command.get("funct") != 2:
            raise ValueError("weight transfer is not a source-bound DMA command")
        shape = (command["rs2"]["immediate"] >> 48 & 0xffff,
                 command["rs2"]["immediate"] >> 32 & 0xffff)
        if shape != (16, 16):
            raise ValueError("weight transfer differs from the checked 16x16 byte tile")
        transfers.setdefault(step["wave"], []).append(operand["byte_offset"])
    if set(transfers) != set(range(k // tile_k)):
        raise ValueError("weight transfer wave set differs from source K loop")
    expected_all = set()
    per_wave = tile_k // 16 * (row_stride // 16)
    for wave in range(k // tile_k):
        expected = {row * row_stride + col
                    for row in range(wave * tile_k, (wave + 1) * tile_k, 16)
                    for col in range(0, row_stride, 16)}
        actual = transfers[wave]
        if len(actual) != per_wave or set(actual) != expected:
            raise ValueError(f"weight bytes are reloaded or omitted in wave {wave}")
        expected_all.update(expected)
    if len(expected_all) * 256 != k * row_stride:
        raise ValueError("weight transfer set does not cover the complete packed source")
    return {"k_waves": k // tile_k, "weight_dma_commands": len(expected_all),
            "weight_bytes_covered_once": k * row_stride,
            "weight_tile_reads": k // tile_k}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "source-root", "rtl-root",
                 "riscv-root", "mx-opt", "radiance-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--baseline-index", type=Path,
                        help="require the new captures, compiled files, ELFs, and Spike logs to match")
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
    profile = load_profile(PROFILE, rtl_root=args.rtl_root)
    profile_digest = profile_sha256(profile)
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "data_materialization.json").write_text(
        json.dumps(source_report, indent=2, sort_keys=True) + "\n")
    rows = []
    for directory, precision, shape, driver_sha, data_sha in CASES:
        case = args.out_dir / directory
        case.mkdir()
        driver = args.source_root / "kernels" / directory / "kernel.cpp"
        kernel = read_source_gemm(driver)
        if (_sha(driver), _sha(kernel.data_header), kernel.shape,
                kernel.datatype.lower()) != (driver_sha, data_sha, shape, precision):
            raise ValueError(f"source data differs from pinned read-once case: {directory}")
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
        manifest_path = spike / "build/artifact_manifest.json"
        manifest = _read(manifest_path)
        program_path = spike / "build/physical_program.json"
        weight = _prove_weight_read_once(
            _read(program_path), m=shape[0], n=shape[1], k=shape[2],
            tile_k=kernel.tile[2], precision=precision)
        if (manifest["status"] != "source_golden_matched_on_pinned_spike" or
                manifest["spike_exit_code"] != 0 or
                manifest["compared_bf16_outputs"] != shape[0] * shape[1] or
                manifest["source_driver_sha256"] != driver_sha or
                manifest["source_header_sha256"] != data_sha or
                manifest["profile_sha256"] != profile_digest or
                manifest["rtl_revision"] != RTL_REVISION):
            raise ValueError(f"Spike comparison differs from pinned source: {directory}")
        rows.append({"directory": directory, "precision": precision,
                     "shape_mnk": list(shape), "tile_mnk": list(kernel.tile),
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
                     **weight})
        print(f"qualified {directory}: {manifest['compared_bf16_outputs']} BF16 outputs, "
              f"{weight['weight_tile_reads']} unique weight tiles", flush=True)
    index = {"schema": "mx_gemmini.radiance_ws_read_once_roster.v1",
             "status": "two_read_once_source_goldens_matched_on_pinned_spike",
             "source_revision": source_report["source_revision"],
             "model2mlir_revision": MODEL2MLIR_REVISION,
             "mxq_revision": MXQ_REVISION, "rtl_revision": RTL_REVISION,
             "compiler_revision": _revision(ROOT), "profile_sha256": profile_digest,
             "data_materialization_sha256": _sha(args.out_dir / "data_materialization.json"),
             "rows": rows}
    if args.baseline_index:
        baseline = _read(args.baseline_index)
        if (baseline["schema"] != index["schema"] or
                baseline["source_revision"] != index["source_revision"] or
                baseline["model2mlir_revision"] != index["model2mlir_revision"] or
                baseline["rtl_revision"] != index["rtl_revision"] or
                len(baseline["rows"]) != len(index["rows"])):
            raise ValueError("reproduction baseline differs from pinned MX sources")
        for actual, expected in zip(index["rows"], baseline["rows"]):
            if actual != expected:
                raise ValueError(f"read-once source or compiled artifact drift: {actual['directory']}")
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"qualified {len(rows)} Radiance read-once MX kernels -> {args.out_dir / 'index.json'}")


if __name__ == "__main__":
    main()
