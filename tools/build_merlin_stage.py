"""Compile a receipt-bound model2MLIR stage with Merlin and the MX catalog.

This is an experimental whole-stage build/run probe. Its result is not a
quantized PyTorch parity certificate or a complete multi-stage session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _capture_model(stage: Path) -> Path:
    source = stage / "model.mlir"
    receipt = stage / "capture_receipt.json"
    if any(path.is_symlink() or not path.is_file() for path in (source, receipt)):
        raise ValueError("MX whole-stage build needs plain model.mlir and capture_receipt.json")
    data = json.loads(receipt.read_text(encoding="utf-8"))
    record = (data.get("artifacts") or {}).get("model.mlir") or {}
    if (data.get("schema") != "m2m.capture-receipt.v1"
            or record.get("bytes") != source.stat().st_size
            or record.get("sha256") != hashlib.sha256(source.read_bytes()).hexdigest()):
        raise ValueError("MX whole-stage build refuses a changed model2MLIR capture source")
    return source


def _can_hash_first_output(stage: Path) -> bool:
    """The current Merlin bare-metal digest supports only f32 first outputs."""
    golden = stage / "golden.npy"
    if not golden.is_file() or golden.is_symlink():
        raise ValueError("MX whole-stage build needs a plain golden.npy")
    return np.load(golden, mmap_mode="r", allow_pickle=False).dtype == np.dtype("float32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path)
    parser.add_argument("--merlin-src", type=Path, required=True)
    parser.add_argument("--llvm-install", type=Path, required=True)
    parser.add_argument("--riscv-root", type=Path, required=True)
    parser.add_argument("--arena-mb", type=int, default=2048)
    parser.add_argument("--output-dump-cap", type=int, default=1)
    parser.add_argument("--spike-extlib", type=Path,
                        help="run the built ELF on Spike with this pinned Gemmini extension")
    parser.add_argument("--spike-timeout", type=int, default=600)
    args = parser.parse_args()
    stage = args.stage.resolve(strict=True)
    work = args.workdir.absolute()
    if work.exists():
        parser.error("--workdir must not already exist")
    source = _capture_model(stage)
    output_sha256 = _can_hash_first_output(stage)
    capture_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if args.arena_mb <= 0 or args.output_dump_cap <= 0 or args.spike_timeout <= 0:
        parser.error("arena, dump cap and Spike timeout must be positive")
    profile_path = args.profile.resolve(strict=True)
    riscv_root = args.riscv_root.resolve(strict=True)
    llvm_root = args.llvm_install.resolve(strict=True)
    merlin_src = args.merlin_src.resolve(strict=True)
    if not (merlin_src / "merlin/runtime/backends/spike_model.py").is_file():
        parser.error("--merlin-src must be Merlin's src directory")
    from mx_gemmini_support.target_profile import load_profile

    profile = load_profile(profile_path, rtl_root=args.rtl_root)
    os.environ["MERLIN_TARGET_PATH"] = str(ROOT)
    os.environ["MERLIN_COMPILER_PYTHON"] = sys.executable
    os.environ["MERLIN_MLIR_INSTALL"] = str(llvm_root)
    os.environ["MERLIN_CLANG"] = str(llvm_root / "bin/clang")
    os.environ["MERLIN_RISCV_GCC"] = str(riscv_root / "bin/riscv64-unknown-elf-gcc")
    os.environ["MERLIN_SPIKE"] = str(riscv_root / "bin/spike")
    sys.path.insert(0, str(merlin_src))
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.runtime.backends import spike_model
    from mx_gemmini_support.merlin_bf16_prepare import MXBF16CatalogPreparation
    from mx_gemmini_support.merlin_catalog_builder import make_merlin_catalog_builder
    from mx_gemmini_support.target_profile import profile_sha256

    device = DeviceRouting(
        device="mx_gemmini", package_dir=ROOT,
        operand_dtype="catalog_mixed_direct_fp4_fp8", accum_dtype="bf16",
        prepared_transform=MXBF16CatalogPreparation(profile),
        catalog_builder=make_merlin_catalog_builder(
            profile_path=profile_path, riscv_root=riscv_root),
    )
    result = spike_model.build(
        stage, work, arena_mb=args.arena_mb, backend="scalar", device=device,
        cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2"],
        output_dump_cap=args.output_dump_cap, output_sha256=output_sha256,
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != capture_sha:
        raise RuntimeError("Merlin changed the receipt-bound model2MLIR capture source")
    elf = Path(result["elf"])
    catalog = json.loads((work / "device_catalog/mx_catalog.json").read_text(encoding="utf-8"))
    bf16_receipt = work / "device_prepared/mx_bf16_catalog.json"
    widened_bf16 = (json.loads(bf16_receipt.read_text(encoding="utf-8"))["widened_operations"]
                    if bf16_receipt.is_file() else 0)
    receipt = {
        "schema": "mx_gemmini.merlin_stage_probe.v1",
        "status": "linked_experimental_numeric_qualification_pending",
        "capture_model_sha256": capture_sha,
        "profile_sha256": profile_sha256(profile),
        "catalog_manifest": str(work / "device_catalog/mx_catalog.json"),
        "selected_operations": catalog["covered_contractions"],
        "unbound_precision_proposals": len(catalog["unbound_precision_proposals"]),
        "widened_bf16_contractions": widened_bf16,
        "unique_kernels": catalog["unique_kernels"],
        "elf": str(elf), "elf_bytes": elf.stat().st_size,
        "build_hash": result.get("build_hash"),
        "mem_bytes": result["mem_bytes"],
        "arena_mb": args.arena_mb,
        "first_output_sha256_enabled": output_sha256,
    }
    if args.spike_extlib is not None:
        extlib = args.spike_extlib.resolve(strict=True)
        if not extlib.is_file():
            raise ValueError("Spike extension must be a regular shared library")
        command = [str(riscv_root / "bin/spike"), "--isa=rv64gc",
                   f"--extlib={extlib}", "--extension=gemmini",
                   f"-m{hex(spike_model.DRAM_BASE)}:{hex(result['mem_bytes'])}",
                   str(elf)]
        log = work / "spike.log"
        with log.open("wb") as stream:
            try:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=args.spike_timeout, check=False)
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
        console = log.read_text(encoding="utf-8", errors="replace")
        receipt["spike"] = {
            "command": command, "extension_sha256": hashlib.sha256(extlib.read_bytes()).hexdigest(),
            "exit_code": exit_code, "done": exit_code == 0 and "DONE\n" in console,
            "log": str(log),
        }
        if not receipt["spike"]["done"]:
            receipt["status"] = "spike_inference_incomplete"
    path = work / "mx_stage_probe.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(path), "status": receipt["status"],
                      "selected_operations": receipt["selected_operations"],
                      "elf_bytes": receipt["elf_bytes"]}, sort_keys=True))
    if args.spike_extlib is not None and not receipt["spike"]["done"]:
        raise SystemExit("MX Spike run did not complete; inspect the probe receipt and log")


if __name__ == "__main__":
    main()
