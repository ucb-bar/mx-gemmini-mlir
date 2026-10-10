"""Compile every Nicolas DIM16 asymmetric source mode on its Rocket profile.

Each row captures a fresh PyTorch matmul through model2MLIR, binds Nicolas's
checked-in packed arrays to typed MX MLIR, emits a standalone RoCC ELF, and
compares all 4,096 BF16 outputs on the pinned Spike extension.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from mx_gemmini_support.asymmetric_specialization import source_recipe
from mx_gemmini_support.target_profile import load_profile


_NAME = {"e2m3": "E2M3", "e3m2": "E3M2", "e4m3": "E4M3",
         "e4m3s": "E4M3", "e5m2": "E5M2", "e5m2s": "E5M2",
         "fp4": "Fp4", "fp6": "Fp6"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover(software: Path, profile_dir: Path, rtl_root: Path) -> list[tuple[str, Path]]:
    """Preflight every checked-in DIM16 source/header against its legal profile."""
    rows = []
    sources = sorted((software / "bareMetalC").glob("matmul_tiled_asym_*_64x64.c"))
    if not sources:
        raise ValueError("Nicolas asymmetric 64-cubed source tests are absent")
    for source in sources:
        match = re.fullmatch(r"matmul_tiled_asym_([a-z0-9]+)_([a-z0-9]+)_64x64\.c",
                             source.name)
        if match is None or any(token not in _NAME for token in match.groups()):
            raise ValueError(f"unrecognized Nicolas source test: {source.name}")
        left, right = match.groups()
        suffix = f"{left}_{right}"
        header = software / "include" / f"matmul_data_asym_{suffix}.h"
        profile_path = profile_dir / f"MxAsym{_NAME[left]}{_NAME[right]}GemminiRocketConfig.json"
        profile = load_profile(profile_path, rtl_root=rtl_root)
        source_recipe(source, header, profile)
        rows.append((suffix, profile_path))
    if len({suffix for suffix, _ in rows}) != len(rows):
        raise ValueError("Nicolas asymmetric source suffixes are not unique")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "rtl-root", "riscv-root",
                 "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1,
                        help="independent Spike builds to run concurrently (1–4)")
    parser.add_argument("--source-suffix", action="append",
                        help="qualify only this named source pair; repeat to select several")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4")
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        parser.error(f"refusing to overwrite {out_dir}")
    root = Path(__file__).resolve().parents[1]
    rtl = args.rtl_root.resolve()
    software = rtl / "software/gemmini-rocc-tests"
    rows = discover(software, root / "profiles/gemmini-mx-cleanup-266c593", rtl)
    if args.source_suffix:
        selected = set(args.source_suffix)
        unknown = selected - {suffix for suffix, _ in rows}
        if unknown:
            parser.error(f"unknown source suffixes: {', '.join(sorted(unknown))}")
        rows = [(suffix, profile) for suffix, profile in rows if suffix in selected]
    out_dir.mkdir(parents=True)

    def qualify(row: tuple[str, Path]) -> dict:
        suffix, profile = row
        directory = out_dir / suffix
        command = [sys.executable, "-m", "tools.qualify_nicolas_asym",
                   "--source-suffix", suffix,
                   "--model2mlir-root", str(args.model2mlir_root.resolve()),
                   "--mxq-root", str(args.mxq_root.resolve()),
                   "--rtl-root", str(rtl), "--profile", str(profile),
                   "--riscv-root", str(args.riscv_root.resolve()),
                   "--mx-opt", str(args.mx_opt.resolve()),
                   "--out-dir", str(directory)]
        run = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
        directory.mkdir(exist_ok=True)
        (directory / "matrix_run.log").write_text(run.stdout)
        receipt_path = directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        passed = (run.returncode == 0 and
                  receipt.get("status") == "source_golden_matched_on_pinned_spike" and
                  receipt.get("compared_bf16_outputs") == 4096)
        return {"source_suffix": suffix, "profile_name": profile.stem,
                "status": "passed" if passed else "failed",
                "exit_code": run.returncode,
                "receipt_sha256": _sha(receipt_path) if receipt else None,
                "compiler_revision": receipt.get("compiler_revision"),
                "physical_program_sha256": receipt.get("physical_program_sha256"),
                "elf_sha256": receipt.get("elf_sha256"),
                "spike_log_sha256": receipt.get("spike_log_sha256"),
                "matched_bf16_outputs": 4096 if passed else 0}

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(executor.map(qualify, rows))
    manifest = {"schema": "mx_gemmini.nicolas_asymmetric_mode_matrix.v1",
                "scope": "named DIM16 64x64x64 asymmetric Rocket/RoCC source tests on pinned Spike",
                "selected_modes": len(rows),
                "passed_modes": sum(row["status"] == "passed" for row in results),
                "rows": results}
    (out_dir / "matrix_receipt.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{manifest['passed_modes']}/{manifest['selected_modes']} source modes matched on Spike")
    if manifest["passed_modes"] != manifest["selected_modes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
