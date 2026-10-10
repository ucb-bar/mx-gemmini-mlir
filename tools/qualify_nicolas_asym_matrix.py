"""Compile Nicolas asymmetric source modes on their matching Rocket mesh profile.

Each row captures a fresh PyTorch matmul through model2MLIR, binds Nicolas's
checked-in packed arrays to typed MX MLIR, emits a standalone RoCC ELF, and
compares every BF16 source output on the pinned Spike extension.
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


def discover(software: Path, profile_dir: Path, rtl_root: Path,
             mesh_dim: int = 16, *, all_asym: bool = False,
             source_shape: str = "64x64"
             ) -> list[tuple[str, Path, dict]]:
    """Preflight each checked-in source/header against its legal mesh profile."""
    if all_asym and mesh_dim != 16:
        raise ValueError("DIM8/DIM32 already select their all-asymmetric profiles")
    rows = []
    dim_suffix = f"_dim{mesh_dim}" if mesh_dim != 16 else ""
    sources = sorted((software / "bareMetalC").glob(
        f"matmul_tiled_asym_*_{source_shape}{dim_suffix}.c"))
    if not sources:
        raise ValueError("Nicolas asymmetric source tests are absent")
    for source in sources:
        match = re.fullmatch(rf"matmul_tiled_asym_([a-z0-9]+)_([a-z0-9]+)_{source_shape}{dim_suffix}\.c",
                             source.name)
        if match is None or any(token not in _NAME for token in match.groups()):
            raise ValueError(f"unrecognized Nicolas source test: {source.name}")
        left, right = match.groups()
        suffix = f"{left}_{right}"
        shape_suffix = f"_{source_shape}" if source_shape != "64x64" else ""
        header = software / "include" / f"matmul_data_asym_{suffix}{shape_suffix}{dim_suffix}.h"
        if mesh_dim == 16:
            profile_name = ("MxAllAsymGemminiRocketConfig" if all_asym else
                            f"MxAsym{_NAME[left]}{_NAME[right]}GemminiRocketConfig")
        else:
            profile_name = f"MxDim{mesh_dim}AllAsymGemminiRocketConfig"
        profile_path = profile_dir / f"{profile_name}.json"
        profile = load_profile(profile_path, rtl_root=rtl_root)
        recipe = source_recipe(source, header, profile)
        rows.append((suffix, profile_path, recipe["compute"]))
    if len({suffix for suffix, _, _ in rows}) != len(rows):
        raise ValueError("Nicolas asymmetric source suffixes are not unique")
    if mesh_dim == 16 and not all_asym and source_shape == "64x64":
        by_profile: dict[Path, list[dict]] = {}
        for _, profile_path, cell in rows:
            by_profile.setdefault(profile_path, []).append(cell)
        dedicated = [path for path in sorted(profile_dir.glob("MxAsym*GemminiRocketConfig.json"))
                     if load_profile(path, rtl_root=rtl_root)["geometry"]["mesh_rows"] == 16]
        for profile_path in dedicated:
            profile = load_profile(profile_path, rtl_root=rtl_root)
            expected = {json.dumps(cell, sort_keys=True) for cell in profile["legal_compute"]}
            actual = [json.dumps(cell, sort_keys=True) for cell in by_profile.get(profile_path, ())]
            if set(actual) != expected or len(actual) != len(expected):
                raise ValueError(f"Nicolas DIM16 source mode coverage is incomplete: {profile_path.name}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "rtl-root", "riscv-root",
                 "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1,
                        help="independent Spike builds to run concurrently (1–4)")
    parser.add_argument("--mesh-dim", type=int, choices=(8, 16, 32), default=16,
                        help="Rocket mesh dimension (default: 16)")
    parser.add_argument("--source-shape", choices=("64x64", "128x128", "128x128x256"),
                        default="64x64", help="named Nicolas source shape")
    parser.add_argument("--all-asym", action="store_true",
                        help="select DIM16 MxAllAsymGemminiRocketConfig instead of dedicated profiles")
    parser.add_argument("--source-suffix", action="append",
                        help="qualify only this named source pair; repeat to select several")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4")
    if args.all_asym and args.mesh_dim != 16:
        parser.error("DIM8/DIM32 already select their all-asymmetric profiles")
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        parser.error(f"refusing to overwrite {out_dir}")
    root = Path(__file__).resolve().parents[1]
    rtl = args.rtl_root.resolve()
    software = rtl / "software/gemmini-rocc-tests"
    rows = discover(software, root / "profiles/gemmini-mx-cleanup-266c593",
                    rtl, args.mesh_dim, all_asym=args.all_asym,
                    source_shape=args.source_shape)
    if args.source_suffix:
        selected = set(args.source_suffix)
        unknown = selected - {suffix for suffix, _, _ in rows}
        if unknown:
            parser.error(f"unknown source suffixes: {', '.join(sorted(unknown))}")
        rows = [(suffix, profile, cell) for suffix, profile, cell in rows
                if suffix in selected]
    out_dir.mkdir(parents=True)

    def qualify(row: tuple[str, Path, dict]) -> dict:
        suffix, profile, cell = row
        directory = out_dir / suffix
        command = [sys.executable, "-m", "tools.qualify_nicolas_asym",
                   "--source-suffix", suffix, "--mesh-dim", str(args.mesh_dim),
                   "--source-shape", args.source_shape,
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
        expected_outputs = 4096 if args.source_shape == "64x64" else 16384
        passed = (run.returncode == 0 and
                  receipt.get("status") == "source_golden_matched_on_pinned_spike" and
                  receipt.get("compared_bf16_outputs") == expected_outputs)
        return {"source_suffix": suffix, "profile_name": profile.stem,
                "compute": cell,
                "status": "passed" if passed else "failed",
                "exit_code": run.returncode,
                "receipt_sha256": _sha(receipt_path) if receipt else None,
                "compiler_revision": receipt.get("compiler_revision"),
                "physical_program_sha256": receipt.get("physical_program_sha256"),
                "elf_sha256": receipt.get("elf_sha256"),
                "spike_log_sha256": receipt.get("spike_log_sha256"),
                "matched_bf16_outputs": expected_outputs if passed else 0}

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(executor.map(qualify, rows))
    legal_cells = {(profile_path.stem, json.dumps(cell, sort_keys=True)): cell
                   for _, profile_path, _ in rows
                   for cell in load_profile(profile_path, rtl_root=rtl)["legal_compute"]}
    selected_cells = {(profile_path.stem, json.dumps(cell, sort_keys=True))
                      for _, profile_path, cell in rows}
    missing_cells = [{"profile_name": key[0], "compute": legal_cells[key]}
                     for key in sorted(legal_cells.keys() - selected_cells)]
    manifest = {"schema": "mx_gemmini.nicolas_asymmetric_mode_matrix.v1",
                "scope": (f"named DIM{args.mesh_dim} {args.source_shape} asymmetric Rocket/RoCC source tests "
                          "on pinned Spike" + (" using the all-asymmetric profile"
                                               if args.all_asym else "")),
                "mesh_dim": args.mesh_dim,
                "source_shape": args.source_shape,
                "all_asym_profile": args.all_asym or args.mesh_dim != 16,
                "selected_modes": len(rows),
                "selected_profiles": len({profile for _, profile, _ in rows}),
                "legal_mode_count": len(legal_cells),
                "uncovered_legal_compute": missing_cells,
                "profile_complete": not missing_cells,
                "passed_modes": sum(row["status"] == "passed" for row in results),
                "rows": results}
    (out_dir / "matrix_receipt.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{manifest['passed_modes']}/{manifest['selected_modes']} source modes matched on Spike")
    if manifest["passed_modes"] != manifest["selected_modes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
