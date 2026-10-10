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

from mx_gemmini_support.asymmetric_specialization import (generated_header_recipe,
                                                           source_recipe)
from mx_gemmini_support.target_profile import load_profile


_NAME = {"e2m3": "E2M3", "e3m2": "E3M2", "e4m3": "E4M3",
         "e4m3s": "E4M3", "e5m2": "E5M2", "e5m2s": "E5M2",
         "fp4": "Fp4", "fp6": "Fp6"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover(software: Path, profile_dir: Path, rtl_root: Path,
             mesh_dim: int = 16, *, all_asym: bool = False,
             source_shape: str = "64x64", include_symmetric_lut: bool = False,
             include_symmetric_fp4: bool = False,
             include_generated: bool = False
             ) -> list[tuple[str, Path, dict]]:
    """Preflight each checked-in source/header against its legal mesh profile."""
    if all_asym and mesh_dim != 16:
        raise ValueError("DIM8/DIM32 already select their all-asymmetric profiles")
    if (include_symmetric_lut or include_symmetric_fp4) and (
            mesh_dim != 16 or source_shape != "64x64" or not all_asym):
        raise ValueError("same-format source tests need DIM16 all-asymmetric 64x64")
    if include_generated and (source_shape != "64x64" or
                              (mesh_dim == 16 and not all_asym)):
        raise ValueError("generated modes need the 64x64 all-asymmetric profile")
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
    if include_symmetric_lut:
        profile_path = profile_dir / "MxAllAsymGemminiRocketConfig.json"
        profile = load_profile(profile_path, rtl_root=rtl_root)
        for name, source_name in (
                ("e2m3", "matmul_tiled_fp6_e2m3_lut_64x64.c"),
                ("e4m3", "matmul_tiled_fp8_e4m3_lut_64x64.c"),
                ("e5m2", "matmul_tiled_fp8_e5m2_64x64.c")):
            source = software / "bareMetalC" / source_name
            header = software / "include" / f"matmul_data_mx_lut_{name}_64x64.h"
            recipe = source_recipe(source, header, profile)
            rows.append((f"{name}_{name}", profile_path, recipe["compute"]))
    if include_symmetric_fp4:
        profile_path = profile_dir / "MxAllAsymGemminiRocketConfig.json"
        profile = load_profile(profile_path, rtl_root=rtl_root)
        source = software / "bareMetalC/matmul_tiled_fp4_64x64.c"
        header = software / "include/matmul_fp4_64x64.h"
        recipe = source_recipe(source, header, profile)
        rows.append(("fp4_fp4", profile_path, recipe["compute"]))
    if include_generated:
        profile_name = ("MxAllAsymGemminiRocketConfig" if mesh_dim == 16 else
                        f"MxDim{mesh_dim}AllAsymGemminiRocketConfig")
        profile_path = profile_dir / f"{profile_name}.json"
        profile = load_profile(profile_path, rtl_root=rtl_root)
        manifest_name = ("mx_gemmini_generated_modes_manifest.json" if mesh_dim == 16 else
                         f"mx_gemmini_generated_modes_dim{mesh_dim}_manifest.json")
        generated = json.loads((software / "include" / manifest_name).read_text())
        for name in sorted(generated["headers_sha256"]):
            header = software / "include" / f"matmul_data_asym_{name}{dim_suffix}.h"
            recipe = generated_header_recipe(software / "gen_asym.py", header, profile)
            rows.append((name, profile_path, recipe["compute"]))
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
    parser.add_argument("--include-symmetric-lut", action="store_true",
                        help="include Nicolas's three same-format DIM16 LUT source tests")
    parser.add_argument("--include-symmetric-fp4", action="store_true",
                        help="include Nicolas's direct FP4 by FP4 DIM16 BF16 source test")
    parser.add_argument("--include-generated", action="store_true",
                        help="include headers from pinned Nicolas gen_asym.py manifest")
    parser.add_argument("--source-suffix", action="append",
                        help="qualify only this named source pair; repeat to select several")
    parser.add_argument("--public-object", action="store_true",
                        help="link each case through tools.compile_object")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4")
    if args.all_asym and args.mesh_dim != 16:
        parser.error("DIM8/DIM32 already select their all-asymmetric profiles")
    if (args.include_symmetric_lut or args.include_symmetric_fp4) and (
            args.mesh_dim != 16 or args.source_shape != "64x64" or not args.all_asym):
        parser.error("same-format tests need --mesh-dim 16 --all-asym --source-shape 64x64")
    if args.include_generated and (args.source_shape != "64x64" or
                                   (args.mesh_dim == 16 and not args.all_asym)):
        parser.error("generated modes need a 64x64 all-asymmetric profile")
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        parser.error(f"refusing to overwrite {out_dir}")
    root = Path(__file__).resolve().parents[1]
    rtl = args.rtl_root.resolve()
    software = rtl / "software/gemmini-rocc-tests"
    rows = discover(software, root / "profiles/gemmini-mx-cleanup-266c593",
                    rtl, args.mesh_dim, all_asym=args.all_asym,
                    source_shape=args.source_shape,
                    include_symmetric_lut=args.include_symmetric_lut,
                    include_symmetric_fp4=args.include_symmetric_fp4,
                    include_generated=args.include_generated)
    if args.source_suffix:
        selected = set(args.source_suffix)
        unknown = selected - {suffix for suffix, _, _ in rows}
        if unknown:
            parser.error(f"unknown source suffixes: {', '.join(sorted(unknown))}")
        rows = [(suffix, profile, cell) for suffix, profile, cell in rows
                if suffix in selected]
    out_dir.mkdir(parents=True)
    generated_names = set()
    generation_manifest_sha256 = None
    if args.include_generated:
        name = ("mx_gemmini_generated_modes_manifest.json" if args.mesh_dim == 16 else
                f"mx_gemmini_generated_modes_dim{args.mesh_dim}_manifest.json")
        path = software / "include" / name
        generated_names = set(json.loads(path.read_text())["headers_sha256"])
        generation_manifest_sha256 = _sha(path)

    def qualify(row: tuple[str, Path, dict]) -> dict:
        suffix, profile, cell = row
        directory = out_dir / suffix
        same_format = (suffix.split("_")[0] if args.include_symmetric_lut and
                       suffix in {"e2m3_e2m3", "e4m3_e4m3", "e5m2_e5m2"} else None)
        source_selection = (["--generated-mode", suffix] if suffix in generated_names else
                            ["--symmetric-fp4"] if args.include_symmetric_fp4 and
                            suffix == "fp4_fp4" else
                            ["--symmetric-lut", same_format] if same_format else
                            ["--source-suffix", suffix])
        command = [sys.executable, "-m", "tools.qualify_nicolas_asym",
                   *source_selection, "--mesh-dim", str(args.mesh_dim),
                   "--source-shape", args.source_shape,
                   *(["--public-object"] if args.public_object else []),
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
                  receipt.get("compared_bf16_outputs") == expected_outputs and
                  (not args.public_object or
                   isinstance(receipt.get("public_object_sha256"), str) and
                   len(receipt["public_object_sha256"]) == 64))
        return {"source_suffix": suffix, "profile_name": profile.stem,
                "compute": cell,
                "status": "passed" if passed else "failed",
                "exit_code": run.returncode,
                "receipt_sha256": _sha(receipt_path) if receipt else None,
                "compiler_revision": receipt.get("compiler_revision"),
                "physical_program_sha256": receipt.get("physical_program_sha256"),
                "public_object_sha256": receipt.get("public_object_sha256"),
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
    passing_cells = {(row["profile_name"], json.dumps(row["compute"], sort_keys=True))
                     for row in results if row["status"] == "passed"}
    missing_keys = (legal_cells.keys() - passing_cells if args.include_generated else
                    legal_cells.keys() - selected_cells)
    missing_cells = [{"profile_name": key[0], "compute": legal_cells[key]}
                     for key in sorted(missing_keys)]
    failed_cells = [{"profile_name": row["profile_name"], "compute": row["compute"]}
                    for row in results if row["status"] == "failed"]
    manifest = {"schema": "mx_gemmini.nicolas_asymmetric_mode_matrix.v1",
                "scope": (f"named DIM{args.mesh_dim} {args.source_shape} " +
                          ("asymmetric and same-format" if
                           args.include_symmetric_lut or args.include_symmetric_fp4
                           else "asymmetric") +
                          (" source and generated-header" if args.include_generated else
                           "") + " Rocket/RoCC tests on pinned Spike" +
                          (" using the all-asymmetric profile" if args.all_asym else "")),
                "mesh_dim": args.mesh_dim,
                "source_shape": args.source_shape,
                "all_asym_profile": args.all_asym or args.mesh_dim != 16,
                "includes_symmetric_lut": args.include_symmetric_lut,
                "includes_symmetric_fp4": args.include_symmetric_fp4,
                "includes_generated": args.include_generated,
                "public_object": args.public_object,
                "generation_manifest_sha256": generation_manifest_sha256,
                "selected_modes": len(rows),
                "selected_profiles": len({profile for _, profile, _ in rows}),
                "legal_mode_count": len(legal_cells),
                "uncovered_legal_compute": missing_cells,
                "selected_but_failed_compute": failed_cells,
                "profile_complete": not missing_cells and not failed_cells,
                "passed_modes": sum(row["status"] == "passed" for row in results),
                "rows": results}
    (out_dir / "matrix_receipt.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{manifest['passed_modes']}/{manifest['selected_modes']} source modes matched on Spike")
    if manifest["passed_modes"] != manifest["selected_modes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
