"""Qualify the eight previously uncovered Nicolas Rocket MX wrappers on Spike.

Each selected Nicolas source or generated header is captured with model2MLIR,
lowered to physical MX commands, built as an RV64 ELF, and run twice. Generated
headers must first be materialized with the pinned Nicolas generator helpers.
The optional compact archive retains both receipts and the first program.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"
CASES = (
    ("dim8_all", "MxDim8AllGemminiRocketConfig", 8,
     ("--source-suffix", "e4m3_e2m3")),
    ("dim32_all", "MxDim32AllGemminiRocketConfig", 32,
     ("--source-suffix", "e4m3_e2m3")),
    ("dim32_base", "MxDim32GemminiRocketConfig", 32,
     ("--generated-mode", "fp4_fp4")),
    ("e2m3_only", "MxE2M3OnlyGemminiRocketConfig", 16,
     ("--symmetric-lut", "e2m3")),
    ("e3m2_only", "MxE3M2OnlyGemminiRocketConfig", 16,
     ("--generated-mode", "e3m2_e3m2")),
    ("e5m2_only", "MxE5M2OnlyGemminiRocketConfig", 16,
     ("--symmetric-lut", "e5m2")),
    ("test_mx", "TestMxGemminiRocketConfig", 16,
     ("--symmetric-fp4",)),
    ("test_requant", "TestRequantizerLutMxGemminiRocketConfig", 16,
     ("--symmetric-fp4",)),
)
ARTIFACTS = (
    "asymmetric_bound.mlir", "handoff.mlir", "model2mlir.mlir",
    "quantization_manifest.json", "recipe.json",
    "physical/artifact_manifest.json", "physical/asymmetric_program.elf",
    "physical/mx_data.S", "physical/mx_driver.c", "physical/mx_issue.c",
    "physical/physical_program.json", "physical/resource_manifest.json",
    "physical/spike.log",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def _one_run(args: argparse.Namespace, case: tuple, label: str) -> tuple[Path, dict]:
    slug, name, dim, selection = case
    output = args.out_dir / slug / label
    command = [
        sys.executable, "-m", "tools.qualify_nicolas_asym", *selection,
        "--mesh-dim", str(dim),
        "--model2mlir-root", str(args.model2mlir_root),
        "--mxq-root", str(args.mxq_root),
        "--rtl-root", str(args.rtl_root),
        "--profile", str(PROFILES / f"{name}.json"),
        "--riscv-root", str(args.riscv_root),
        "--mx-opt", str(args.mx_opt), "--out-dir", str(output),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / f"{label}.log").write_text(result.stdout)
    receipt = _read(output / "receipt.json") if (output / "receipt.json").is_file() else {}
    if (result.returncode or receipt.get("status") !=
            "source_golden_matched_on_pinned_spike" or
            receipt.get("compared_bf16_outputs") != 4096 or
            receipt.get("profile_name") != name or
            receipt.get("mesh_dim") != dim or
            receipt.get("spike_exit_code") != 0):
        raise RuntimeError(f"{name} {label} failed; see {output.parent / f'{label}.log'}")
    return output, receipt


def _qualify_case(args: argparse.Namespace, case: tuple) -> dict:
    slug, name, dim, selection = case
    profile = load_profile(PROFILES / f"{name}.json", rtl_root=args.rtl_root)
    if profile["geometry"]["mesh_rows"] != dim:
        raise ValueError(f"{name} geometry differs")
    first, a = _one_run(args, case, "first")
    repro, b = _one_run(args, case, "repro")
    stable = (
        "compiler_revision", "compiler_source_closure_sha256", "model2mlir_revision",
        "mxq_revision", "rtl_revision", "software_revision",
        "source_header_sha256", "profile_sha256", "physical_program_sha256",
        "generated_c_sha256", "elf_sha256", "spike_log_sha256",
        "extension_source_closure_sha256", "physical_command_count",
        "physical_fence_count", "compared_bf16_outputs",
    )
    if (any(a[key] != b[key] for key in stable) or
            any(a.get(key) != b.get(key) for key in
                ("source_driver_sha256", "source_generation_manifest_sha256")) or
            a["profile_sha256"] != profile_sha256(profile) or
            a["rtl_revision"] != _revision(args.rtl_root)):
        raise ValueError(f"{name} did not reproduce its pinned profile and output")
    for relative in ARTIFACTS:
        if _sha(first / relative) != _sha(repro / relative):
            raise ValueError(f"{name} did not reproduce {relative}")
    files = {relative: _sha(first / relative) for relative in ARTIFACTS}
    files.update({f"physical/{path.name}": _sha(path)
                  for path in sorted((first / "physical").glob("*.bin"))})
    row = {
        "slug": slug, "profile_name": name, "mesh_dim": dim,
        "source_selection": list(selection), "profile_sha256": a["profile_sha256"],
        "compiler_revision": a["compiler_revision"],
        "compiler_source_closure_sha256": a["compiler_source_closure_sha256"],
        "model2mlir_revision": a["model2mlir_revision"],
        "rtl_revision": a["rtl_revision"],
        "source_driver_sha256": a.get("source_driver_sha256"),
        "source_generation_manifest_sha256": a.get(
            "source_generation_manifest_sha256"),
        "source_header_sha256": a["source_header_sha256"],
        "compared_bf16_outputs_per_run": 4096,
        "artifact_sha256": files,
    }
    print(f"{name}: two Spike runs, 4096 / 4096 BF16 outputs each", flush=True)
    return row


def _archive(args: argparse.Namespace, rows: list[dict], index: dict) -> None:
    archive = args.archive_dir
    if archive.exists():
        raise ValueError(f"refusing to overwrite {archive}")
    archive.mkdir(parents=True)
    for row in rows:
        slug = row["slug"]
        source = args.out_dir / slug / "first"
        dest = archive / slug
        dest.mkdir()
        for label in ("first", "repro"):
            shutil.copyfile(args.out_dir / slug / label / "receipt.json",
                            dest / f"receipt_{label}.json")
            shutil.copyfile(args.out_dir / slug / label / "physical/spike.log",
                            dest / f"spike_{label}.log")
        for relative in row["artifact_sha256"]:
            if relative == "physical/spike.log":
                continue
            target = dest / (relative.replace("/", "__") + ".gz")
            target.write_bytes(gzip.compress((source / relative).read_bytes(), mtime=0))
    (archive / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "rtl-root", "riscv-root",
                 "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--archive-dir", type=Path,
                        help="write compact checked-in evidence after all runs pass")
    parser.add_argument("--baseline-index", type=Path,
                        help="require the same stable artifact hashes as this index")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 4:
        parser.error("--jobs must be between 1 and 4")
    args.out_dir = args.out_dir.resolve()
    args.rtl_root = args.rtl_root.resolve()
    if args.out_dir.exists() or (args.archive_dir and args.archive_dir.exists()):
        parser.error("refusing to overwrite an output or archive directory")
    args.out_dir.mkdir(parents=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_case = {executor.submit(_qualify_case, args, case): case
                          for case in CASES}
        rows = [future.result() for future in as_completed(future_to_case)]
    rows.sort(key=lambda row: row["slug"])
    if len(rows) != 8 or len({row["profile_name"] for row in rows}) != 8:
        raise ValueError("Rocket wrapper matrix did not cover eight distinct profiles")
    index = {
        "schema": "mx_gemmini.rocket_wrapper_spike_matrix.v1",
        "scope": "one source-derived legal BF16 mode per named Rocket wrapper; two stock Spike runs",
        "rtl_revision": _revision(args.rtl_root),
        "profile_count": 8, "compared_bf16_outputs_per_run": 32768,
        "rows": rows,
    }
    if args.baseline_index and _read(args.baseline_index) != index:
        raise ValueError("Rocket wrapper matrix differs from pinned baseline")
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    if args.archive_dir:
        _archive(args, rows, index)
    print(args.out_dir / "index.json")


if __name__ == "__main__":
    main()
