"""Run the selected Nicolas requantizer wrapper through three MX output modes.

The frontend handoffs must come from a current model2MLIR Radiance roster
capture. FP8 and FP4 use the source requant drivers. FP6 uses the checked-in
fullout driver and explicitly derives a quantized terminal readout; it does
not stand in for a missing Radiance FP6 requant header.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/TestRequantizerLutMxGemminiRocketConfig.json"
CASES = (
    ("fp8", "mxgemm.fp8.singletile.tm64tn64tk64.requant", False, 4096, 128),
    ("fp4", "mxgemm.fp4.singletile.tm64tn64tk64.requant", False, 2048, 128),
    ("fp6", "mxgemm.fp6.m128n128k2048.tm128tn128tk128.fullout", True, 8192, 512),
)
ARTIFACTS = (
    "payload_bound.mlir", "bundle/manifest.json",
    "build/physical_program.json", "build/mx_issue.c", "build/mx_driver.c",
    "build/mx_data.S", "build/mx_program.elf", "build/spike.log",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _identity(index: dict) -> list[dict]:
    return [{key: row[key] for key in (
        "precision", "source_driver", "source_driver_sha256", "source_header_sha256",
        "capture_receipt_sha256", "profile_sha256", "status", "compared_codes_or_bytes",
        "compared_e8m0_scales", "source_quant_code_differences",
        "source_quant_scale_differences", "golden_basis", "artifact_sha256")}
            for row in index["rows"]]


def _qualify(args: argparse.Namespace, profile: dict, case: tuple) -> dict:
    precision, stem, derived, codes, scales = case
    driver = args.source_root / "kernels/gemm_mxgemmini" / f"{stem}.cpp"
    frontend = args.capture_root / "frontend" / stem
    handoff = frontend / "mx_gemm.handoff.mlir"
    capture = _json(frontend / "receipt.json")
    kernel = read_source_gemm(driver)
    if (capture.get("status") != "source_shape_frontend_handoff_only" or
            capture.get("model2mlir_revision") != _revision(args.model2mlir_root) or
            capture.get("source_revision") != _revision(args.source_root) or
            capture.get("source_driver_sha256") != _sha(driver) or
            capture.get("source_data_header_sha256") != _sha(kernel.data_header) or
            capture.get("handoff_mlir_sha256") != _sha(handoff) or
            capture.get("opaque_calls") or kernel.datatype != precision.upper() or
            kernel.quant_output == derived):
        raise ValueError(f"{stem}: model2MLIR capture or source payload differs")
    output = args.out_dir / precision
    output.mkdir()
    bound = output / "profile_bound.mlir"
    bound.write_text(bind_handoff(handoff.read_text(), profile))
    command = [
        sys.executable, "-m", "tools.qualify_source_mx",
        "--mlir", str(bound), "--driver", str(driver),
        "--profile", str(PROFILE), "--rtl-root", str(args.rtl_root),
        "--riscv-root", str(args.riscv_root), "--out-dir", str(output / "compile"),
    ]
    if derived:
        command.append("--fp6-quantized-specialization")
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    (output / "compile.log").write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"{stem}: compilation or Spike failed; see {output / 'compile.log'}")
    build = output / "compile" / "build"
    receipt = _json(build / "artifact_manifest.json")
    count_key = {"fp8": "compared_fp8_codes", "fp4": "compared_fp4_packed_bytes",
                 "fp6": "compared_fp6_packed_bytes"}[precision]
    golden_basis = {
        "fp8": "nicolas_mxquant_po2_rne_from_source_bf16",
        "fp4": "nicolas_fp4_e3m1_e2m1_from_source_bf16",
        "fp6": "nicolas_fp6_e3m2_lut_from_source_bf16",
    }[precision]
    if (receipt.get("status") != "nicolas_oracle_matched_on_pinned_spike" or
            receipt.get("spike_exit_code") != 0 or
            receipt.get("profile_sha256") != profile_sha256(profile) or
            receipt.get("rtl_revision") != _revision(args.rtl_root) or
            receipt.get("source_driver_sha256") != _sha(driver) or
            receipt.get("source_header_sha256") != _sha(kernel.data_header) or
            receipt.get(count_key) != codes or
            receipt.get("compared_e8m0_scales") != scales or
            receipt.get("golden_basis") != golden_basis or
            receipt.get("source_quant_scale_differences", 0) <= 0 or
            (precision != "fp4" and
             receipt.get("source_quant_code_differences", 0) <= 0)):
        raise ValueError(f"{stem}: selected requantizer path did not match its oracle")
    artifacts = {"profile_bound.mlir": _sha(bound),
                 "capture_handoff.mlir": _sha(handoff)}
    for relative in ARTIFACTS:
        artifacts[relative] = _sha(output / "compile" / relative)
    for path in sorted((output / "compile" / "bundle").glob("*.bin")):
        artifacts[f"bundle/{path.name}"] = _sha(path)
    return {
        "precision": precision, "source_driver": str(driver.relative_to(args.source_root)),
        "source_driver_sha256": _sha(driver),
        "source_header_sha256": _sha(kernel.data_header),
        "capture_receipt_sha256": _sha(frontend / "receipt.json"),
        "profile_sha256": profile_sha256(profile),
        "status": receipt["status"], "compared_codes_or_bytes": codes,
        "compared_e8m0_scales": scales,
        "source_quant_code_differences": receipt.get("source_quant_code_differences"),
        "source_quant_scale_differences": receipt["source_quant_scale_differences"],
        "golden_basis": receipt["golden_basis"],
        "artifact_sha256": artifacts,
        "compiler_revision": receipt["compiler_revision"],
        "model2mlir_revision": capture["model2mlir_revision"],
        "rtl_revision": receipt["rtl_revision"],
        "fp6_quantized_readout_derived_from_fullout": derived,
    }


def _archive(args: argparse.Namespace, rows: list[dict], index: dict) -> None:
    archive = args.archive_dir
    archive.mkdir()
    for row in rows:
        precision = row["precision"]
        output = args.out_dir / precision
        target = archive / precision
        target.mkdir()
        shutil.copyfile(output / "compile/build/artifact_manifest.json",
                        target / "artifact_manifest.json")
        shutil.copyfile(output / "compile.log", target / "compile.log")
        for relative in row["artifact_sha256"]:
            source = (args.capture_root / "frontend" /
                      Path(row["source_driver"]).stem / "mx_gemm.handoff.mlir"
                      if relative == "capture_handoff.mlir" else
                      output / "profile_bound.mlir" if relative == "profile_bound.mlir" else
                      output / "compile" / relative)
            (target / f"{relative.replace('/', '__')}.gz").write_bytes(
                gzip.compress(source.read_bytes(), mtime=0))
    (archive / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-root", "model2mlir-root", "source-root", "rtl-root",
                 "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--baseline-index", type=Path)
    args = parser.parse_args()
    for name in ("capture_root", "model2mlir_root", "source_root", "rtl_root",
                 "riscv_root", "out_dir", "archive_dir"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve())
    if args.out_dir.exists() or (args.archive_dir and args.archive_dir.exists()):
        parser.error("refusing to overwrite an output or archive directory")
    profile = load_profile(PROFILE, rtl_root=args.rtl_root)
    if (profile["name"] != "TestRequantizerLutMxGemminiRocketConfig" or
            profile["geometry"]["mesh_columns"] != 16 or
            not profile["resources"]["requantizer"] or
            set(profile["candidate_output_modes"]) !=
            {"bf16", "fp4_e2m1", "fp6_e3m2", "fp8_e4m3"}):
        raise ValueError("selected Nicolas requantizer profile differs")
    args.out_dir.mkdir(parents=True)
    rows = []
    for case in CASES:
        row = _qualify(args, profile, case)
        rows.append(row)
        print(f"{row['precision']}: {row['compared_codes_or_bytes']} codes/bytes and "
              f"{row['compared_e8m0_scales']} scales matched on Nicolas Spike", flush=True)
    index = {
        "schema": "mx_gemmini.nicolas_requantizer_wrapper.v1",
        "scope": "FP8 and FP4 source requant drivers; FP6 terminal readout derived from a source fullout driver",
        "profile_name": profile["name"], "profile_sha256": profile_sha256(profile),
        "rtl_revision": _revision(args.rtl_root),
        "source_revision": _revision(args.source_root),
        "model2mlir_revision": _revision(args.model2mlir_root),
        "rows": rows,
    }
    if args.baseline_index and _identity(index) != _identity(_json(args.baseline_index)):
        raise ValueError("selected requantizer wrapper output differs from pinned baseline")
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    if args.archive_dir:
        _archive(args, rows, index)
    print(args.out_dir / "index.json")


if __name__ == "__main__":
    main()
