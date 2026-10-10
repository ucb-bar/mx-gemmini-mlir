"""Archive two matching full-roster captures and Spike qualifications.

Only textual compiler outputs and receipts are stored in Git. The manifests
retain hashes of the generated payloads, objects, ELF, and Spike extension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


FRONTEND_FILES = (
    "receipt.json", "mx_gemm.model2mlir.mlir", "mx_gemm.handoff.mlir",
    "mx_gemm.profile_bound.mlir", "quantization_manifest.json",
)
NUMERICAL_FILES = (
    "build/artifact_manifest.json", "payload_bound.mlir",
    "build/physical_program.json", "build/mx_issue.c", "build/mx_driver.c",
    "build/mx_data.S", "build/spike.log",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frontend-first", "frontend-repro", "spike-first",
                 "spike-repro", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--header-materialization", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    front = [_read(path / "index.json") for path in
             (args.frontend_first, args.frontend_repro)]
    spike = [_read(path / "index.json") for path in
             (args.spike_first, args.spike_repro)]
    if (any(index["covered_drivers"] != 31 for index in spike) or
            any(index["captured_drivers"] != 31 for index in front) or
            any(index["fullout_drivers"] != 23 or index["requant_drivers"] != 8
                for index in front + spike)):
        raise ValueError("expected the complete 31-driver Radiance MX roster")
    for key in ("source_revision", "model2mlir_revision", "rtl_revision"):
        values = [index[key] for index in front + spike]
        if len(set(values)) != 1:
            raise ValueError(f"roster revisions differ: {key}")
    if spike[0]["compiler_revision"] != spike[1]["compiler_revision"]:
        raise ValueError("Spike compiler revisions differ")
    headers = _read(args.header_materialization)
    if (headers.get("schema") != "mx_gemmini.radiance_header_materialization.v1" or
            headers.get("drivers") != 31 or len(headers.get("rows", [])) != 31):
        raise ValueError("header materialization is not a full Radiance MX roster")
    for header, captured in zip(headers["rows"], front[0]["rows"]):
        if (header["driver"] != captured["driver"] or
                header["driver_sha256"] != captured["driver_sha256"] or
                header["header_sha256"] != captured["header_sha256"]):
            raise ValueError(f"materialized source differs: {captured['driver']}")
    for capture, qualification in zip((args.frontend_first,
                                       args.frontend_repro),
                                      (args.spike_first, args.spike_repro)):
        if _sha(capture / "index.json") != _read(qualification / "index.json")[
                "frontend_index_sha256"]:
            raise ValueError("numerical run does not cite its frontend index")

    for position in range(31):
        captures = [index["rows"][position] for index in front]
        numerics = [index["rows"][position] for index in spike]
        driver = captures[0]["driver"]
        if any(row["driver"] != driver for row in captures + numerics):
            raise ValueError(f"roster order differs at row {position}")
        for key in ("driver_sha256", "header_sha256", "source_mlir_sha256",
                    "bound_mlir_sha256", "profile_sha256", "shape_mnk",
                    "tile_mnk", "precision", "quant_output", "site_id"):
            if captures[0][key] != captures[1][key]:
                raise ValueError(f"frontend reproduction differs: {driver}: {key}")
        for key in ("source_driver_sha256", "source_header_sha256",
                    "profile_bound_mlir_sha256",                     "payload_bound_mlir_sha256", "elf_sha256",
                    "payload_bound_mlir_sha256", "elf_sha256",
                    "spike_log_sha256", "comparison", "compared_count",
                    "status"):
            if numerics[0][key] != numerics[1][key]:
                raise ValueError(f"Spike reproduction differs: {driver}: {key}")
        capture_receipts = []
        for root, row, numeric in zip((args.frontend_first, args.frontend_repro),
                                      captures, numerics):
            receipt = root / row["receipt"]
            if _sha(receipt) != row["receipt_sha256"]:
                raise ValueError(f"frontend receipt hash differs: {driver}")
            if numeric["frontend_receipt_sha256"] != row["receipt_sha256"]:
                raise ValueError(f"Spike receipt cites another frontend: {driver}")
            capture_receipts.append(_read(receipt))
            case = receipt.parent
            if (_sha(case / "mx_gemm.model2mlir.mlir") != row["source_mlir_sha256"] or
                    _sha(case / "mx_gemm.profile_bound.mlir") !=
                    row["bound_mlir_sha256"]):
                raise ValueError(f"frontend artifact hash differs: {driver}")
        for receipt, index in zip(capture_receipts, front):
            if receipt["mx_support_revision"] != index["compiler_revision"]:
                raise ValueError(f"frontend compiler revision differs: {driver}")
        receipt_first, receipt_repro = capture_receipts
        receipt_first.pop("mx_support_revision")
        receipt_repro.pop("mx_support_revision")
        if receipt_first != receipt_repro:
            raise ValueError(f"frontend receipt changed beyond compiler revision: {driver}")
        manifests = []
        for root, row in zip((args.spike_first, args.spike_repro), numerics):
            receipt = root / row["receipt"]
            if _sha(receipt) != row["receipt_sha256"]:
                raise ValueError(f"numerical receipt hash differs: {driver}")
            manifest = _read(receipt)
            if (manifest["bound_mlir_sha256"] != row["payload_bound_mlir_sha256"] or
                    manifest["elf_sha256"] != row["elf_sha256"] or
                    manifest["spike_log_sha256"] != row["spike_log_sha256"]):
                raise ValueError(f"numerical artifact hash differs: {driver}")
            manifests.append(manifest)
        for key in ("files_sha256", "object_sha256", "extension_sha256",
                    "elf_sha256", "spike_log_sha256", "bound_mlir_sha256"):
            if manifests[0][key] != manifests[1][key]:
                raise ValueError(f"generated artifacts differ: {driver}: {key}")

    for name, root, files in (
            ("frontend", args.frontend_first, FRONTEND_FILES),
            ("frontend_repro", args.frontend_repro, ("receipt.json",)),
            ("spike", args.spike_first, NUMERICAL_FILES),
            ("spike_repro", args.spike_repro, ("build/artifact_manifest.json",))):
        _copy(root / "index.json", args.out_dir / name / "index.json")
        for row in _read(root / "index.json")["rows"]:
            case = Path(row["driver"]).stem
            for filename in files:
                _copy(root / case / filename, args.out_dir / name / case / filename)
    _copy(args.header_materialization,
          args.out_dir / "header_materialization.json")
    print(f"archived 31 reproducible MX GEMM cases -> {args.out_dir}")


if __name__ == "__main__":
    main()
