"""Archive and verify the current-frontend Nicolas connected-chain suite."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


CASES = tuple(f"{precision}_{dimension}"
              for dimension in (64, 128) for precision in ("fp8", "fp4", "fp6"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _selected(source: Path) -> list[Path]:
    result = [Path("index.json"), Path("model2mlir_64.mlir"),
              Path("model2mlir_128.mlir"), Path("libgemmini.so")]
    for case in CASES:
        folder = Path(case)
        result += [folder / name for name in (
            "model2mlir.mlir", "frontend_binding.json", "connected.mlir",
            "abi.json", "receipt.json", "object_compile.log",
            "object/compile_manifest.json", "object/object_manifest.json",
            "object/physical_program.json", "object/mx_issue.c",
            "object/mx_issue.h", "object/mx_issue.o", "run/spike.log")]
        result += [folder / "run/mx_program.elf" if case.startswith("fp8")
                   else folder / "run/program.elf"]
        result += ([folder / "run/mx_driver.c", folder / "run/mx_data.S"]
                   if case.startswith("fp8") else [folder / "compiler_driver.c"])
        result += [path.relative_to(source)
                   for path in sorted((source / case / "resources").glob("*.bin"))]
    if len(result) != len(set(result)) or any(not (source / path).is_file()
                                             for path in result):
        raise ValueError("portable chain suite lacks selected archived artifacts")
    return result


def archive(source: Path, out: Path) -> dict:
    if out.exists():
        raise ValueError("refusing to overwrite a portable chain archive")
    suite = json.loads((source / "index.json").read_text())
    if (suite.get("schema") != "mx_gemmini.nicolas_portable_chain_suite.v1" or
            suite.get("status") !=
            "six_current_frontend_chains_matched_source_headers_on_pinned_spike" or
            [row["case"] for row in suite.get("rows", [])] != list(CASES)):
        raise ValueError("portable chain suite is incomplete")
    records = {}
    for path in _selected(source):
        data = (source / path).read_bytes()
        compress = path.suffix in {".bin", ".o", ".elf", ".so"}
        stored = Path(str(path) + ".gz") if compress else path
        target = out / stored
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(gzip.compress(data, compresslevel=9, mtime=0)
                           if compress else data)
        records[str(path)] = {"stored_as": str(stored), "sha256": _sha(data),
                              "bytes": len(data)}
    manifest = {"schema": "mx_gemmini.nicolas_portable_chain_archive.v1",
                "suite_sha256": _sha((source / "index.json").read_bytes()),
                "files": records}
    (out / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    verify(out)
    return manifest


def verify(out: Path) -> dict:
    manifest = json.loads((out / "archive_manifest.json").read_text())
    files = manifest.get("files")
    if (manifest.get("schema") != "mx_gemmini.nicolas_portable_chain_archive.v1" or
            not isinstance(files, dict) or not files):
        raise ValueError("portable chain archive manifest is malformed")
    if manifest.get("suite_sha256") != files.get("index.json", {}).get("sha256"):
        raise ValueError("portable chain archive suite digest differs")
    for name, record in files.items():
        stored = record["stored_as"]
        path = out / stored
        if not path.is_file() or (stored != name and stored != name + ".gz"):
            raise ValueError(f"portable chain archive is missing {name}")
        payload = path.read_bytes()
        data = gzip.decompress(payload) if stored.endswith(".gz") else payload
        if _sha(data) != record.get("sha256") or len(data) != record.get("bytes"):
            raise ValueError(f"portable chain archive artifact changed: {name}")
    suite = json.loads((out / "index.json").read_text())
    if (suite.get("schema") != "mx_gemmini.nicolas_portable_chain_suite.v1" or
            suite.get("status") !=
            "six_current_frontend_chains_matched_source_headers_on_pinned_spike" or
            suite.get("spike_extension_sha256") !=
            files["libgemmini.so"]["sha256"] or
            [row["case"] for row in suite.get("rows", [])] != list(CASES) or
            set(files) != {str(path) for path in _selected_archive(out)}):
        raise ValueError("portable chain archive case set differs")
    for row in suite["rows"]:
        case = row["case"]
        receipt_path = out / case / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        dimension = 64 if case.endswith("_64") else 128
        precision = case.split("_")[0].upper()
        if (row["receipt_sha256"] != _sha(receipt_path.read_bytes()) or
                files[f"{case}/model2mlir.mlir"]["sha256"] !=
                files[f"model2mlir_{dimension}.mlir"]["sha256"] or
                row["object_sha256"] != files[f"{case}/object/mx_issue.o"]["sha256"] or
                row["spike_log_sha256"] != files[f"{case}/run/spike.log"]["sha256"] or
                receipt.get("status") !=
                "compiler_matched_both_source_header_outputs_on_pinned_spike" or
                receipt.get("precision") != precision or
                receipt.get("dimension") != dimension or
                receipt.get("profile_sha256") != suite.get("profile_sha256") or
                receipt.get("compiler_spike", {}).get("matched") is not True or
                receipt.get("compiler_spike", {}).get("exit_code") != 0 or
                receipt.get("checked_outputs") != {
                    "c1_codes": receipt["dimension"] ** 2,
                    "c1_scales": receipt["dimension"] ** 2 // 32,
                    "c2_codes": receipt["dimension"] ** 2,
                    "c2_scales": receipt["dimension"] ** 2 // 32}):
            raise ValueError(f"portable chain archive result differs: {case}")
        for field, name in (("frontend_mlir_sha256", "model2mlir.mlir"),
                            ("frontend_binding_sha256", "frontend_binding.json"),
                            ("bound_mlir_sha256", "connected.mlir"),
                            ("object_manifest_sha256", "object/object_manifest.json"),
                            ("object_sha256", "object/mx_issue.o"),
                            ("physical_sha256", "object/physical_program.json")):
            if receipt.get(field) != files[f"{case}/{name}"]["sha256"]:
                raise ValueError(f"portable chain archive receipt changed: {case} {field}")
        elf_name = "run/mx_program.elf" if case.startswith("fp8") else "run/program.elf"
        if (receipt["compiler_spike"]["elf_sha256"] !=
                files[f"{case}/{elf_name}"]["sha256"] or
                receipt["compiler_spike"]["spike_log_sha256"] !=
                files[f"{case}/run/spike.log"]["sha256"]):
            raise ValueError(f"portable chain ELF or Spike log changed: {case}")
        log = (out / case / "run/spike.log").read_text()
        if precision == "FP8":
            marker = (f"lowered connected {dimension}x{dimension}: C1 0 codes 0 scales; "
                      "C2 0 codes 0 scales")
        elif precision == "FP4":
            marker = "fp4 chain test PASSED (MM1 resident + reused as MM2 operand, scales reused)."
        else:
            marker = ("fp6 chain test PASSED (MM1 resident + reused as MM2 operand, "
                      "LUT+scales reused).")
        if marker not in log:
            raise ValueError(f"portable chain Spike comparison marker is absent: {case}")
        obj = json.loads((out / case / "object/object_manifest.json").read_text())
        compiled = json.loads((out / case / "object/compile_manifest.json").read_text())
        bound = json.loads((out / case / "frontend_binding.json").read_text())
        if (compiled.get("lowering_family") != "resident_pair" or
                obj.get("profile_sha256") != suite["profile_sha256"] or
                obj.get("object_sha256") != receipt["object_sha256"] or
                obj.get("physical_program_sha256") != receipt["physical_sha256"] or
                obj.get("allocated_data_section_bytes") != 0 or
                obj.get("embedded_operand_bytes") != 0 or
                obj.get("embedded_golden_bytes") != 0 or
                obj.get("transport") != "rocket_rocc" or
                bound.get("schema") != "mx_gemmini.portable_chain_binding.v1" or
                bound.get("precision") != precision or
                bound.get("source_driver_sha256") != receipt["source_sha256"] or
                bound.get("source_header_sha256") != receipt["header_sha256"] or
                bound.get("source_mlir_sha256") != receipt["frontend_mlir_sha256"] or
                bound.get("profile_sha256") != suite["profile_sha256"]):
            raise ValueError(f"portable chain profile, source, or object binding differs: {case}")
        for name, digest in obj["input_sha256"].items():
            if files.get(f"{case}/resources/{name}.bin", {}).get("sha256") != digest:
                raise ValueError(f"portable chain input bytes differ: {case} {name}")
    return manifest


def _selected_archive(out: Path) -> list[Path]:
    """Derive required original paths from archived inputs, without task tmp."""
    paths = [Path("index.json"), Path("model2mlir_64.mlir"),
             Path("model2mlir_128.mlir"), Path("libgemmini.so")]
    for case in CASES:
        folder = Path(case)
        paths += [folder / name for name in (
            "model2mlir.mlir", "frontend_binding.json", "connected.mlir",
            "abi.json", "receipt.json", "object_compile.log",
            "object/compile_manifest.json", "object/object_manifest.json",
            "object/physical_program.json", "object/mx_issue.c",
            "object/mx_issue.h", "object/mx_issue.o", "run/spike.log")]
        paths += [folder / "run/mx_program.elf" if case.startswith("fp8")
                  else folder / "run/program.elf"]
        paths += ([folder / "run/mx_driver.c", folder / "run/mx_data.S"]
                  if case.startswith("fp8") else [folder / "compiler_driver.c"])
        paths += [Path(str(path.relative_to(out))[:-3])
                  for path in sorted((out / case / "resources").glob("*.bin.gz"))]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        verify(args.out_dir.resolve())
    elif args.source_dir is None:
        parser.error("--source-dir is required to create an archive")
    else:
        archive(args.source_dir.resolve(), args.out_dir.resolve())
    print(args.out_dir)


if __name__ == "__main__":
    main()
