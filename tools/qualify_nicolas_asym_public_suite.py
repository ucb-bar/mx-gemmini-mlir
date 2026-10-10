"""Replay all 74 pinned Nicolas asymmetric C programs with public MX objects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tools.qualify_nicolas_plain_matrix_object import (
    MODEL2MLIR_REVISION, MXQ_REVISION, ROOT, RTL_REVISION, _revision,
)


GROUPS = (
    ("dim8_64x64", 8, "64x64", False, 21),
    ("dim16_64x64", 16, "64x64", True, 26),
    ("dim32_64x64", 32, "64x64", False, 21),
    ("dim16_16x32", 16, "16x32", True, 1),
    ("dim16_128x128", 16, "128x128", True, 1),
    ("dim32_128x128x256", 32, "128x128x256", False, 4),
)
SCHEMA = "mx_gemmini.nicolas_asymmetric_public_object_suite.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(root: Path, group: tuple[str, int, str, bool, int],
            inventory: dict[str, str], compiler_revision: str) -> list[dict]:
    name, dim, shape, _, count = group
    directory = root / name
    matrix_path = directory / "matrix_receipt.json"
    matrix = json.loads(matrix_path.read_text())
    if (matrix["public_object"] is not True or
            matrix["mesh_dim"] != dim or matrix["source_shape"] != shape or
            matrix["selected_modes"] != count or matrix["passed_modes"] != count or
            len(matrix["rows"]) != count or
            any(row["status"] != "passed" for row in matrix["rows"])):
        raise ValueError(f"asymmetric public object group {name} is incomplete")
    rows = []
    for row in matrix["rows"]:
        source_name = f"matmul_tiled_asym_{row['source_suffix']}_{shape}"
        if dim != 16:
            source_name += f"_dim{dim}"
        case = directory / row["source_suffix"]
        receipt_path = case / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        obj_path = case / "object/mx_issue.o"
        dispatch_path = case / "object/compile_manifest.json"
        object_manifest_path = case / "object/object_manifest.json"
        elf_path = case / "physical/asymmetric_program.elf"
        spike_path = case / "physical/spike.log"
        dispatch = json.loads(dispatch_path.read_text())
        obj = json.loads(object_manifest_path.read_text())
        expected_outputs = {"16x32": 512, "64x64": 4096,
                            "128x128": 16384, "128x128x256": 16384}[shape]
        expected_shape = {"16x32": [16, 32, 32], "64x64": [64, 64, 64],
                          "128x128": [128, 128, 128],
                          "128x128x256": [128, 128, 256]}[shape]
        if (source_name not in inventory or
                receipt["status"] != "source_golden_matched_on_pinned_spike" or
                receipt["source_driver_sha256"] != inventory[source_name] or
                receipt["compiler_revision"] != compiler_revision or
                receipt["rtl_revision"] != RTL_REVISION or
                receipt["model2mlir_revision"] != MODEL2MLIR_REVISION or
                receipt["mxq_revision"] != MXQ_REVISION or
                receipt["spike_exit_code"] != 0 or
                receipt["compared_bf16_outputs"] != expected_outputs or
                obj["shape_mnk"] != expected_shape or
                row["matched_bf16_outputs"] != expected_outputs or
                row["receipt_sha256"] != _sha(receipt_path) or
                row["public_object_sha256"] != receipt["public_object_sha256"] or
                receipt["public_object_sha256"] != _sha(obj_path) or
                receipt["public_object_dispatch_sha256"] != _sha(dispatch_path) or
                receipt["public_object_manifest_sha256"] != _sha(object_manifest_path) or
                receipt["elf_sha256"] != _sha(elf_path) or
                receipt["spike_log_sha256"] != _sha(spike_path) or
                receipt["physical_program_sha256"] != _sha(
                    case / "physical/physical_program.json") or
                dispatch["lowering_family"] != "asymmetric_source" or
                obj["object_sha256"] != receipt["public_object_sha256"] or
                not any(slot["name"] == "output_bf16" and slot["role"] == "write"
                        for slot in obj["buffer_abi"]) or
                obj["allocated_data_section_bytes"] != 0 or
                obj["embedded_operand_bytes"] != 0 or
                obj["embedded_golden_bytes"] != 0):
            raise ValueError(f"asymmetric public object case {source_name} failed its audit")
        rows.append({
            "source_program": source_name,
            "source_driver_sha256": receipt["source_driver_sha256"],
            "source_header_sha256": receipt["source_header_sha256"],
            "profile_name": receipt["profile_name"],
            "compute": row["compute"],
            "mesh_dim": dim, "shape": shape,
            "compared_bf16_outputs": expected_outputs,
            "receipt": f"{name}/{row['source_suffix']}/receipt.json",
            "receipt_sha256": _sha(receipt_path),
            "object_sha256": receipt["public_object_sha256"],
            "elf_sha256": receipt["elf_sha256"],
            "spike_log_sha256": receipt["spike_log_sha256"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "mxq-root", "rtl-root", "riscv-root",
                 "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite {out}")
    rtl = args.rtl_root.resolve()
    if (_revision(rtl) != RTL_REVISION or
            _revision(args.model2mlir_root.resolve()) != MODEL2MLIR_REVISION or
            _revision(args.mxq_root.resolve()) != MXQ_REVISION):
        raise ValueError("asymmetric public suite needs pinned RTL and frontend revisions")
    roster_path = ROOT / "docs/evidence/nicolas_mx_source_inventory_266c593/index.json"
    roster = json.loads(roster_path.read_text())
    inventory = {entry["name"]: entry["source_sha256"] for entry in roster["entries"]
                 if entry["family"] == "asymmetric_matrix"}
    if len(inventory) != 74 or roster["rtl_revision"] != RTL_REVISION:
        raise ValueError("pinned Nicolas asymmetric source roster changed")
    compiler_revision = _revision(ROOT)
    out.mkdir(parents=True)
    (out / "source_inventory_baseline.json").write_bytes(roster_path.read_bytes())
    index = {
        "schema": SCHEMA, "status": "running",
        "scope": "all 74 pinned asymmetric C source numerical outputs via model2MLIR and public RV64 object",
        "compiler_revision": compiler_revision,
        "rtl_revision": RTL_REVISION,
        "model2mlir_revision": MODEL2MLIR_REVISION,
        "mxq_revision": MXQ_REVISION,
        "source_inventory_sha256": _sha(roster_path),
        "groups": [], "cases": [],
    }

    def write_index() -> None:
        (out / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    write_index()
    for group in GROUPS:
        name, dim, shape, all_asym, count = group
        command = [
            sys.executable, "-m", "tools.qualify_nicolas_asym_matrix",
            "--mesh-dim", str(dim), "--source-shape", shape,
            *(["--all-asym"] if all_asym else []), "--public-object",
            "--jobs", str(args.jobs),
            "--model2mlir-root", str(args.model2mlir_root.resolve()),
            "--mxq-root", str(args.mxq_root.resolve()),
            "--rtl-root", str(rtl),
            "--riscv-root", str(args.riscv_root.resolve()),
            "--mx-opt", str(args.mx_opt.resolve()),
            "--out-dir", str(out / name),
        ]
        print(f"replaying {name}: {count} sources", flush=True)
        run = subprocess.run(command, cwd=ROOT, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             check=False)
        (out / f"{name}.log").write_text(run.stdout)
        if run.returncode:
            index["status"], index["failed_group"] = "failed", name
            write_index()
            raise RuntimeError(f"asymmetric public group {name} failed: {out / f'{name}.log'}")
        rows = _record(out, group, inventory, compiler_revision)
        index["groups"].append({
            "name": name, "selected_sources": count,
            "matched_bf16_outputs": sum(row["compared_bf16_outputs"] for row in rows),
            "matrix_receipt": f"{name}/matrix_receipt.json",
            "matrix_receipt_sha256": _sha(out / name / "matrix_receipt.json"),
        })
        index["cases"].extend(rows)
        write_index()
        print(f"matched {name}: {len(rows)} sources", flush=True)
    if ({row["source_program"] for row in index["cases"]} != set(inventory) or
            len(index["cases"]) != 74):
        raise ValueError("public asymmetric suite did not cover the complete pinned roster")
    index["status"] = "all_pinned_asymmetric_source_goldens_matched_on_pinned_spike"
    index["matched_sources"] = 74
    index["total_bf16_outputs_checked"] = sum(
        row["compared_bf16_outputs"] for row in index["cases"])
    write_index()
    print(f"matched 74 asymmetric sources and {index['total_bf16_outputs_checked']} BF16 outputs",
          flush=True)


if __name__ == "__main__":
    main()
