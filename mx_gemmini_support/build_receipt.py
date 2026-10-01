"""Verify present source and artifact bytes against an MX build diagnostic receipt.

A receipt is a declaration made after a build. Matching current bytes cannot
prove which inputs an earlier tool invocation actually read.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from .config_facts import selected_config_facts
from .contract import compile_contract


REQUIRED_ARTIFACTS = {"firrtl", "simulator", "gemmini_header", "annotations"}


def _head(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _contained(root: Path, member: str) -> Path:
    path = (root / member).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"receipt path escapes checkout: {member}")
    return path


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _source_census(gemmini: Path, census: dict) -> str:
    recorded = census["sha256_by_relative_path"]
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("empty or invalid Scala source census")
    actual = {
        path.relative_to(gemmini).as_posix()
        for subtree in ("src/main/scala", "mxgen/src/main/scala")
        for path in (gemmini / subtree).rglob("*.scala")
    }
    if actual != set(recorded) or census["scala_files"] != len(actual):
        raise ValueError("Scala source census omits or adds files")
    for member, expected in recorded.items():
        if _file_digest(_contained(gemmini, member))[0] != expected:
            raise ValueError(f"Scala source differs from build receipt: {member}")
    manifest = hashlib.sha256("".join(
        f"{member} {recorded[member]}\n" for member in sorted(recorded)
    ).encode()).hexdigest()
    if manifest != census["manifest_sha256"]:
        raise ValueError("Scala source census digest differs from build receipt")
    return manifest


def _mesh_grid(firrtl: Path) -> dict:
    """Derive the selected MX tile coordinates from elaborated FIRRTL instances."""
    mesh_modules = 0
    inside = False
    coordinates = set()
    with firrtl.open(encoding="utf-8") as stream:
        for line in stream:
            module = re.match(r"^  module (\S+)\s*:", line)
            if module:
                inside = module.group(1) == "Mesh"
                mesh_modules += int(inside)
                continue
            if not inside or not line.startswith("    inst mesh_"):
                continue
            tile = re.match(r"^    inst mesh_(\d+)_(\d+) of Tile(?:_\d+)?\b", line)
            if tile is None:
                raise ValueError("unparsed selected Mesh tile instance")
            coordinate = int(tile.group(1)), int(tile.group(2))
            if coordinate in coordinates:
                raise ValueError("duplicate selected Mesh tile coordinate")
            coordinates.add(coordinate)
    if mesh_modules != 1 or not coordinates:
        raise ValueError("selected FIRRTL lacks one populated Mesh module")
    rows = max(row for row, _ in coordinates) + 1
    columns = max(column for _, column in coordinates) + 1
    if len(coordinates) != rows * columns or any(
        (row, column) not in coordinates
        for row in range(rows) for column in range(columns)
    ):
        raise ValueError("selected FIRRTL Mesh tile coordinates are not a dense grid")
    return {"rows": rows, "columns": columns, "tiles": len(coordinates),
            "basis": "FIRRTL Mesh mesh_row_column Tile instances"}


def _header_integer(header: Path, name: str) -> int:
    matches = re.findall(rf"^#define {re.escape(name)}\s+(\d+)\s*$", header.read_text(), re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"generated header lacks one literal {name}")
    return int(matches[0])


def verify_build_receipt(
    chipyard_root: str | Path, receipt_bytes: bytes, contract_bytes: bytes,
) -> dict:
    """Check current bytes and commits; do not certify historical build inputs."""
    root = Path(chipyard_root).resolve()
    receipt = json.loads(receipt_bytes)
    contract = compile_contract(contract_bytes)
    if receipt.get("schema") != "mx_gemmini.latest_rtl_build_diagnostic.v1":
        raise ValueError("unexpected MX build receipt schema")
    gemmini = (root / "generators/gemmini").resolve()
    if not gemmini.is_relative_to(root):
        raise ValueError("Gemmini checkout escapes Chipyard root")
    if (_head(root) != receipt["chipyard_commit"] or
        _head(gemmini) != receipt["gemmini_checkout_override_commit"] or
        _head(gemmini / "mxgen") != receipt["mxgen_commit"]):
        raise ValueError("present checkout commits differ from build receipt")
    if (receipt["gemmini_checkout_override_commit"] != contract["rtl_commit"] or
        receipt["mxgen_commit"] != contract["mxgen_commit"] or
        receipt["config_object"] != contract["rtl_config"] or
        receipt["config_classes"][0] != f"gemmini.{contract['rtl_config_class']}"):
        raise ValueError("build receipt selects a different software contract")
    gitlink = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD:generators/gemmini"], text=True,
    ).strip()
    if gitlink != receipt["chipyard_gemmini_gitlink"]:
        raise ValueError("Chipyard Gemmini gitlink differs from build receipt")
    manifest = _source_census(gemmini, receipt["source_census"])
    artifacts = receipt["artifacts"]
    if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS.issubset(artifacts):
        raise ValueError("build receipt omits required artifacts")
    checked = {}
    for name, row in artifacts.items():
        digest, size = _file_digest(_contained(root, row["relative_path"]))
        if digest != row["sha256"] or size != row["bytes"]:
            raise ValueError(f"{name} differs from build receipt")
        checked[name] = {"sha256": digest, "bytes": size}
    mesh = _mesh_grid(_contained(root, artifacts["firrtl"]["relative_path"]))
    config = selected_config_facts((gemmini / "src/main/scala/gemmini/ConfigsFP.scala").read_text())
    header = _contained(root, artifacts["gemmini_header"]["relative_path"])
    if (mesh["rows"] != config["meshRows"]["value"] or
        mesh["columns"] != config["meshColumns"]["value"] or
        mesh["rows"] != _header_integer(header, "DIM") or
        config["sp_banks"]["value"] != _header_integer(header, "BANK_NUM")):
        raise ValueError("elaborated Mesh, selected Scala config, and generated header disagree")
    return {
        "schema": "mx_gemmini.build_receipt_audit.v1",
        "status": "observed_consistency",
        "gemmini_commit": contract["rtl_commit"],
        "mxgen_commit": contract["mxgen_commit"],
        "scala_files": receipt["source_census"]["scala_files"],
        "scala_manifest_sha256": manifest,
        "artifacts": checked,
        "elaborated_mesh": mesh,
        "historical_build_provenance_verified": False,
        "phase0_admitted": False,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audit current bytes of an MX build receipt")
    parser.add_argument("chipyard_root")
    parser.add_argument("receipt")
    parser.add_argument("contract")
    args = parser.parse_args()
    report = verify_build_receipt(
        args.chipyard_root, Path(args.receipt).read_bytes(), Path(args.contract).read_bytes(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
