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


def _mesh_compute_hierarchy(firrtl: Path, mesh: dict) -> dict:
    """Follow each selected tile through the elaborated multiplier hierarchy.

    Module names and instance containment establish structure, not a connected
    arithmetic path or a numerical contract.
    """
    children: dict[str, list[tuple[str, str]]] = {}
    module = None
    with firrtl.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("  module "):
                module = line.split()[1]
                if module == "Mesh" or any(
                    module == stem or module.startswith(stem + "_")
                    for stem in ("Tile", "PE", "MacUnit", "MxFpMul")
                ):
                    if module in children:
                        raise ValueError(f"duplicate selected FIRRTL module {module}")
                    children[module] = []
                continue
            if module not in children or not line.startswith("    inst "):
                continue
            parts = line.split()
            if len(parts) < 4 or parts[2] != "of":
                raise ValueError(f"unparsed instance in {module}")
            children[module].append((parts[1], parts[3]))
    selected = {
        f"mesh_{row}_{column}" for row in range(mesh["rows"])
        for column in range(mesh["columns"])
    }
    mesh_children = children.get("Mesh", [])
    tiles = {name: child for name, child in mesh_children if name in selected}
    if len(tiles) != mesh["tiles"] or set(tiles) != selected:
        raise ValueError("selected Mesh tile instances differ from grid")
    if any(name.startswith("mesh_") and name not in selected for name, _ in mesh_children):
        raise ValueError("selected Mesh has unexpected tile coordinates")

    def one_child(parent: str, stem: str) -> str:
        matches = [child for _, child in children.get(parent, [])
                   if child == stem or child.startswith(stem + "_")]
        if len(matches) != 1 or matches[0] not in children:
            raise ValueError(f"{parent}: expected one elaborated {stem} child")
        return matches[0]

    paths = []
    fused_counts = set()
    for name in sorted(selected):
        tile = tiles[name]
        if tile != "Tile" and not re.fullmatch(r"Tile_\d+", tile):
            raise ValueError(f"{name}: selected Mesh child is not a Tile")
        pe = one_child(tile, "PE")
        mac = one_child(pe, "MacUnit")
        mul = one_child(mac, "MxFpMul")
        fused = sorted(child for _, child in children[mul]
                       if child == "MxMulAddRecFN" or child.startswith("MxMulAddRecFN_"))
        if not fused:
            raise ValueError(f"{mul}: no elaborated fused multiply-add child")
        fused_counts.add(len(fused))
        paths.append((name, tile, pe, mac, mul, fused))
    if len(fused_counts) != 1:
        raise ValueError("selected tiles have nonuniform fused multiplier hierarchy")
    manifest = "".join(" ".join((name, tile, pe, mac, mul, *fused)) + "\n"
                       for name, tile, pe, mac, mul, fused in paths)
    return {
        "tiles_with_hierarchy": len(paths),
        "fused_units_per_tile": fused_counts.pop(),
        "path_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "basis": "FIRRTL Mesh→Tile→PE→MacUnit→MxFpMul→MxMulAddRecFN instances",
        "connected_arithmetic_verified": False,
    }


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
    selected_firrtl = _contained(root, artifacts["firrtl"]["relative_path"])
    mesh = _mesh_grid(selected_firrtl)
    hierarchy = _mesh_compute_hierarchy(selected_firrtl, mesh)
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
        "mesh_compute_hierarchy": hierarchy,
        "selected_config_features": {
            name: config[name]
            for name in ("has_nonlinear_activations", "has_normalizations", "has_max_pool",
                         "enable_lut", "lut_present")
        },
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
