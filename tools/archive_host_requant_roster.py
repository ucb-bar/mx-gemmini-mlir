"""Archive two byte-identical source-derived MX host-object Spike replays."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy(source: Path, target: Path, *, compress: bool = False) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        target.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
    else:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-dir", "fresh-dir", "source-root", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("run_dir", "fresh_dir", "source_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    primary = json.loads((args.run_dir / "index.json").read_text())
    fresh = json.loads((args.fresh_dir / "index.json").read_text())
    if (primary != fresh or len(primary.get("cases", [])) != 8 or
            primary.get("compared_codes") != 90112 or
            primary.get("compared_scales") != 3328 or
            primary.get("status") !=
            "all_8_requant_objects_matched_source_goldens_on_pinned_spike"):
        raise ValueError("host-object replays or full-output counts differ")
    source_index = args.source_root / "index.json"
    if primary.get("source_roster_index_sha256") != _sha(source_index):
        raise ValueError("host-object replay differs from selected source roster")
    names = (
        "object/object_manifest.json", "object/compile_manifest.json",
        "object/physical_program.json", "object/mx_issue.c", "object/mx_issue.h",
        "object/mx_issue.o", "spike/index.json", "spike/spike.log",
        "spike/mx_driver.c", "spike/mx_host_object.elf",
    )
    args.out_dir.mkdir(parents=True)
    _copy(args.run_dir / "index.json", args.out_dir / "index.json")
    copied = []
    for row in primary["cases"]:
        case = row["case"]
        for rel in names:
            original = args.run_dir / case / rel
            replay = args.fresh_dir / case / rel
            if _sha(original) != _sha(replay):
                raise ValueError(f"fresh checkout differs at {case}/{rel}")
            archived = args.out_dir / "cases" / case / rel
            compress = rel.endswith((".c", ".o", ".elf"))
            if compress:
                archived = archived.with_name(archived.name + ".gz")
            _copy(original, archived, compress=compress)
            copied.append({"path": str(archived.relative_to(args.out_dir)),
                           "raw_sha256": _sha(original)})
        source_case = args.source_root / case
        for rel in ("payload_bound.mlir", "bundle/manifest.json"):
            source = source_case / rel
            archived = args.out_dir / "source" / case / rel
            compress = rel.endswith(".mlir")
            if compress:
                archived = archived.with_name(archived.name + ".gz")
            _copy(source, archived, compress=compress)
            copied.append({"path": str(archived.relative_to(args.out_dir)),
                           "raw_sha256": _sha(source)})
    evidence = {
        "schema": "mx_gemmini.runtime_host_object_evidence.v1",
        "status": "fresh_checkout_reproduced_all_8_host_objects_and_spike_logs",
        "compiler_revision": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True).strip(),
        "index_sha256": _sha(args.run_dir / "index.json"),
        "fresh_index_sha256": _sha(args.fresh_dir / "index.json"),
        "files": copied,
    }
    (args.out_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"archived {len(primary['cases'])} fresh-reproduced host objects")


if __name__ == "__main__":
    main()
