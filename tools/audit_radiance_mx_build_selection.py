"""Separate source Makefile selection from MX compiler numerical parity.

The complete 31-driver roster contains source recipes that Radiance excludes
from its own 128 KiB build. This audit checks every current driver byte against
the archived model2MLIR and Spike rosters, then records which drivers its
Makefile actually selects. It does not claim a source ELF was executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from mx_gemmini_support.source_build_selection import source_makefile_selection


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def audit(source_root: Path, frontend_index: Path, spike_index: Path) -> dict:
    frontend_bytes, spike_bytes = frontend_index.read_bytes(), spike_index.read_bytes()
    frontend, spike = json.loads(frontend_bytes), json.loads(spike_bytes)
    if (frontend.get("schema") != "mx_gemmini.radiance_frontend_roster.v1" or
            spike.get("schema") != "mx_gemmini.radiance_mx_gemm_source_parity.v2" or
            frontend.get("source_revision") != spike.get("source_revision") or
            frontend.get("captured_drivers") != 31 or
            spike.get("covered_drivers") != 31):
        raise ValueError("selected MX frontend and Spike rosters are not a matching 31-driver set")
    frontend_rows = {row["driver"]: row for row in frontend["rows"]}
    spike_rows = {row["driver"]: row for row in spike["rows"]}
    names = {Path(driver).name for driver in frontend_rows}
    if (len(frontend_rows) != 31 or len(spike_rows) != 31 or
            set(frontend_rows) != set(spike_rows) or len(names) != 31 or
            any(Path(driver).parent != Path("kernels/gemm_mxgemmini")
                for driver in frontend_rows)):
        raise ValueError("MX roster contains a duplicate, missing, or unrelated driver")
    selected, makefile = source_makefile_selection(source_root, names)
    rows = []
    for driver in sorted(frontend_rows):
        captured, qualified = frontend_rows[driver], spike_rows[driver]
        digest = _sha((source_root / driver).read_bytes())
        if (captured["driver_sha256"] != digest or
                qualified["source_driver_sha256"] != digest or
                captured["header_sha256"] != qualified["source_header_sha256"] or
                qualified["status"] not in {
                    "source_golden_matched_on_pinned_spike",
                    "radiance_header_matched_on_pinned_spike"}):
            raise ValueError(f"MX source driver or numerical receipt differs: {driver}")
        rows.append({
            "driver": driver,
            "source_driver_sha256": digest,
            "source_makefile_selected": Path(driver).name in selected,
            "precision": captured["precision"],
            "shape_mnk": captured["shape_mnk"],
            "tile_mnk": captured["tile_mnk"],
            "quant_output": captured["quant_output"],
            "spike_status": qualified["status"],
        })
    return {
        "schema": "mx_gemmini.radiance_mx_build_selection_audit.v1",
        "source_revision": _revision(source_root),
        "roster_source_revision": frontend["source_revision"],
        "source_makefile_sha256": _sha(makefile),
        "frontend_index_sha256": _sha(frontend_bytes),
        "spike_index_sha256": _sha(spike_bytes),
        "named_driver_count": len(rows),
        "source_makefile_selected_count": len(selected),
        "source_recipe_only_count": len(rows) - len(selected),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--frontend-index", required=True, type=Path)
    parser.add_argument("--spike-index", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    report = audit(args.source_root.resolve(), args.frontend_index.resolve(),
                   args.spike_index.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{report['source_makefile_selected_count']} source Makefile selections; "
          f"{report['source_recipe_only_count']} source recipes only; "
          f"{report['named_driver_count']} Spike matched")


if __name__ == "__main__":
    main()
