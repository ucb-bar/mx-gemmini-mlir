"""Audit current Radiance MX GEMM drivers against the source and selected RTL geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from mx_gemmini_support.source_gemm import (plan_source_gemm, read_source_gemm,
                                            source_scratchpad_bytes)
from mx_gemmini_support.target_profile import load_profile, profile_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _active_sources(makefile: Path) -> set[str]:
    text = makefile.read_text()
    section = re.search(r'^MU_SRCS\s*=(.*?)^HOST_SRCS', text, re.M | re.S)
    if section is None:
        raise ValueError("source MX GEMM Makefile lacks an MU_SRCS roster")
    return set(re.findall(r'\b[\w.]+\.cpp\b', section.group(1)))


def audit(source_root: Path, profile: dict | None = None) -> dict:
    source_root = source_root.resolve()
    directory = source_root / "kernels/gemm_mxgemmini"
    library = source_root / "lib/mxgemm/mxgemm_lib.hpp"
    makefile = directory / "Makefile"
    source_bytes = source_scratchpad_bytes(library)
    active = _active_sources(makefile)
    records = []
    for driver in sorted(directory.glob("mxgemm.*.cpp")):
        row = {"driver": driver.name, "active_in_makefile": driver.name in active}
        try:
            kernel = read_source_gemm(driver)
            row["data_header_present"] = kernel.data_header_present
            row["source_plan"] = plan_source_gemm(kernel, scratchpad_bytes=source_bytes)
            row["source_status"] = ("layout_feasible_with_data" if kernel.data_header_present
                                    else "layout_feasible_missing_data")
            if profile is not None:
                try:
                    row["target_plan"] = plan_source_gemm(
                        kernel, scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
                        profile=profile)
                    row["target_status"] = "structural_plan_only"
                except ValueError as exc:
                    row["target_status"] = "unsupported"
                    row["target_reason"] = str(exc)
        except (OSError, ValueError) as exc:
            row["source_status"] = "unsupported"
            row["source_reason"] = str(exc)
        records.append(row)
    observed = {row["driver"] for row in records}
    report = {
        "schema": "mx_gemmini.source_kernel_match.v1",
        "scope": "analytical_tile_layout_and_k_schedule",
        "source_revision": subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip(),
        "source_library_sha256": _sha(library),
        "source_makefile_sha256": _sha(makefile),
        "source_scratchpad_bytes": source_bytes,
        "target_profile_name": profile["name"] if profile else None,
        "target_profile_sha256": profile_sha256(profile) if profile else None,
        "active_sources_outside_shape_ladder": sorted(active - observed),
        "drivers": records,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--rtl-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.profile) != bool(args.rtl_root):
        parser.error("--profile and --rtl-root must be supplied together")
    profile = load_profile(args.profile, rtl_root=args.rtl_root) if args.profile else None
    report = audit(args.source_root, profile)
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    counts = {}
    for row in report["drivers"]:
        counts[row["source_status"]] = counts.get(row["source_status"], 0) + 1
    print(json.dumps({"drivers": len(report["drivers"]), "source_status": counts,
                      "output": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
