"""Generate only missing Radiance MX GEMM headers for the source roster.

Radiance's --all-fp8 command also rewrites a committed 128x128x512 header with
different contents. Preserve every existing header and generate only those
referenced by a driver but absent in the selected checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.source_gemm import read_source_gemm
from tools.generate_radiance_fp6_header import generate as generate_fp6


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--expected-index", type=Path,
                        help="frontend roster index whose driver/header hashes must match")
    parser.add_argument("--out", type=Path, help="write a JSON materialization report")
    args = parser.parse_args()
    source = args.source_root.resolve()
    directory = source / "kernels/gemm_mxgemmini"
    drivers = sorted(directory.glob("mxgemm.fp*.cpp"))
    if len(drivers) != 31:
        raise ValueError(f"expected 31 Radiance MX GEMM drivers, found {len(drivers)}")
    golden = source / "lib/golden/mx_golden"
    if not golden.is_file():
        raise ValueError(f"build Radiance's golden tool first: make -C {golden.parent} mx_golden")
    generator = directory / "gen_mxgemm_data.py"
    expected = None
    if args.expected_index:
        index = json.loads(args.expected_index.read_text())
        if (index.get("schema") != "mx_gemmini.radiance_frontend_roster.v1" or
                index.get("captured_drivers") != 31):
            raise ValueError("expected index is not a full Radiance MX frontend roster")
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if revision != index["source_revision"]:
            raise ValueError("selected Radiance checkout differs from pinned source")
        expected = {row["driver"]: row for row in index["rows"]}
    rows = []
    for driver in drivers:
        kernel = read_source_gemm(driver)
        header = kernel.data_header
        existed = header.is_file()
        if not existed:
            if kernel.datatype == "FP6":
                if kernel.shape[:2] != (128, 128):
                    raise ValueError(f"unsupported FP6 header shape: {kernel.shape}")
                generate_fp6(source, kernel.shape[2], header)
            elif kernel.datatype in ("FP8", "FP4"):
                fmt = kernel.datatype.lower()
                result = subprocess.run(
                    [sys.executable, str(generator), fmt,
                     *(str(dimension) for dimension in kernel.shape)],
                    cwd=source, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    raise RuntimeError(f"Radiance header generation failed for {header}:\n"
                                       f"{result.stdout}")
            else:
                raise ValueError(f"unsupported source precision: {kernel.datatype}")
            print(f"generated {header.name}", file=sys.stderr, flush=True)
        name = str(driver.relative_to(source))
        row = {"driver": name, "driver_sha256": _sha(driver),
               "header": str(header.relative_to(source)),
               "header_sha256": _sha(header),
               "origin": "existing" if existed else "generated"}
        if expected:
            pinned = expected.get(name)
            if (pinned is None or row["driver_sha256"] != pinned["driver_sha256"] or
                    row["header_sha256"] != pinned["header_sha256"]):
                raise ValueError(f"source differs from pinned frontend roster: {name}")
        rows.append(row)
    if expected and len(expected) != len(rows):
        raise ValueError("selected source driver set differs from pinned frontend roster")
    report = {"schema": "mx_gemmini.radiance_header_materialization.v1",
              "drivers": len(rows),
              "generated_headers": sorted({row["header"] for row in rows
                                           if row["origin"] == "generated"}),
              "rows": rows}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
