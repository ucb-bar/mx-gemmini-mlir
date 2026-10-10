"""Rebuild the two pinned Radiance read-once MX data blobs without overwriting them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


SOURCE_REVISION = "ee22e0b87180436cd0fa1583c411a3cf49d7586a"
GENERATOR_SHA256 = "5e405988edf10a9c25a12a6111599cfa93963b0e2586bc9fc2fad30f7b4abef7"
CASES = (
    ("gemm_mxgemmini_ws", "fp8", (256, 64, 2048),
     "d1cca6d8b12d875a5cffa73ade73976ebaaa9e4db1323b9d385f23dc8fb11384",
     "a3fad0f73fd71a8f93d06467f2f653f86bdf5c3405933217983cbf9683329f43"),
    ("gemm_mxgemmini_ws_downproj_fp4", "fp4", (256, 64, 5632),
     "4e38f4f040ade76bc0c1584fe3374d719152087b0c9d21fd56ed2c06585b0678",
     "4c023254973cdad850d6138c5bc7b0375554d662f1f935537d907ce9f3c5c472"),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(source_root: Path) -> dict:
    root = source_root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    generator = root / "kernels/gemm_mxgemmini/gen_mxgemm_data.py"
    if revision != SOURCE_REVISION or _sha(generator) != GENERATOR_SHA256:
        raise ValueError("selected Radiance source or MX data generator differs from pinned revision")
    rows = []
    generated = []
    for directory, precision, (m, n, k), driver_sha, data_sha in CASES:
        driver = root / "kernels" / directory / "kernel.cpp"
        target = driver.parent / "data"
        header = generator.parent / f"mxgemm.data.{precision}.m{m}n{n}k{k}.h"
        if _sha(driver) != driver_sha:
            raise ValueError(f"source read-once driver changed: {directory}")
        if target.exists() and _sha(target) != data_sha:
            raise ValueError(f"refusing to replace changed source data: {target}")
        if not target.exists():
            if not header.exists():
                subprocess.run(["make", "-C", str(root / "lib/golden"), "mx_golden"],
                               check=True)
                subprocess.run([sys.executable, str(generator), precision, str(m), str(n), str(k)],
                               cwd=generator.parent, check=True)
                generated.append(str(header.relative_to(root)))
            if _sha(header) != data_sha:
                raise ValueError(f"generated source header differs from pinned golden: {header}")
            shutil.copy2(header, target)
            generated.append(str(target.relative_to(root)))
        rows.append({"driver": str(driver.relative_to(root)),
                     "driver_sha256": driver_sha,
                     "data": str(target.relative_to(root)),
                     "data_sha256": _sha(target),
                     "precision": precision, "shape_mnk": [m, n, k]})
    return {"schema": "mx_gemmini.radiance_ws_data_materialization.v1",
            "source_revision": revision,
            "generator_sha256": GENERATOR_SHA256,
            "generated": generated, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    report = materialize(args.source_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"verified {len(report['rows'])} read-once MX source blobs -> {args.out}")


if __name__ == "__main__":
    main()
