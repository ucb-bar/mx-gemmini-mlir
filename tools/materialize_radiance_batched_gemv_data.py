"""Rebuild pinned Radiance batched GEMV data and verify every source byte."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tools.materialize_radiance_ws_data import SOURCE_REVISION, GENERATOR_SHA256


CASES = (
    ("gemv_batched_fp8_m32", "fp8", 32,
     "393ad997812814bfffdc4f8981a1c53dfa6378c790b644bd279f1fd34ec65361",
     "7b72d6d5ef0a69d11aa6df25ee3c191fedee7514bed580e933af216c8aab1398"),
    ("gemv_batched_fp8_m64", "fp8", 64,
     "b7cc19524f6a45fbe5c9100fcdf5aace85a25d08c3edec2fd6dd41841e8db56c",
     "8705b3f68cd36001125d0cc7c74c79e02b81e78b6c67ffbdf8643c6973cbf4b0"),
    ("gemv_batched_fp8_m128", "fp8", 128,
     "6ea00c259362570a03a36c83a897e3c08c02f468371c10264befaac6340ad51c",
     "8ccb145e95461b0b25c8f135a221a665724a928d0217cbe0c3b968d7d89db7f2"),
    ("gemv_batched_fp4_m128", "fp4", 128,
     "a96cc0ce5e77570b80d6b309b5b0a1928ee4462375254b2b8da853cc551e6049",
     "e13bd3e79be68ce91592f5d4829c6417fe0ce05281b6df17b16f838391910d64"),
)

VERIFY_GLUE = (b"__global uint16_t C_raw[MATMUL_M*MATMUL_N]={0};\n"
               b"static const uint16_t* gold_raw=&C_out_bf16[0][0];\n"
               b"#define VERIFY_COUNT (MATMUL_M*MATMUL_N)\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(source_root: Path) -> dict:
    root = source_root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    generator = root / "kernels/gemm_mxgemmini/gen_mxgemm_data.py"
    if revision != SOURCE_REVISION or _sha(generator) != GENERATOR_SHA256:
        raise ValueError("selected Radiance source or generator differs from pinned revision")
    rows = []
    generated = []
    for directory, precision, m, driver_sha, data_sha in CASES:
        driver = root / "kernels" / directory / "kernel.cpp"
        target = driver.parent / "data"
        header = generator.parent / f"mxgemm.data.{precision}.m{m}n128k2048.h"
        if _sha(driver) != driver_sha:
            raise ValueError(f"source batched GEMV driver changed: {directory}")
        if target.exists() and _sha(target) != data_sha:
            raise ValueError(f"refusing to replace changed source data: {target}")
        if not target.exists():
            if not header.exists():
                subprocess.run(["make", "-C", str(root / "lib/golden"), "mx_golden"],
                               check=True)
                subprocess.run([sys.executable, str(generator), precision, str(m), "128", "2048"],
                               cwd=generator.parent, check=True)
                generated.append(str(header.relative_to(root)))
            payload = header.read_bytes() + VERIFY_GLUE
            if hashlib.sha256(payload).hexdigest() != data_sha:
                raise ValueError(f"generated source data differs from pinned golden: {directory}")
            target.write_bytes(payload)
            generated.append(str(target.relative_to(root)))
        rows.append({"driver": str(driver.relative_to(root)),
                     "driver_sha256": driver_sha,
                     "data": str(target.relative_to(root)),
                     "data_sha256": _sha(target),
                     "precision": precision, "shape_mnk": [m, 128, 2048]})
    return {"schema": "mx_gemmini.radiance_batched_gemv_data_materialization.v1",
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
    print(f"verified {len(report['rows'])} batched GEMV source blobs -> {args.out}")


if __name__ == "__main__":
    main()
