"""Relink one compiled stage with Nicolas's native MX matrix issuers.

The compiled host graph, quantization, VPU objects, and stage ABI stay in place.
This is a matrix-issuer comparison, not a fused handwritten model baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command: list[str], log: Path | None = None) -> None:
    if log is None:
        subprocess.run(command, check=True)
    else:
        with log.open("w") as output:
            subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--npu-root", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, required=True)
    parser.add_argument("--riscv-root", type=Path, required=True)
    parser.add_argument("--max-m-tile", type=int, default=128)
    parser.add_argument("--object-only", action="store_true",
                        help="emit only the replacement catalog, without a stage ELF")
    args = parser.parse_args()
    stage = args.stage.resolve(strict=True)
    cat = stage / "device_catalog"
    manifest = cat / "mx_catalog.json"
    entries = json.loads(manifest.read_text())["kernels"]
    issuers = {}
    for entry in entries:
        name = entry["schedule_file"].removesuffix(".mk")
        previous = issuers.setdefault(name, entry)
        if (previous["precision"], previous["padded_shape_mnk"]) != (
                entry["precision"], entry["padded_shape_mnk"]):
            raise ValueError(f"inconsistent shared issuer {name}")
    if not issuers or args.max_m_tile <= 0 or args.max_m_tile % 32:
        raise ValueError("native source requires aligned nonempty issuer tiles")
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    header = args.npu_root / "baremetal/mxgemmini/include/mx_native.h"
    assert header.is_file()
    lines = [
        "#include <stdint.h>", "#include <stdio.h>", "#define DIM 16",
        "#define BANK_ROWS 4096", "#define SPIKE_SIM 1",
        "#define printf(...) ((void)0)",
        '#include "baremetal/mxgemmini/include/mx_native.h"',
    ]
    for name, entry in sorted(issuers.items(), key=lambda pair: int(pair[0].split("_")[-1])):
        m, n, k = entry["padded_shape_mnk"]
        fmt = entry["precision"]
        # Nicolas's FP8 loop accepts DIM16 rows/columns; FP4's packed
        # two-lane mesh requires 2*DIM in both output axes.
        output_alignment = 32 if fmt == "mxfp4" else 16
        if (fmt not in {"mxfp4", "mxfp8"} or m % output_alignment
                or n % output_alignment or k % 32):
            raise ValueError(f"unsupported source shape: {name} {fmt} {m,n,k}")
        tile = min(m, args.max_m_tile)
        if m % tile:
            raise ValueError(f"source row tile must divide M: {name} {m} % {tile}")
        if fmt == "mxfp4":
            call = (f"mxn_matmul_fp4((const uint8_t *)a + row/2*{k}, {k}, "
                    f"(const uint8_t *)b, {n//2}, (uint16_t *)c + row*{n}, {n}, "
                    f"(const uint8_t *)as + row, {m}, (const uint8_t *)bs, {n}, "
                    f"{tile}, {k}, {n})")
        else:
            call = (f"mxn_matmul_ex((const uint8_t *)a + row*{k}, {k}, "
                    f"(const uint8_t *)b, {n}, (uint16_t *)c + row*{n}, {n}, "
                    f"(const uint8_t *)as + row, {m}, (const uint8_t *)bs, {n}, "
                    f"{tile}, {k}, {n}, 0, 0)")
        lines.extend((
            f"void {name}(const void *a,const void *as,const void *b,const void *bs,"
            "const void *c,const void *ss) {",
            "  (void)ss;",
            f"  for (int row = 0; row < {m}; row += {tile}) {{",
            f"    if ({call}) __builtin_trap();",
            "  }",
            "  gemmini_fence();",
            "}",
        ))
    source = out / "nicolas_issuers.c"
    source.write_text("\n".join(lines) + "\n")
    bin_dir = args.riscv_root / "bin"
    cc, ld, nm = (bin_dir / f"riscv64-unknown-elf-{tool}" for tool in ("gcc", "ld", "nm"))
    issuer_obj = out / "nicolas_issuers.o"
    run([str(cc), "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2",
         "-ffreestanding", "-fno-fast-math", "-fno-common", "-std=gnu11",
         "-DSPIKE_SIM", "-I", str(args.npu_root),
         "-I", str(args.rtl_root / "software/gemmini-rocc-tests"),
         "-c", str(source), "-o", str(issuer_obj)], out / "compile.log")
    parts = sorted(path for path in cat.glob("*.o") if path.name != "mx_catalog.o"
                   and not path.name.startswith("mx_issue_"))
    catalog_obj = out / "mx_catalog_nicolas.o"
    run([str(ld), "-r", "-o", str(catalog_obj),
         *(str(path) for path in parts), str(issuer_obj)], out / "catalog_link.log")
    unresolved = subprocess.run([str(nm), "-u", str(catalog_obj)],
                                check=True, capture_output=True, text=True).stdout
    if unresolved:
        raise ValueError(f"Nicolas catalog has undefined symbols: {unresolved[:1000]}")
    elf = None
    if not args.object_only:
        recipe = json.loads((stage / "compilation_recipe.json").read_text())
        command = list(recipe["commands"][-1]["argv"])
        command[command.index(str(cat / "mx_catalog.o"))] = str(catalog_obj)
        elf = out / "model.elf"
        command[command.index(str(stage / "model.elf"))] = str(elf)
        run(command, out / "stage_link.log")
    receipt = {
        "schema": "mx_gemmini.nicolas_issuer_substitution.v1",
        "scope": "same compiled host and quantization; Nicolas native MX matrix loop only",
        "stage_receipt_sha256": sha(stage / "mx_stage_probe.json"),
        "source_revision": subprocess.run(
            ["git", "-C", str(args.npu_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip(),
        "source_header_sha256": sha(header),
        "catalog_manifest_sha256": sha(manifest),
        "original_catalog_object_sha256": sha(cat / "mx_catalog.o"),
        "replacement_catalog_object_sha256": sha(catalog_obj),
        "source_sha256": sha(source),
        "issuers": {name: {"precision": entry["precision"],
                             "padded_shape_mnk": entry["padded_shape_mnk"]}
                    for name, entry in sorted(issuers.items())},
        "max_m_tile": args.max_m_tile,
    }
    if elf is not None:
        receipt["elf_sha256"] = sha(elf)
    else:
        receipt["output_kind"] = "catalog_object"
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"issuers": len(issuers),
                      "catalog_sha256": receipt["replacement_catalog_object_sha256"],
                      "elf_sha256": receipt.get("elf_sha256")}))


if __name__ == "__main__":
    main()
