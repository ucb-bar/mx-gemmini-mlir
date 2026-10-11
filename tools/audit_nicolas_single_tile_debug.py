"""Audit the one Nicolas MX debug test without an active numerical oracle.

This is source analysis, not an accelerator execution receipt. It pins the
source bytes and records why a printed PASS cannot qualify compiler output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from mx_gemmini_support.source_fp6 import _array


SOURCE_SHA256 = "99917208653896f8bbb46597b4a067a98eb39c588ea4d93bb491a0808c8a6cdc"
HEADER_SHA256 = "0d67d2adf0913c79754365bb37cf8f0918cbdb12c286316c4479c30c9aadfad0"
RTL_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
DEFAULT_OUTPUT = (Path(__file__).resolve().parents[1] /
                  "docs/evidence/nicolas_single_tile_source_audit_266c593/index.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _active_c(source: str) -> str:
    """Remove comments only; the source digest fixes all remaining C syntax."""
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)


def audit(rtl_root: Path) -> dict:
    bench = rtl_root / "software/gemmini-rocc-tests"
    source_path = bench / "bareMetalC/matmul_single_tile_test.c"
    header_path = bench / "include/matmul_data_mx_fp8.h"
    if (_sha(source_path) != SOURCE_SHA256 or _sha(header_path) != HEADER_SHA256):
        raise ValueError("Nicolas single-tile source or header differs from pinned bytes")
    source = _active_c(source_path.read_text())
    header = header_path.read_text()
    dims = {axis: int(re.search(rf"^#define MATMUL_{axis} (\d+)$", header,
                                re.MULTILINE).group(1)) for axis in ("M", "N", "K", "GK")}
    if (dims != {"M": 32, "N": 32, "K": 32, "GK": 16} or
            not re.search(r"#define DIM 16\b", source)):
        raise ValueError("single-tile debug dimensions changed")
    for loop, axis in (("i", "I"), ("j", "J"), ("k", "K")):
        if not re.search(rf"for \(int {loop} = 1; {loop} < tiles_{axis}; {loop}\+\+\)",
                         source):
            raise ValueError("single-tile source no longer begins MVIN loops at one")
    if (source.count("gemmini_extended_mvin(") != 2 or
            "gemmini_loop_ws_spad(" not in source or
            "0x7f7f7f7f7f7f7f7f" not in source or
            "#define GEMMINI_CTRL 0x40084000" not in source):
        raise ValueError("single-tile load, compute, scale, or MMIO path changed")
    if ("int errors = 0;" not in source or
            re.search(r"\berrors\s*(?:\+\+|\+=|=\s*errors\s*\+)", source) or
            "if (got != exp)" in source or
            "C_hw[i][j] = *(smem_start_addr + (i*8 + j))" not in source):
        raise ValueError("single-tile numerical checker or readout changed")
    a_scales = _array(header, name="A_scales_row", ctype="uint8_t",
                      dimensions="[MATMUL_GK][MATMUL_M]", count=512, maximum=255)
    b_scales = _array(header, name="B_scales_col", ctype="uint8_t",
                      dimensions="[MATMUL_GK][MATMUL_N]", count=512, maximum=255)
    if any(scales[group * 32:(group + 1) * 32] != scales[:32]
           for scales in (a_scales, b_scales) for group in range(16)):
        raise ValueError("single-tile header scale groups are no longer repeated")
    return {
        "schema": "mx_gemmini.nicolas_single_tile_source_audit.v1",
        "status": "debug_source_has_no_active_numerical_oracle",
        "rtl_revision": RTL_REVISION,
        "source": "matmul_single_tile_test.c",
        "source_sha256": SOURCE_SHA256,
        "header": "matmul_data_mx_fp8.h",
        "header_sha256": HEADER_SHA256,
        "shape_mnk": [32, 32, 32],
        "tile_mnk": [16, 16, 16],
        "active_mvin_a_tile_coordinates": [[1, 1]],
        "active_mvin_b_tile_coordinates": [[1, 1]],
        "scale_upload": "constant_0x7f_not_header_scales",
        "header_declared_k_scale_groups": 16,
        "shape_required_k_scale_groups": 1,
        "header_scale_groups_identical": True,
        "readout": "fixed_mmio_scratchpad",
        "active_output_comparisons": 0,
        "printed_pass_is_numerical_evidence": False,
        "compiler_selected_output_qualification": "not_tested",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = json.dumps(audit(args.rtl_root), indent=2, sort_keys=True) + "\n"
    if args.check:
        if args.out.read_text() != result:
            raise ValueError("archived Nicolas single-tile audit differs from pinned source")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result)
    print(args.out)


if __name__ == "__main__":
    main()
