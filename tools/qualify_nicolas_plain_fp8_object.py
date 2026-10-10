"""Compatibility entry point for the pinned Nicolas FP8 128-cubed replay."""

from __future__ import annotations

from pathlib import Path

from mx_gemmini_support.source_gemm import SourceGemm
from tools.qualify_nicolas_plain_matrix_object import (
    CASES, MODEL2MLIR_REVISION, RTL_REVISION, main as _main,
    source_kernel as _source_kernel)


_CASE = CASES["fp8_128x128x128"]
SOURCE_SHA256 = _CASE.source_sha256
HEADER_SHA256 = _CASE.header_sha256


def source_kernel(rtl_root: Path) -> SourceGemm:
    return _source_kernel(rtl_root, _CASE)


def main() -> None:
    _main(default_case=_CASE.key)


if __name__ == "__main__":
    main()
