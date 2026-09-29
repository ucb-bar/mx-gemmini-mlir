"""Reproduce the bounded source-bound RTL bringup vectors as C programs.

The vectors have exact BF16 integer sums, so their expected results do not
depend on the TorchAO fake-quant implementation. This module generates
diagnostic programs; it does not run or certify a simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

from .contraction import plan_mx_contraction_payload
from .diagnostic_program import (
    emit_single_window_baremetal_c,
    emit_two_wave_baremetal_c,
)


_CODES = {
    "mxfp8": (0x38, 0x40, 0x48),
    "mxfp6": (0x0c, 0x10, 0x14),
    "mxfp4": (0x02, 0x04, 0x06),
}
_CASES = {"square32": (32, 32, 32), "square64": (64, 64, 64), "split32x64": (32, 64, 32)}


def _bf16_exact_integer(value: int) -> int:
    bits = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    if bits & 0xffff:
        raise ValueError("bringup sum must be exactly representable as BF16")
    return bits >> 16


def canonical_bringup_source(fmt: str, case: str) -> str:
    """Generate one pinned three-format quadrant diagnostic source file."""
    if fmt not in _CODES or case not in _CASES:
        raise ValueError("selected bringup needs a declared MX format and case")
    m, k, n = _CASES[case]
    one, two, four = _CODES[fmt]
    split = case == "split32x64"
    activation = [
        ([one if row < m // 2 else two] * (k // 2)
         + [two if row < m // 2 else four] * (k // 2))
        if split else [one if row < m // 2 else two] * k
        for row in range(m)
    ]
    weight = [[one if col < n // 2 else two for col in range(n)] for _ in range(k)]
    scales = [[127] * (k // 32) for _ in range(m)]
    lut = None
    if fmt == "mxfp6":
        line = [0, one, two, four] + [0] * 12 if split else [0, one, two] + [0] * 13
        lut = [line[:] for _ in range(m // 2)]
    payload = plan_mx_contraction_payload(
        fmt, activation, weight, scales, scales,
        activation_lut=lut, weight_lut=lut,
        max_blocks_per_wave=1 if split else None,
    )
    expected = [
        [
            _bf16_exact_integer(
                (32 * (3 if row < m // 2 else 6) if split
                 else k * (1 if row < m // 2 else 2))
                * (1 if col < n // 2 else 2)
            )
            for col in range(n)
        ]
        for row in range(m)
    ]
    if split:
        return emit_two_wave_baremetal_c(payload, expected)
    return emit_single_window_baremetal_c(payload, expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=sorted(_CODES), required=True)
    parser.add_argument("--case", choices=sorted(_CASES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = canonical_bringup_source(args.format, args.case).encode()
    with args.output.open("xb") as output:
        output.write(source)
    print(f"{args.format} {args.case} sha256={hashlib.sha256(source).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
