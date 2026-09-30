"""Generate bounded 2029218 scale-load diagnostics; never select the active contract."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .candidate_protocol import scale_load_2d_operands
from .contraction import plan_mx_contraction_payload
from .diagnostic_program import _c_bytes, emit_single_window_baremetal_c

_CONTRACT = Path(__file__).resolve().parent / "contracts/software-spec-2029218-candidate.yaml"
_CODES = {"mxfp8": 0x38, "mxfp6": 0x0C, "mxfp4": 0x02}
_VARIANTS = (
    "pitched", "pitched-gated-alt-half", "loop-managed", "loop-managed-pitched-64",
)


def _loop_managed_source(fmt: str, *, size: int) -> str:
    if size == 64 and fmt != "mxfp8":
        raise ValueError("the 64-square loop diagnostic is selected only for MXFP8")
    code = _CODES[fmt]
    codes = [[code] * size for _ in range(size)]
    scales = [[127] * (size // 32) for _ in range(size)]
    lut = [[0, code] + [0] * 14 for _ in range(size // 2)] if fmt == "mxfp6" else None
    payload = plan_mx_contraction_payload(
        fmt, codes, codes, scales, scales, activation_lut=lut, weight_lut=lut,
    )
    expected = 0x4280 if size == 64 else 0x4200
    source = emit_single_window_baremetal_c(
        payload, [[expected] * size for _ in range(size)],
    )
    for name, data, operand in (
        ("A_scales", payload.waves[0].activation_scale_bytes, "activation"),
        ("B_scales", payload.waves[0].weight_scale_bytes, "weight"),
    ):
        if size == 64:
            if len(data) != 128:
                raise AssertionError("64-square MXFP8 scale layout changed")
            old_array = _c_bytes(name, data)
            if source.count(old_array) != 1:
                raise AssertionError("bounded diagnostic scale array changed")
            source = source.replace(
                old_array, _c_bytes(name, data[:64] + bytes(64) + data[64:]), 1,
            )
        old_load = (
            f"  gemmini_mx_load_scales((uint64_t){name}, sizeof {name}, "
            f"{int(operand == 'weight')});\n"
        )
        if source.count(old_load) != 1:
            raise AssertionError("bounded diagnostic scale command changed")
        source = source.replace(old_load, "", 1)
    marker = "  gemmini_loop_ws_spad("
    if source.count(marker) != 1:
        raise AssertionError("bounded diagnostic loop command changed")
    stride = 128 if size == 64 else 0
    configure = (
        "  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, "
        "(uint64_t)(uintptr_t)A_scales, (uint64_t)(uintptr_t)B_scales, 31);\n"
        f"  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, {stride}, {stride}, 32);\n"
    )
    source = source.replace(marker, configure + marker, 1)
    label = f"generated {fmt.upper()} {size}x{size}x{size}:"
    replacement = (
        f"{fmt.upper()} loop-managed pitched rows:" if size == 64
        else f"{fmt.upper()} loop-managed scales:"
    )
    if source.count(label) != 1:
        raise AssertionError("bounded diagnostic result label changed")
    return source.replace(label, replacement, 1)


def canonical_scale_load_source(fmt: str, variant: str = "pitched") -> str:
    """Render one exact-BF16 check of the selected 2029218 scale protocol.

    Explicit-load variants supply two 16-byte scale rows 32 bytes apart. The
    gated variant selects the second 4 KiB half. Loop variants use funct 31/32;
    the 64-square FP8 case supplies two 64-byte rows 128 bytes apart. The
    32-square FP6/FP4 loop cases are expected to fail on this RTL revision.
    These are diagnostics, not a scheduler or a format-wide qualification.
    """
    if fmt not in _CODES or variant not in _VARIANTS:
        raise ValueError("candidate bringup needs a declared MX format and variant")
    if variant == "loop-managed":
        return _loop_managed_source(fmt, size=32)
    if variant == "loop-managed-pitched-64":
        return _loop_managed_source(fmt, size=64)
    code = _CODES[fmt]
    codes = [[code] * 32 for _ in range(32)]
    scales = [[127] for _ in range(32)]
    lut = [[0, code] + [0] * 14 for _ in range(16)] if fmt == "mxfp6" else None
    payload = plan_mx_contraction_payload(
        fmt, codes, codes, scales, scales,
        activation_lut=lut, weight_lut=lut,
    )
    source = emit_single_window_baremetal_c(payload, [[0x4200] * 32 for _ in range(32)])
    spec = _CONTRACT.read_bytes()
    alt = variant == "pitched-gated-alt-half"
    for name, data, operand in (
        ("A_scales", payload.waves[0].activation_scale_bytes, "activation"),
        ("B_scales", payload.waves[0].weight_scale_bytes, "weight"),
    ):
        if len(data) != 32:
            raise ValueError("candidate diagnostic needs 32 scale bytes per operand")
        old_array = _c_bytes(name, data)
        if source.count(old_array) != 1:
            raise AssertionError("bounded diagnostic scale array changed")
        source = source.replace(old_array, _c_bytes(name, data[:16] + bytes(16) + data[16:]), 1)
        _, rs2 = scale_load_2d_operands(
            spec, source_address=0x1000, row_pitch=32, bytes_per_row=16,
            operand=operand, destination_offset=4096 if alt else 0,
            row_count=2, gated=alt,
        )
        old_load = (
            f"  gemmini_mx_load_scales((uint64_t){name}, sizeof {name}, "
            f"{int(operand == 'weight')});"
        )
        if source.count(old_load) != 1:
            raise AssertionError("bounded diagnostic scale command changed")
        new_load = (
            "  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, "
            f"((uint64_t)(uintptr_t){name}) | (32ULL << 40), "
            f"0x{rs2:016x}ULL, k_MX_LOAD_SCALES);"
        )
        source = source.replace(old_load, new_load, 1)
    if alt:
        old_config = (
            "  gemmini_mxquant_config_mvout((uint64_t)output_scales,\n"
            "      tiles_I, tiles_J, tiles_K, 0, 0, 1);"
        )
        new_config = (
            "  ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, "
            "((uint64_t)(uintptr_t)output_scales) | ((uint64_t)tiles_I << 33) "
            "| ((uint64_t)tiles_J << 42) | ((uint64_t)tiles_K << 51) "
            "| (1ULL << 60) | (1ULL << 61), "
            "(1ULL << 17) | (1ULL << 16) | 1ULL, CONFIG_SCALE_MEM);"
        )
        if source.count(old_config) != 1:
            raise AssertionError("bounded diagnostic scale config changed")
        source = source.replace(old_config, new_config, 1)
        source = source.replace(
            f"generated {fmt.upper()} 32x32x32:",
            f"{fmt.upper()} pitched gated alternate half:",
        )
    return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=sorted(_CODES), required=True)
    parser.add_argument("--variant", choices=_VARIANTS, default="pitched")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = canonical_scale_load_source(args.format, args.variant).encode()
    with args.output.open("xb") as output:
        output.write(source)
    print(f"{args.format} {args.variant} sha256={hashlib.sha256(source).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
