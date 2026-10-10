"""Source-bound GQA QK tile and an explicitly altered MX-safe quantization probe.

The Radiance generator's saved attention golden uses BF16 products and
accumulation for its matrix model. Nicolas's DIM16 MX model instead has an
E4M3 product and a reduced-precision accumulator ramp. The emitted Q/K codes
overflow that path. This module keeps the original bytes and a derived probe
separate; the probe is not a qualification of the committed attention kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re

from .source_fp6 import _array, _bytes
from .source_payload import Resource


SOURCE_REVISION = "ee22e0b87180436cd0fa1583c411a3cf49d7586a"
LATEST_SOURCE_REVISION = "80f84caedbabc663a7433c1da4455b936cca41f3"
SUBMODULE_REVISION = "6fc8ec79bc828b33628ec951bfa1d0f65278a338"
DRIVER_SHA256 = "fd6b7d62322a97b1ded81390400b76eb714a62149b665f2d7ad034401cbc1a0f"
GENERATOR_SHA256 = "52ec97a918dfa795b7117c4e7f97e314a8870aba5d68c9763ad1a584d4e0df7c"
MODEL_SHA256 = "70862cc02ba6ae6f05d7c969512bf7be7c708b5a6804815c91c3041b856e88be"
LOW_LEVEL_MODEL_SHA256 = "6ffa2de07f14333ed2d618c9acbbac2f4edbf2b46394c1a3a600fa407e71e9a3"
HEADER_SHA256 = "900712a90bad7c19d311e9ed5c42800bce0cb1c34c1d2788f19c0b25e2d23e65"
HARDWARE_GENERATOR_SHA256 = "338be03f27619a612d5de0ec9b6d3ae63114b07ce41df3d056b4d39e747f74f3"
HARDWARE_MODEL_SHA256 = "50fb59cecb0c808ae04899f62855bd3dac5fea5947e6f772574a0d924f70e073"
HARDWARE_HEADER_SHA256 = "01e041222fec2508b585707d768620a5126e16647d406a2e85bfb99737fd78b8"
HARDWARE_SHIFT = 6
PRODUCT_PRECISION = [(4, 3)] * 16
ACCUMULATOR_PRECISION = [(4, 4)] * 8 + [(4, 5)] * 2 + [(4, 6)] * 5 + [(8, 7)]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class QkSourceTile:
    driver: Path
    header: Path
    activation: bytes
    weight: bytes
    activation_scales: bytes
    weight_scales: bytes

    def hashes(self) -> dict[str, str]:
        return {name: _sha(getattr(self, name)) for name in (
            "activation", "weight", "activation_scales", "weight_scales")}


def read_first_gqa_qk(source_root: Path, *, hardware_generated: bool = False) -> QkSourceTile:
    """Select head zero, key block zero from the pinned causal GQA driver."""
    directory = source_root / "kernels/flash_attention_mx_gqa"
    driver, header = directory / "kernel.cpp", directory / "include/fa_data.h"
    generator_sha = HARDWARE_GENERATOR_SHA256 if hardware_generated else GENERATOR_SHA256
    model_sha = HARDWARE_MODEL_SHA256 if hardware_generated else MODEL_SHA256
    header_sha = HARDWARE_HEADER_SHA256 if hardware_generated else HEADER_SHA256
    if (_sha(driver.read_bytes()) != DRIVER_SHA256 or
            _sha((directory / "fa_gen_data.py").read_bytes()) != generator_sha or
            _sha((directory / "flash_attention_model.py").read_bytes()) != model_sha or
            _sha((source_root / "lib/mxgemmini/fp8_matmul_model.py").read_bytes()) !=
            LOW_LEVEL_MODEL_SHA256 or _sha(header.read_bytes()) != header_sha):
        raise ValueError("selected GQA source, model, or generated header differs from pinned bytes")
    code = driver.read_text()
    for marker in (
            ".TILE_M = FA_SQ, .TILE_N = FA_BK, .TILE_K = FA_D,",
            "const uint8_t *Qh        = &QK_A_in[h * FA_SQ][0];",
            "&QK_B_blocks[kvb * FA_D][0]",
            "mxgemm_compute_tile<QK>(tid, /*c_spad=*/S_ROW[0]);"):
        if code.count(marker) != 1:
            raise ValueError("pinned GQA QK source stage changed")
    if re.search(r"constexpr GemmConfig QK\s*\{.*?\.DATATYPE = "
                 r"GemmDatatype::FP8, \.QUANT_OUTPUT = false,", code, re.DOTALL) is None:
        raise ValueError("pinned GQA QK format changed")
    data = header.read_text(encoding="ascii")
    if hardware_generated and f"#define FA_MX_HARDWARE_SHIFT {HARDWARE_SHIFT}\n" not in data:
        raise ValueError("generated GQA header lacks the pinned MX headroom shift")
    for macro, expected in (("FA_SQ", 64), ("FA_SK", 256), ("FA_D", 64),
                            ("FA_BK", 64), ("FA_NQ", 8), ("FA_NKV", 2),
                            ("FA_NBLK_USED", 2), ("FA_GK", 2)):
        found = re.search(rf"^#define {macro}\s+(\d+)\b", data, re.MULTILINE)
        if found is None or int(found.group(1)) != expected:
            raise ValueError(f"pinned GQA {macro} changed")
    arrays = {
        "activation": ("QK_A_in", "[FA_NQ*FA_SQ][FA_D]", 8 * 64 * 64, 64 * 64),
        "weight": ("QK_B_blocks", "[FA_NKV*FA_NBLK_USED*FA_D][FA_BK]", 2 * 2 * 64 * 64,
                   64 * 64),
        "activation_scales": ("QK_A_scales_row", "[FA_NQ*FA_GK][FA_SQ]", 8 * 2 * 64,
                              2 * 64),
        "weight_scales": ("QK_B_scales_blocks", "[FA_NKV*FA_NBLK_USED*FA_GK][FA_BK]",
                          2 * 2 * 2 * 64, 2 * 64),
    }
    selected = {}
    for resource, (name, dimensions, count, length) in arrays.items():
        all_values = _array(data, name=name, ctype="uint8_t", dimensions=dimensions,
                            count=count, maximum=255)
        selected[resource] = bytes(all_values[:length])
    return QkSourceTile(driver, header, **selected)


def decode_e4m3(code: int) -> float:
    if (code & 0x7f) == 0x7f:
        raise ValueError("source QK contains an E4M3 NaN code")
    sign = -1.0 if code & 0x80 else 1.0
    exponent, mantissa = (code >> 3) & 15, code & 7
    return sign * ((mantissa / 8) * 2.0**-6 if exponent == 0 else
                   (1 + mantissa / 8) * 2.0**(exponent - 7))


def source_product_overflow_count(tile: QkSourceTile) -> int:
    """Count source Q/K pairs outside Nicolas's finite E4M3 product range."""
    a = [decode_e4m3(code) for code in tile.activation]
    b = [decode_e4m3(code) for code in tile.weight]
    return sum(abs(a[row * 64 + inner] * b[inner * 64 + col]) > 448
               for row in range(64) for col in range(64) for inner in range(64))


def derive_shifted_qk(tile: QkSourceTile, shift: int, *, torch, low_level_model) -> tuple:
    """Return candidate resources and a BF16 oracle using Nicolas's precision ramp."""
    if type(shift) is not int or not 1 <= shift <= 8:
        raise ValueError("candidate E8M0 shift must be an explicit integer in 1..8")
    if (len(tile.activation) != 4096 or len(tile.weight) != 4096 or
            len(tile.activation_scales) != 128 or len(tile.weight_scales) != 128):
        raise ValueError("pinned GQA QK stage has wrong resource sizes")
    for name in ("activation_scales", "weight_scales"):
        if any(code == 0xff or code + shift > 254 for code in getattr(tile, name)):
            raise ValueError("candidate E8M0 shift overflows a source scale")

    def tensor_from_codes(codes: bytes):
        return torch.tensor([decode_e4m3(code) for code in codes],
                            dtype=torch.float32).reshape(64, 64)

    def codes_from_tensor(values) -> bytes:
        codes, bits = low_level_model.tensor_to_custom_fp_codes(
            values, low_level_model.INPUT_SPEC)
        if bits != 8:
            raise ValueError("candidate QK encoder changed FP8 code width")
        return bytes(code for row in codes for code in row)

    activation = codes_from_tensor(tensor_from_codes(tile.activation) / (1 << shift))
    weight = codes_from_tensor(tensor_from_codes(tile.weight) / (1 << shift))
    activation_scales = bytes(code + shift for code in tile.activation_scales)
    weight_scales = bytes(code + shift for code in tile.weight_scales)
    a = tensor_from_codes(activation)
    b = tensor_from_codes(weight)
    a_scales = torch.tensor([math.ldexp(1.0, code - 127)
                             for code in activation_scales], dtype=torch.float32).reshape(2, 64).t()
    b_scales = torch.tensor([math.ldexp(1.0, code - 127)
                             for code in weight_scales], dtype=torch.float32).reshape(2, 64)
    golden = low_level_model.tiled_matmul_hwlike(
        a, b, a_scales, b_scales, verbose=False,
        prod_precision_list=PRODUCT_PRECISION,
        acc_precision_list=ACCUMULATOR_PRECISION)
    if not torch.isfinite(golden).all():
        raise ValueError("candidate QK still overflows Nicolas's product or accumulator")
    codes, bits = low_level_model.tensor_to_custom_fp_codes(golden, "bf16")
    if bits != 16:
        raise ValueError("candidate QK oracle changed BF16 code width")
    golden_bytes = _bytes(tuple(code for row in codes for code in row), 2)
    resources = {
        "activation": Resource(activation, (64, 64), 8, "row_major_codes"),
        "weight": Resource(weight, (64, 64), 8, "row_major_codes"),
        "activation_scales": Resource(activation_scales, (2, 64), 8, "k_group_row_e8m0"),
        "weight_scales": Resource(weight_scales, (2, 64), 8, "k_group_column_e8m0"),
        # BF16 readout does not consume output scales; bind a neutral value explicitly.
        "output_scales": Resource(bytes([127] * 128), (2, 64), 8, "n_group_row_e8m0"),
        "golden_bf16": Resource(golden_bytes, (64, 64), 16, "row_major_bf16"),
    }
    return resources, {
        "schema": "mx_gemmini.attention_qk_shift.v1",
        "stage": "gqa_qk_head0_block0", "e8m0_shift": shift,
        "oracle": "dim16_reduced_precision_product_and_accumulator",
        "source_header_sha256": _sha(tile.header.read_bytes()),
        "source_arrays_sha256": tile.hashes(),
    }


def derive_generated_hardware_qk(tile: QkSourceTile, *, torch, low_level_model,
                                  patch_sha256: str) -> tuple:
    """Bind the *emitted* experimental header bytes to Nicolas's QK numerical model."""
    if _sha(tile.header.read_bytes()) != HARDWARE_HEADER_SHA256:
        raise ValueError("QK header is not the pinned hardware-model generation")
    if (len(tile.activation) != 4096 or len(tile.weight) != 4096 or
            len(tile.activation_scales) != 128 or len(tile.weight_scales) != 128):
        raise ValueError("generated GQA QK stage has wrong resource sizes")
    if len(patch_sha256) != 64 or any(c not in "0123456789abcdef" for c in patch_sha256):
        raise ValueError("hardware-model source patch must have a SHA-256 digest")

    def fp8_tensor(values: bytes):
        return torch.tensor([decode_e4m3(code) for code in values],
                            dtype=torch.float32).reshape(64, 64)

    a = fp8_tensor(tile.activation)
    b = fp8_tensor(tile.weight)
    a_scales = torch.tensor([math.ldexp(1.0, code - 127)
                             for code in tile.activation_scales],
                            dtype=torch.float32).reshape(2, 64).t()
    b_scales = torch.tensor([math.ldexp(1.0, code - 127)
                             for code in tile.weight_scales],
                            dtype=torch.float32).reshape(2, 64)
    golden = low_level_model.tiled_matmul_hwlike(
        a, b, a_scales, b_scales, verbose=False,
        prod_precision_list=PRODUCT_PRECISION,
        acc_precision_list=ACCUMULATOR_PRECISION)
    if not torch.isfinite(golden).all():
        raise ValueError("generated GQA QK stage overflows Nicolas's mesh model")
    codes, bits = low_level_model.tensor_to_custom_fp_codes(golden, "bf16")
    if bits != 16:
        raise ValueError("generated GQA oracle changed BF16 width")
    resources = {
        "activation": Resource(tile.activation, (64, 64), 8, "row_major_codes"),
        "weight": Resource(tile.weight, (64, 64), 8, "row_major_codes"),
        "activation_scales": Resource(tile.activation_scales, (2, 64), 8,
                                      "k_group_row_e8m0"),
        "weight_scales": Resource(tile.weight_scales, (2, 64), 8,
                                  "k_group_column_e8m0"),
        "output_scales": Resource(bytes([127] * 128), (2, 64), 8,
                                  "n_group_row_e8m0"),
        "golden_bf16": Resource(_bytes(tuple(code for row in codes for code in row), 2),
                                (64, 64), 16, "row_major_bf16"),
    }
    policy = {
        "schema": "mx_gemmini.attention_qk_shift.v1",
        "stage": "gqa_qk_head0_block0", "e8m0_shift": HARDWARE_SHIFT,
        "oracle": "dim16_reduced_precision_product_and_accumulator",
        "source_header_sha256": _sha(tile.header.read_bytes()),
        "source_arrays_sha256": tile.hashes(),
        "generated_from_patch_sha256": patch_sha256,
    }
    return resources, policy
