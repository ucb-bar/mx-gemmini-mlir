"""Compile pinned Nicolas plain FP8/FP4/FP6 source data through typed MX MLIR.

The structural matmul comes from a fresh PyTorch -> model2MLIR capture. The
packed operands and BF16 reference come from Nicolas's checked-in C header.
The driver only calls the data-free compiled issuer object and checks output.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from mx_gemmini_support.bind_payload import bind_payload, select_bf16_output_layout
from mx_gemmini_support.bind_profile import bind_handoff
from mx_gemmini_support.chunked_i import bind_i_chunks
from mx_gemmini_support.native_dram import bind_native_dram
from mx_gemmini_support.smem_readback import bind_smem_zero_readout
from mx_gemmini_support.source_gemm import SourceGemm
from mx_gemmini_support.source_fp6 import read_source_fp6_payload
from mx_gemmini_support.source_payload import (NICOLAS_SOURCE_HEADER_ORIGIN,
                                              load_bundle, write_bundle)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _require_gitlink, _run


ROOT = Path(__file__).resolve().parents[1]
RTL_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
MODEL2MLIR_REVISION = "e9ded36eb85abf2d9097ac4dc11457c825853388"
MXQ_REVISION = "b4af5430bac147f4a16126931cc0177367cc3982"


@dataclass(frozen=True)
class Case:
    key: str
    precision: str
    shape: tuple[int, int, int]
    tile: tuple[int, int, int]
    source_name: str
    header_name: str
    source_sha256: str
    header_sha256: str
    profile_name: str
    buffer_abi: tuple[str, ...]
    call_arguments: tuple[str, ...]
    label: str
    quant_output: bool = False
    i_chunks: int = 1
    native_dram: bool = False
    native_dram_chunks: int = 1
    native_dram_scale_mode: str = "preload"
    native_dram_k_tiles: int = 1
    native_dram_store_activation: str = "none"
    smem_zero_readout: bool = False
    dram_mvout_spike_fallback: bool = False


CASES = {
    "fp8_64x64x64": Case(
        "fp8_64x64x64", "FP8", (64, 64, 64), (64, 64, 64),
        "matmul_tiled_fp8_64x64.c", "matmul_fp8_64x64.h",
        "16622c9da9a0b5e8a1875b926ab4a504a994714c4a5f5a2db420d79f89fd7a00",
        "1e6cea94563028a7cd37b486b30ac227b3bcb3eeddd1978430cfa5e8b94d1ecc",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 64 cubed"),
    "fp8_64x64x64_smem_mvout": Case(
        "fp8_64x64x64_smem_mvout", "FP8", (64, 64, 64), (64, 64, 64),
        "matmul_tiled_fp8_64x64_smem_mvout.c", "matmul_fp8_64x64.h",
        "7c3b43b252143a3495ecada4e6d23cfaccae6fa07e5922ebc02cc5737345eb41",
        "1e6cea94563028a7cd37b486b30ac227b3bcb3eeddd1978430cfa5e8b94d1ecc",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 64 cubed zero-base SMEM readout",
        smem_zero_readout=True),
    "fp8_64x64x64_dram_mvout_spike": Case(
        "fp8_64x64x64_dram_mvout_spike", "FP8", (64, 64, 64), (64, 64, 64),
        "matmul_tiled_fp8_64x64_DRAMMvout.c", "matmul_fp8_64x64.h",
        "0c866be0565b1ea0c8f675bdf0468123da53acfd3e130dcb6d9aecc8c95d86a4",
        "1e6cea94563028a7cd37b486b30ac227b3bcb3eeddd1978430cfa5e8b94d1ecc",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 64 cubed DRAM-mvout Spike fallback",
        smem_zero_readout=True, dram_mvout_spike_fallback=True),
    "fp4_64x64x64_dram_mvout_spike": Case(
        "fp4_64x64x64_dram_mvout_spike", "FP4", (64, 64, 64), (64, 64, 64),
        "matmul_tiled_fp4_64x64_DRAMMvout.c", "matmul_fp4_64x64.h",
        "c1729048ccab7836598b55bdb747d6c75489d792683abd300c10368627301155",
        "22851fc6ff791f2748a2bbc501aa98176c4cf26b73c7dcea13dcca2a6202c06b",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in_hw", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP4 64 cubed DRAM-mvout Spike fallback",
        smem_zero_readout=True, dram_mvout_spike_fallback=True),
    "fp8_128x128x256_dram_mvout_spike": Case(
        "fp8_128x128x256_dram_mvout_spike", "FP8", (128, 128, 256),
        (128, 128, 128),
        "matmul_tiled_fp8_128x128x256_DRAMMvout.c",
        "matmul_fp8_128x128x256.h",
        "2d87a0a987c90f7855e69ae46e3429e490a6d4661a63f3d6d4d3043b777dd689",
        "feab7b991acf85b79819c8d7285ccd236330d1a83f79500a2a7144081591209c",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 128x128x256 DRAM-mvout Spike fallback",
        smem_zero_readout=True, dram_mvout_spike_fallback=True),
    "fp8_64x64x64_requant": Case(
        "fp8_64x64x64_requant", "FP8", (64, 64, 64), (64, 64, 64),
        "matmul_tiled_fp8_64x64_requant.c", "matmul_fp8_64x64.h",
        "95c5676526ed5817af6b8fac4d0274eb1b86c985fa1cc2287545b2605a490d0c",
        "1e6cea94563028a7cd37b486b30ac227b3bcb3eeddd1978430cfa5e8b94d1ecc",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_quantized",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 64 cubed requant", True),
    "fp8_96x32x32": Case(
        "fp8_96x32x32", "FP8", (96, 32, 32), (96, 32, 32),
        "matmul_tiled_fp8_96x32x32.c", "matmul_fp8_96x32x32.h",
        "58a6152c1fd60cc62d4125fe6bde759e2c5cc8f34cb0cfd409c3042596168d33",
        "872254ca7a0de8ab0d49ea51a5df357f5c2b7703114356a6eaec363297079cc7",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 96x32x32"),
    "fp8_32x32x32": Case(
        "fp8_32x32x32", "FP8", (32, 32, 32), (32, 32, 32),
        "matmul_tiled_fp8_32x32x32.c", "matmul_fp8_32x32x32.h",
        "716e8c1b6ad1ea415e43f418d698cda18aee98c6616a5b507d1c831b75977bb2",
        "70c32822155c5e329b761952a06f4a36836361b18622d22114d48b58fd708fc6",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 32 cubed"),
    "fp8_32x32x32_alternate_mvin": Case(
        "fp8_32x32x32_alternate_mvin", "FP8", (32, 32, 32), (32, 32, 32),
        "matmul_tiled_fp8.c", "matmul_fp8_32x32x32.h",
        "d9b20beb693cd70dacf02d1b22bbd4ab7d85fd1f2466e38ba9aa41e9bc14e176",
        "70c32822155c5e329b761952a06f4a36836361b18622d22114d48b58fd708fc6",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 32 cubed alternate mvin"),
    "fp8_64x256x64_requant_dim32": Case(
        "fp8_64x256x64_requant_dim32", "FP8", (64, 256, 64), (64, 256, 64),
        "matmul_tiled_fp8_64x256x64_requant_dim32.c",
        "matmul_fp8_64x256x64_dim32.h",
        "0d29269ed7cb089cbfeb9145a3bab517af5f2b249e055d25c31133bd1f438abf",
        "fc989d62d1cbe4c684e90e586df621864bb17e67399de42b31c0b5577d4319f1",
        "MxDim32GemminiRocketConfig",
        ("activation", "activation_scales", "output_quantized",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 64x256x64 requant DIM32", True),
    "fp8_128x128x256_requant_dim32": Case(
        "fp8_128x128x256_requant_dim32", "FP8", (128, 128, 256), (128, 128, 128),
        "matmul_tiled_fp8_128x128x256_requant_dim32.c",
        "matmul_fp8_128x128x256_dim32.h",
        "6870676773a40ad6b249d72a0e833c2d032378dff8ca1d936bae8e301042e382",
        "dfb8818d71ae8489f2f13328ef326181d0a30035583b0ec46a9056877089a84a",
        "MxDim32GemminiRocketConfig",
        ("activation", "activation_scales", "output_quantized",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 128x128x256 requant DIM32", True),
    "fp4_128x128x512": Case(
        "fp4_128x128x512", "FP4", (128, 128, 512), (128, 128, 512),
        "matmul_tiled_fp4_128x128x512.c", "matmul_fp4_128x128x512.h",
        "450779560f0f90618f4e18375419ea6cb19979377d1ea30992c9759695f64ab2",
        "a20619dd213e28238ea29a9bcb7d918e60d8a30cb02a4f609a2fa74377ab44b1",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in_hw", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP4 128x128x512"),
    "fp8_128x128x128": Case(
        "fp8_128x128x128", "FP8", (128, 128, 128), (128, 128, 128),
        "matmul_tiled_fp8_128x128.c", "matmul_fp8_128x128.h",
        "1a0016d2ca9ebcca5840b31486ed6bda2df4cf3756aec0922781db398c98d5ce",
        "16241671c4df2d4f738e77225063caac4895cdcba4d4495941e599d0db185bcd",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 128 cubed"),
    "fp8_128x128x256": Case(
        "fp8_128x128x256", "FP8", (128, 128, 256), (128, 128, 128),
        "matmul_tiled_fp8_128x128x256.c", "matmul_fp8_128x128x256.h",
        "c4fd2385d9223d4e44c8b69e0305dd0ceb5992e986da73ab5d81ab86dfe8da49",
        "feab7b991acf85b79819c8d7285ccd236330d1a83f79500a2a7144081591209c",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 128x128x256"),
    "fp8_96x96x64": Case(
        "fp8_96x96x64", "FP8", (96, 96, 64), (96, 96, 64),
        "matmul_tiled_fp8_96x96x64.c", "matmul_fp8_96x96x64.h",
        "e72eeda20f99f8e1b68ba9a967ca69eab9720970ca48f5a2f912f6e70058d500",
        "2d83916439936348eedd9375dd24bbf5727db5e0facacc02f30c3130bb9abf25",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 96x96x64"),
    "fp8_128x128x128_requant": Case(
        "fp8_128x128x128_requant", "FP8", (128, 128, 128), (128, 128, 128),
        "matmul_tiled_fp8_128x128_requant.c", "matmul_fp8_128x128.h",
        "bc446ca5f0d08a308771b9cdabbf3f9d9feddc8a661b82f5200a2c9d52b118de",
        "16241671c4df2d4f738e77225063caac4895cdcba4d4495941e599d0db185bcd",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_quantized",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 128 requant", True),
    "fp8_64x64x64_requant_dim32": Case(
        "fp8_64x64x64_requant_dim32", "FP8", (64, 64, 64), (64, 64, 64),
        "matmul_tiled_fp8_64x64_requant_dim32.c", "matmul_fp8_64x64_dim32.h",
        "1ddf6983ee813ee6ec286d0d86ec2f45575bf0bd81b40125d5f454ed466baa64",
        "3e2565521f6cd0ade7cc58b873043c7e6ccc59ea41fd1ff6b2dca6aeb4e8a407",
        "MxDim32GemminiRocketConfig",
        ("activation", "activation_scales", "output_quantized",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP8 64 requant DIM32", True),
    "fp4_64x64x64": Case(
        "fp4_64x64x64", "FP4", (64, 64, 64), (64, 64, 64),
        "matmul_tiled_fp4_64x64.c", "matmul_fp4_64x64.h",
        "080d817557d8affdae299b155601b8f3a927970c39542fc6387c1e955492338c",
        "22851fc6ff791f2748a2bbc501aa98176c4cf26b73c7dcea13dcca2a6202c06b",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in_hw", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP4 64x64x64"),
    "fp4_64x64x64_requant": Case(
        "fp4_64x64x64_requant", "FP4", (64, 64, 64), (64, 64, 64),
        "matmul_tiled_fp4_64x64_requant.c", "matmul_fp4_64x64.h",
        "7bc43a4aa97b15834114c9de1c52bbb21a83328daed462c03d18f1350cfcebbc",
        "22851fc6ff791f2748a2bbc501aa98176c4cf26b73c7dcea13dcca2a6202c06b",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_quantized",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in_hw", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP4 64 requant", True),
    "fp4_128x128x512_requant": Case(
        "fp4_128x128x512_requant", "FP4", (128, 128, 512), (128, 128, 512),
        "matmul_tiled_fp4_128x128x512_requant.c", "matmul_fp4_128x128x512.h",
        "07c970dc15e983ff91aa68a1fe578a37902b7e201e826b17e9a36db007f7c026",
        "a20619dd213e28238ea29a9bcb7d918e60d8a30cb02a4f609a2fa74377ab44b1",
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_quantized",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in_hw", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP4 128x128x512 requant", True),
    "fp4_128x128x128_requant_dim32": Case(
        "fp4_128x128x128_requant_dim32", "FP4", (128, 128, 128), (128, 128, 128),
        "matmul_tiled_fp4_128x128_requant_dim32.c", "matmul_fp4_128x128_dim32.h",
        "2688bb9ba9cee539d83a80a9e748d01b255ff92c23466b7903a4a7f1845291b2",
        "b28981f5e0cd9e74a33a9663f753b68e76e71e76f630c82afcd2364c8a4a8745",
        "MxDim32GemminiRocketConfig",
        ("activation", "activation_scales", "output_quantized",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in_hw", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), "FP4 128 requant DIM32", True),
    "fp6_128x128x512": Case(
        "fp6_128x128x512", "FP6", (128, 128, 512), (128, 128, 512),
        "matmul_tiled_fp6_128x128x512.c", "matmul_fp6_128x128x512.h",
        "dec4c96493b1a7eb7b534d6625db135e60f0474807f8688b3b5b79fbfd4203d1",
        "7e499e594e324e48c9f0b17b70a2fe7ec57f7d156af118f9dfa9d59b7e4507fd",
        "MxE3M2OnlyGemminiRocketConfig",
        ("activation", "activation_lut", "activation_scales", "output_bf16",
         "output_lut", "scratch_output_scales", "weight", "weight_lut",
         "weight_scales"),
        ("A_in_hw", "A_lut", "A_scales_row", "C_hw", "C_lut",
         "scratch_output_scales", "B_in", "B_lut", "B_scales_col"),
        "FP6 128x128x512"),
    "fp6_128x128x128": Case(
        "fp6_128x128x128", "FP6", (128, 128, 128), (128, 128, 128),
        "matmul_tiled_fp6_128x128.c", "matmul_data_mx_lut_hw.h",
        "2a57f4f475ae1c63049bc3c5aed2b77f1f80a2a2fa81d9c569c0683828c00665",
        "127e962daebcbd890b4d76c9f884453e88c37aa5e1d9034954e7bb1cdfd42751",
        "MxE3M2OnlyGemminiRocketConfig",
        ("activation", "activation_lut", "activation_scales", "output_bf16",
         "output_lut", "scratch_output_scales", "weight", "weight_lut",
         "weight_scales"),
        ("A_in_hw", "A_lut", "A_scales_row", "C_hw", "C_lut",
         "scratch_output_scales", "B_in", "B_lut", "B_scales_col"),
        "FP6 128x128x128"),
    "fp6_128x128x512_requant": Case(
        "fp6_128x128x512_requant", "FP6", (128, 128, 512), (128, 128, 512),
        "matmul_tiled_fp6_128x128x512_requant.c", "matmul_fp6_128x128x512.h",
        "5f8bbd631ae19684a63d945dd2272930aad4553bc0f500c0bea58cb75c51235b",
        "7e499e594e324e48c9f0b17b70a2fe7ec57f7d156af118f9dfa9d59b7e4507fd",
        "MxE3M2OnlyGemminiRocketConfig",
        ("activation", "activation_lut", "activation_scales", "output_lut",
         "output_quantized", "scratch_output_scales", "weight", "weight_lut",
         "weight_scales"),
        ("A_in_hw", "A_lut", "A_scales_row", "C_lut", "C_hw",
         "scratch_output_scales", "B_in", "B_lut", "B_scales_col"),
        "FP6 128x128x512 requant", True),
}


def _register_standard_source(key: str, precision: str,
                              shape: tuple[int, int, int], source_name: str,
                              header_name: str, source_sha256: str,
                              header_sha256: str, *, quant_output: bool = False,
                              profile_name: str = "MxGemminiRocketConfig") -> None:
    """Register a pinned matrix source using the ordinary MX pointer ABI."""
    if key in CASES or precision not in {"FP8", "FP4"}:
        raise ValueError(f"unsupported or duplicate Nicolas source case: {key}")
    output = "output_quantized" if quant_output else "output_bf16"
    activation = "A_in_hw" if precision == "FP4" else "A_in"
    CASES[key] = Case(
        key, precision, shape, shape, source_name, header_name,
        source_sha256, header_sha256, profile_name,
        ("activation", "activation_scales", output, "scratch_output_scales",
         "weight", "weight_scales"),
        (activation, "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"),
        f"{precision} {shape[0]}x{shape[1]}x{shape[2]}" +
        (" requant" if quant_output else ""), quant_output)


_register_standard_source(
    "fp8_64x96x64", "FP8", (64, 96, 64),
    "matmul_tiled_fp8_64x96x64.c", "matmul_fp8_64x96x64.h",
    "026aa3f9510306aa43b7513a1a27ee610091eacc96af2dd0ce255d9e372b219d",
    "5b71b66fdb15fd776958e8d995d15f9aa0ddbbd39892c3cdb27d1ca144c15dd2")
_register_standard_source(
    "fp8_64x96x64_requant", "FP8", (64, 96, 64),
    "matmul_tiled_fp8_64x96x64_requant.c", "matmul_fp8_64x96x64.h",
    "5b0f17b108bf81e5146f513f0ca325490a6a15eb55bcbf25178881577a0cfb38",
    "5b71b66fdb15fd776958e8d995d15f9aa0ddbbd39892c3cdb27d1ca144c15dd2",
    quant_output=True)
_register_standard_source(
    "fp8_96x32x32_requant", "FP8", (96, 32, 32),
    "matmul_tiled_fp8_96x32x32_requant.c", "matmul_fp8_96x32x32.h",
    "747acfe2358bab83913ffd8c6a4db23dbcb50bde08ca9ad496fca054eeb519fe",
    "872254ca7a0de8ab0d49ea51a5df357f5c2b7703114356a6eaec363297079cc7",
    quant_output=True)
_register_standard_source(
    "fp8_96x96x64_requant", "FP8", (96, 96, 64),
    "matmul_tiled_fp8_96x96x64_requant.c", "matmul_fp8_96x96x64.h",
    "00b13d84f832e0217adf00e367788a179c1572755e6846ce51346c9687aadca5",
    "2d83916439936348eedd9375dd24bbf5727db5e0facacc02f30c3130bb9abf25",
    quant_output=True)
_register_standard_source(
    "fp8_32x32x32_requant", "FP8", (32, 32, 32),
    "matmul_tiled_fp8_32x32x32_requant.c", "matmul_fp8_32x32x32.h",
    "58ebb708907ccd61989b8a22a37578ece4005b33e1af69f4564f3af3dbc5829c",
    "70c32822155c5e329b761952a06f4a36836361b18622d22114d48b58fd708fc6",
    quant_output=True)
_register_standard_source(
    "fp4_128x128x128", "FP4", (128, 128, 128),
    "matmul_tiled_fp4_128x128.c", "matmul_fp4_128x128.h",
    "cbd724db7a161305fa4763470346562efdc3665ab3d5d04b604df2a82c1ca8bd",
    "f3ab8dbc5d492c661756d163b42e3be3914f6819b52549d08f0ee453706e9607")
_register_standard_source(
    "fp4_128x128x128_requant", "FP4", (128, 128, 128),
    "matmul_tiled_fp4_128x128_requant.c", "matmul_fp4_128x128.h",
    "5290df310cf1f4af2ef051792726bf451a454cc1ae4a3b124686eecf2a42cd46",
    "f3ab8dbc5d492c661756d163b42e3be3914f6819b52549d08f0ee453706e9607",
    quant_output=True)
_register_standard_source(
    "fp4_64x64x64_requant_dim32", "FP4", (64, 64, 64),
    "matmul_tiled_fp4_64x64_requant_dim32.c", "matmul_fp4_64x64_dim32.h",
    "724bda79b2f594520dff04808689703a593abc7ea656e16022a756c2fe018cfa",
    "5a1f3fa7806fff120dbd7c283ff136a952798a07da122bf2d78ef5a629b72bc5",
    quant_output=True, profile_name="MxDim32GemminiRocketConfig")
_register_standard_source(
    "fp4_64x64x64_nonrequant_dim32", "FP4", (64, 64, 64),
    "matmul_tiled_fp4_64x64_nonrequant_dim32.c", "matmul_fp4_64x64_dim32.h",
    "35bc2bf18ae3314835a1e34379788e4822cacb927f9680c5647a9105bf45fa74",
    "5a1f3fa7806fff120dbd7c283ff136a952798a07da122bf2d78ef5a629b72bc5",
    profile_name="MxDim32GemminiRocketConfig")
_register_standard_source(
    "fp4_128x128x128_nonrequant_dim32", "FP4", (128, 128, 128),
    "matmul_tiled_fp4_128x128_nonrequant_dim32.c", "matmul_fp4_128x128_dim32.h",
    "8e78585cdd9a03f1a7f007ed13bb11bc210c0a500e87a4ed79bd9b7dfcfe7834",
    "b28981f5e0cd9e74a33a9663f753b68e76e71e76f630c82afcd2364c8a4a8745",
    profile_name="MxDim32GemminiRocketConfig")
_register_standard_source(
    "fp4_128x128x64_requant_dim32", "FP4", (128, 128, 64),
    "matmul_tiled_fp4_128x128x64_requant_dim32.c", "matmul_fp4_128x128x64_dim32.h",
    "d50779e8836a3d7b7296610ab362fc7d730e3dbd07e66e50b3da76b95ddb3572",
    "9f55f60d82647988eacc91d82e3399199acf11d5cae4e10163620166be0eef97",
    quant_output=True, profile_name="MxDim32GemminiRocketConfig")
_register_standard_source(
    "fp8_64x64x64_single_dim8", "FP8", (64, 64, 64),
    "matmul_tiled_fp8_64x64_single_dim8.c", "matmul_fp8_64x64_dim8.h",
    "f270fff49b70b468c88690f6baf068a25936689039802328e3c46c2f6c697179",
    "be9aad16d3f4b2ba08b4f75944958a3234631c86d6c8c65aa143d7ad622b5593",
    profile_name="MxDim8AllGemminiRocketConfig")
_register_standard_source(
    "fp8_128x128x128_single_dim8", "FP8", (128, 128, 128),
    "matmul_tiled_fp8_128x128_single_dim8.c", "matmul_fp8_128x128_dim8.h",
    "32dc131ab86a93cf4c60628f5b1e1bf297c37428de1036d65ddbf1057a13022e",
    "5ecc24f4ceb8e2dc9533e96c0ca842ed43d677fc6a359b1d529c23767fc5e624",
    profile_name="MxDim8AllGemminiRocketConfig")
_register_standard_source(
    "fp8_64x64x64_requant_dim8", "FP8", (64, 64, 64),
    "matmul_tiled_fp8_64x64_requant_dim8.c", "matmul_fp8_64x64_dim8.h",
    "239394c2252865c6480f2851da8aadef9872fb842b473fab0df525373af271e3",
    "be9aad16d3f4b2ba08b4f75944958a3234631c86d6c8c65aa143d7ad622b5593",
    quant_output=True, profile_name="MxDim8AllGemminiRocketConfig")
_register_standard_source(
    "fp8_128x128x128_requant_dim8", "FP8", (128, 128, 128),
    "matmul_tiled_fp8_128x128_requant_dim8.c", "matmul_fp8_128x128_dim8.h",
    "56291b1d6f38addff5751755d7a01c8beb650c384f663a1179d88eeda3420567",
    "5ecc24f4ceb8e2dc9533e96c0ca842ed43d677fc6a359b1d529c23767fc5e624",
    quant_output=True, profile_name="MxDim8AllGemminiRocketConfig")
_register_standard_source(
    "fp4_64x64x64_requant_dim8", "FP4", (64, 64, 64),
    "matmul_tiled_fp4_64x64_requant_dim8.c", "matmul_fp4_64x64_dim8.h",
    "f410f6d70d03ca40dff626fff0b26987e69a88339d07dd77ef4f67cf22712095",
    "252364c3b867cabb0dc555174cb90278bab2107d18ddcfbf00e46ca36f275e14",
    quant_output=True, profile_name="MxDim8AllGemminiRocketConfig")

_CHUNK_HEADER_SHA256 = "16241671c4df2d4f738e77225063caac4895cdcba4d4495941e599d0db185bcd"
for _chunks, _name, _sha256 in (
        (2, "matmul_tiled_fp8_128x128_chunked.c",
         "c2cd8affbbbd079a3553bcfc8bf46246114c8a2ff62e38f07994447cb512661e"),
        (4, "matmul_tiled_fp8_128x128_chunked4.c",
         "12bc6da5f5a54f87a4df95664a8259f6b0610675a92d0259121cbf13fe891275")):
    _key = f"fp8_128x128x128_chunked{_chunks}"
    CASES[_key] = Case(
        _key, "FP8", (128, 128, 128), (128, 128, 128), _name,
        "matmul_fp8_128x128.h", _sha256, _CHUNK_HEADER_SHA256,
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), f"FP8 128 cubed chunked{_chunks}",
        i_chunks=_chunks)

_chunk2d_key = "fp8_128x128x128_chunked_2d"
CASES[_chunk2d_key] = Case(
    _chunk2d_key, "FP8", (128, 128, 128), (128, 128, 128),
    "matmul_tiled_fp8_128x128_chunked_2d.c", "matmul_fp8_128x128.h",
    "1bdf2d901039ec068fc597ee8b8a57c21f1f13bfb0a38de6b107f640093aebb9",
    _CHUNK_HEADER_SHA256, "MxGemminiRocketConfig",
    ("activation", "activation_scales", "output_bf16",
     "scratch_output_scales", "weight", "weight_scales"),
    ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
     "B_in", "B_scales_col"), "FP8 128 cubed chunked 2D scales", i_chunks=2)

_native_key = "fp8_128x128x128_native_dram"
CASES[_native_key] = Case(
    _native_key, "FP8", (128, 128, 128), (128, 128, 128),
    "matmul_tiled_fp8_128x128_dramloop.c", "matmul_fp8_128x128.h",
    "53b5f60844725aa9e289a1c3833002fe777c2a3b41ea64a31b05c2237eb32277",
    _CHUNK_HEADER_SHA256, "MxGemminiRocketConfig",
    ("activation", "activation_scales", "output_bf16",
     "scratch_output_scales", "weight", "weight_scales"),
    ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
     "B_in", "B_scales_col"), "FP8 128 cubed native DRAM loop",
    native_dram=True)

for _chunks, _name, _sha256 in (
        (2, "matmul_tiled_fp8_128x128_dramloop_nc.c",
         "eb80572c6e1a9ad1a700d4a98d28e106dda214bca377903cea30e37ebd3f7879"),
        (4, "matmul_tiled_fp8_128x128_dramloop_nc4.c",
         "88cab5f1ff321e1bb34687654d226dc134f0bd74c13d68ae1e88b9e79f20e99a")):
    _key = f"fp8_128x128x128_native_dram_nc{_chunks}"
    CASES[_key] = Case(
        _key, "FP8", (128, 128, 128), (128, 128, 128), _name,
        "matmul_fp8_128x128.h", _sha256, _CHUNK_HEADER_SHA256,
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"), f"FP8 128 cubed native DRAM NC{_chunks}",
        native_dram=True, native_dram_chunks=_chunks)

for _chunks, _name, _sha256 in (
        (2, "matmul_tiled_fp8_128x128_dramloop_ls.c",
         "b7d8b95e0e8a3cc16730125f3c2bf5477a042754b065cc2e4725973a57fe4e34"),
        (4, "matmul_tiled_fp8_128x128_dramloop_ls4.c",
         "946515729cd3d962253b238af23522b849bf6baef721238ec4f2dacfc784a706")):
    _key = f"fp8_128x128x128_native_dram_ls{_chunks}"
    CASES[_key] = Case(
        _key, "FP8", (128, 128, 128), (128, 128, 128), _name,
        "matmul_fp8_128x128.h", _sha256, _CHUNK_HEADER_SHA256,
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "B_in", "B_scales_col"),
        f"FP8 128 cubed native DRAM LS{_chunks}",
        native_dram=True, native_dram_chunks=_chunks,
        native_dram_scale_mode="loop")

_kt_key = "fp8_128x128x128_native_dram_kt2"
CASES[_kt_key] = Case(
    _kt_key, "FP8", (128, 128, 128), (128, 128, 128),
    "matmul_tiled_fp8_128x128_dramloop_kt.c", "matmul_fp8_128x128.h",
    "586910dcd6445a3ba0fc0e9db485d85de3ea265b8efe7464931ba3ff005dfa08",
    _CHUNK_HEADER_SHA256, "MxGemminiRocketConfig",
    ("activation", "activation_scales", "output_bf16",
     "weight", "weight_scales"),
    ("A_in", "A_scales_row", "C_hw", "B_in", "B_scales_col"),
    "FP8 128 cubed native DRAM K-tiled", native_dram=True,
    native_dram_chunks=2, native_dram_scale_mode="loop",
    native_dram_k_tiles=2)

for _suffix, _sha256, _mode in (
        ("nc_2d", "4d5fffa5153779041d4f271cad33c6971af028b03031eb21d7d547d06aba4854",
         "direct_2d"),
        ("nc_wait", "c69f99fcc6bb791f705ce94c7ceeef8a10efc80a4fcdb47e1ba7e578ce5ea974",
         "wait")):
    _key = f"fp8_128x128x128_native_dram_{_suffix}"
    CASES[_key] = Case(
        _key, "FP8", (128, 128, 128), (128, 128, 128),
        f"matmul_tiled_fp8_128x128_dramloop_{_suffix}.c",
        "matmul_fp8_128x128.h", _sha256, _CHUNK_HEADER_SHA256,
        "MxGemminiRocketConfig",
        ("activation", "activation_scales", "output_bf16",
         "scratch_output_scales", "weight", "weight_scales"),
        ("A_in", "A_scales_row", "C_hw", "scratch_output_scales",
         "B_in", "B_scales_col"),
        f"FP8 128 cubed native DRAM {_suffix}", native_dram=True,
        native_dram_chunks=2, native_dram_scale_mode=_mode)

_relu_key = "fp8_128x128x128_native_dram_relu"
CASES[_relu_key] = Case(
    _relu_key, "FP8", (128, 128, 128), (128, 128, 128),
    "matmul_tiled_fp8_128x128_dramloop_relu.c", "matmul_fp8_128x128.h",
    "cf9c3431d3fab5dd71e4956fb785cd73b1c49651a3d1c9ff44046f29b22b17cf",
    _CHUNK_HEADER_SHA256, "MxGemminiRocketConfig",
    ("activation", "activation_scales", "output_bf16",
     "weight", "weight_scales"),
    ("A_in", "A_scales_row", "C_hw", "B_in", "B_scales_col"),
    "FP8 128 cubed native DRAM ReLU", native_dram=True,
    native_dram_chunks=2, native_dram_scale_mode="loop",
    native_dram_store_activation="relu")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


def _audit_alternate_fp8_mvin(object_dir: Path, profile: dict,
                              source_sha256: str) -> dict:
    """Prove the two 32³ source loop nests flatten to the emitted transfers."""
    physical = json.loads((object_dir / "physical_program.json").read_text())
    dim = profile["geometry"]["mesh_columns"]
    if dim != 16 or physical["shape_mnk"] != [32, 32, 32]:
        raise ValueError("alternate FP8 transfer audit needs Nicolas's DIM16 32³ case")
    m = n = k = 32
    ti, tj, tki = m // dim, n // dim, k // dim
    b_base = profile["resources"]["scratchpad_bytes"] // 16 - tki * tj * dim
    # Literal source formulas, pinned by source_kernel's source hash and anchors.
    expected = {
        "move_activation": [(i * dim * m + ki * dim,
                             (i * tki + ki) * dim)
                            for i in range(ti) for ki in range(tki)],
        "move_weight": [(j * dim * m + ki * dim,
                         b_base + (j * tki + ki) * dim)
                        for j in range(tj) for ki in range(tki)],
    }
    observed = {}
    for phase, pairs in expected.items():
        steps = [step["command"] for step in physical["steps"]
                 if step["phase"] == phase and step["command"].get("funct") == 2]
        observed[phase] = [(step["rs1"]["byte_offset"],
                            step["rs2"]["immediate"] & 0xffffffff)
                           for step in steps]
        if observed[phase] != pairs:
            raise ValueError(f"compiler {phase} differs from Nicolas's alternate loop")
    return {"schema": "mx_gemmini.nicolas_alternate_fp8_mvin_audit.v1",
            "source_driver_sha256": source_sha256,
            "profile_sha256": profile_sha256(profile),
            "physical_program_sha256": _sha(object_dir / "physical_program.json"),
            "source_transfer_pairs": expected,
            "compiler_transfer_pairs": observed,
            "object_sha256": _sha(object_dir / "mx_issue.o")}


def _audit_smem_zero_readout(object_dir: Path, profile: dict,
                              case: Case) -> dict:
    """Match the source's scratchpad-zero C store and flat BF16 readback."""
    physical = json.loads((object_dir / "physical_program.json").read_text())
    plan = physical["plan"]
    steps = physical["steps"]
    compute = [step["command"] for step in steps
               if step["phase"] == "compute" and step["command"].get("funct") == 8]
    reads = [step["command"] for step in steps
             if step["phase"] == "readout" and step["command"].get("funct") == 3]
    stores = [step["command"] for step in steps
              if step["phase"] == "configure" and step["command"].get("funct") == 0
              and step["command"]["rs1"].get("immediate") == 2]
    m, n, _ = case.shape
    c_rows = m * n * 2 // 16
    if (not case.smem_zero_readout or profile["geometry"]["mesh_columns"] != 16 or
            physical["shape_mnk"] != list(case.shape) or
            plan.get("c_spad_dest") != 0 or
            plan.get("readout_transport") != "spad_mvout_zero_base" or
            plan.get("readout_source_sha256") != case.source_sha256 or
            len(compute) != len(plan["waves"]) or len(stores) != 1 or
            len(reads) != c_rows // 16 or
            [command["rs2"].get("immediate") & 0xffffffff
             for command in compute] !=
            [0x2b8] * (len(plan["waves"]) - 1) + [0x238] or
            stores[0]["rs2"].get("immediate") != 16 or
            [command["rs2"].get("immediate") & 0xffffffff for command in reads] !=
            list(range(0, c_rows, 16)) or
            [command["rs1"].get("byte_offset") for command in reads] !=
            [row * 16 for row in range(0, c_rows, 16)]):
        raise ValueError("zero-base scratchpad commands differ from source geometry")
    return {"schema": "mx_gemmini.nicolas_fp8_smem_zero_readout_audit.v1",
            "source_driver_sha256": case.source_sha256,
            "profile_sha256": profile_sha256(profile),
            "physical_program_sha256": _sha(object_dir / "physical_program.json"),
            "compute_c_scratchpad_row": 0,
            "readout_rows": list(range(0, c_rows, 16)),
            "readout_transport": "spad_mvout_zero_base",
            "readout_count": len(reads)}


def _audit_chunked_fp8(object_dir: Path, profile: dict, case: Case) -> dict:
    """Check the source I-chunk loop, scale placement, and bank-toggle commands."""
    if case.i_chunks not in {2, 4} or profile["geometry"]["mesh_columns"] != 16:
        raise ValueError("chunked FP8 audit needs the pinned DIM16 source")
    physical = json.loads((object_dir / "physical_program.json").read_text())
    if (physical["shape_mnk"] != [128, 128, 128] or
            physical["plan"]["i_chunks"] != case.i_chunks):
        raise ValueError("compiler did not select the source I-chunk plan")
    steps = physical["steps"]
    chunks = case.i_chunks
    chunk_m, chunk_i = 128 // chunks, 8 // chunks
    scales = [(step["wave"], step["command"]["rs1"].get("buffer"),
               step["command"]["rs1"].get("byte_offset"),
               step["command"]["rs1"].get("or_bits"),
               step["command"]["rs2"].get("immediate"))
              for step in steps if step["phase"] == "upload_chunk_scales"
              and step["command"].get("funct") == 27]
    expected_scales = []
    for c in range(chunks):
        for buffer, offset, count, dest, selector in (
                ("activation_scales", c * chunk_m, chunk_m,
                 c * 4 * chunk_m, 0),
                ("weight_scales", 0, 128, c * 4 * 128, 1)):
            expected_scales.append((c, buffer, offset, 128 << 40,
                                    (4 << 46) | (dest << 33) |
                                    (selector << 32) | count))
    commands = [(step["wave"], step["command"]["funct"],
                 step["command"]["rs1"]["immediate"],
                 step["command"]["rs2"]["immediate"])
                for step in steps if step["phase"] == "compute_chunk"
                and step["command"].get("funct") is not None]
    expected_commands = []
    for c in range(chunks):
        expected_commands.extend(((c, 9, 0, (8 << 32) | (8 << 16) | chunk_i),
                                  (c, 24, c * chunk_i * 8 * 16, 16384),
                                  (c, 8, 0,
                                   ((1024 + c * chunk_m * 128 * 2 // 16) << 32) |
                                   0x200 | 0x138)))
    selection = [step for step in steps if step["phase"] == "select_chunk_scales"]
    if (scales != expected_scales or commands != expected_commands or
            len(selection) != 1 or selection[0]["command"].get("funct") != 26 or
            selection[0]["command"]["rs1"].get("or_bits") !=
            ((8 * chunks << 51) | (8 << 42) | (chunk_i << 33)) or
            sum(step["command"] == {} for step in steps
                if step["phase"] == "compute_chunk") != 1):
        raise ValueError("compiler I-chunk commands differ from Nicolas's source")
    return {"schema": "mx_gemmini.nicolas_fp8_i_chunks_audit.v1",
            "source_driver_sha256": case.source_sha256,
            "profile_sha256": profile_sha256(profile),
            "physical_program_sha256": _sha(object_dir / "physical_program.json"),
            "source_chunk_count": chunks,
            "compiler_chunk_count": len(commands) // 3,
            "scale_uploads": len(scales),
            "accumulator_bank_toggle_bits": [cmd[3] & 0x100 for cmd in commands
                                             if cmd[1] == 8]}


def _audit_native_dram_fp8(object_dir: Path, profile: dict, case: Case) -> dict:
    """Match the native loop's source pointer, stride, and scale semantics."""
    if case.native_dram_chunks > 1:
        return _audit_native_dram_nc_fp8(object_dir, profile, case)
    physical = json.loads((object_dir / "physical_program.json").read_text())
    steps = physical["steps"]
    if (not case.native_dram or physical["shape_mnk"] != [128, 128, 128] or
            profile["geometry"]["mesh_columns"] != 16 or
            physical["plan"].get("execution_transport") != "native_dram_loop" or
            any(step["phase"] in {"move_activation", "move_weight", "compute",
                                   "readout"} for step in steps)):
        raise ValueError("compiler did not select Nicolas's native DRAM loop")
    native = [step["command"] for step in steps
              if step["phase"] == "native_dram_loop"]
    if [command.get("funct") for command in native] != [9, 10, 11, 12, 13, 8, None]:
        raise ValueError("compiler native DRAM loop command order differs")
    expected = (
        (0, (8 << 32) | (8 << 16) | 8),
        ("activation", "weight"),
        (0, "output_bf16"),
        (128, 128),
        (0, 128),
        ((1 << 18) | (1 << 16), 0),
    )
    actual = tuple(tuple(command[register]["buffer"] if
                         command[register]["buffer"] is not None else
                         command[register]["immediate"]
                         for register in ("rs1", "rs2")) for command in native[:-1])
    if actual != expected:
        raise ValueError("compiler native DRAM loop operands differ from source")
    scale_commands = [step["command"] for step in steps
                      if step["phase"] == "upload_scales" and
                      step["command"].get("funct") == 27]
    if (len(scale_commands) != 2 or
            [(command["rs1"]["buffer"], command["rs1"]["byte_offset"],
              command["rs1"]["or_bits"], command["rs2"]["immediate"])
             for command in scale_commands] != [
                ("activation_scales", 0, 128 << 40, (4 << 46) | 128),
                ("weight_scales", 0, 128 << 40,
                 (4 << 46) | (1 << 32) | 128)]):
        raise ValueError("compiler scales differ from contiguous source arrays")
    selectors = [step["command"] for step in steps
                 if step["phase"] == "select_scales"]
    if (len(selectors) != 1 or selectors[0].get("funct") != 26 or
            selectors[0]["rs1"]["buffer"] != "scratch_output_scales" or
            selectors[0]["rs1"]["or_bits"] !=
            ((8 << 51) | (8 << 42) | (8 << 33))):
        raise ValueError("compiler native loop scale selector differs")
    return {"schema": "mx_gemmini.nicolas_fp8_native_dram_audit.v1",
            "source_driver_sha256": case.source_sha256,
            "profile_sha256": profile_sha256(profile),
            "physical_program_sha256": _sha(object_dir / "physical_program.json"),
            "native_command_functs": [command["funct"] for command in native[:-1]],
            "native_pointer_buffers": ["activation", "weight", "output_bf16"],
            "loop_bounds_ijk": [8, 8, 8],
            "scale_bytes_per_side": 512,
            "explicit_operand_dma_commands": 0,
            "explicit_output_dma_commands": 0}


def _audit_native_dram_nc_fp8(object_dir: Path, profile: dict,
                              case: Case) -> dict:
    """Audit Nicolas's N-column loop sequence and alternating B bank."""
    if case.native_dram_k_tiles == 2:
        return _audit_native_dram_kt_fp8(object_dir, profile, case)
    chunks = case.native_dram_chunks
    loop_scales = case.native_dram_scale_mode == "loop"
    scale_wait = case.native_dram_scale_mode == "wait"
    store_relu = case.native_dram_store_activation == "relu"
    if chunks not in {2, 4} or profile["geometry"]["mesh_columns"] != 16:
        raise ValueError("native column-loop audit needs the pinned DIM16 case")
    physical = json.loads((object_dir / "physical_program.json").read_text())
    steps = physical["steps"]
    if (physical["shape_mnk"] != [128, 128, 128] or
            physical["plan"].get("execution_transport") != "native_dram_loop" or
            physical["plan"].get("native_dram_n_chunks") != chunks or
            physical["plan"].get("native_dram_scale_mode") !=
            ("loop_managed" if loop_scales else None) or
            physical["plan"].get("native_dram_scale_wait") !=
            (True if scale_wait else None) or
            physical["plan"].get("native_dram_store_activation") !=
            ("relu" if store_relu else None) or
            any(step["phase"] in {"move_activation", "move_weight", "compute",
                                   "readout"} for step in steps)):
        raise ValueError("compiler did not select native column-loop transport")
    native = [step["command"] for step in steps
              if step["phase"] == "native_dram_loop"]
    if [command.get("funct") for command in native] != \
            [9, 10, 11, 12, 13, 8] * chunks + [None]:
        raise ValueError("native column-loop issue order differs from source")
    nc = 128 // chunks
    for c in range(chunks):
        command = native[c * 6:(c + 1) * 6]
        expected = (
            (0, (8 << 32) | ((8 // chunks) << 16) | 8),
            ("activation" if c == 0 else 0, "weight"),
            (0, "output_bf16"),
            (128, 128),
            (0, 128),
            ((1 << 18) | ((1 + (c & 1)) << 16), 0),
        )
        actual = tuple(tuple(item[register]["buffer"] if
                             item[register]["buffer"] is not None else
                             item[register]["immediate"]
                             for register in ("rs1", "rs2")) for item in command)
        if (actual != expected or command[1]["rs2"]["byte_offset"] != c * nc or
                command[2]["rs2"]["byte_offset"] != c * nc * 2):
            raise ValueError("native column-loop pointer or bank differs from source")
    scales = [step for step in steps if step["phase"] == "upload_chunk_scales"
              and step["command"].get("funct") == 27]
    selectors = [step["command"] for step in steps
                 if step["phase"] == "select_scales"]
    stores = [step["command"] for step in steps if step["phase"] == "configure"
              and step["command"].get("funct") == 0 and
              step["command"]["rs1"].get("immediate") in {2, 6}]
    if (len(stores) != 1 or stores[0]["rs1"]["immediate"] !=
            (6 if store_relu else 2) or
            stores[0]["rs2"]["immediate"] != 256 or
            physical.get("golden_derivation") !=
            ("bf16_store_relu_sign_clear" if store_relu else None) or
            physical.get("source_golden_preserving", True) != (not store_relu)):
        raise ValueError("native column-loop store activation differs from source")
    if loop_scales:
        per_loop = [step for step in steps if step["phase"] == "loop_scales"]
        if (scales or selectors or
                [step["command"].get("funct") for step in per_loop] !=
                [31, 32] * chunks):
            raise ValueError("native loop-managed scale commands differ")
        for c in range(chunks):
            pointers, strides = [step["command"] for step in
                                 per_loop[2 * c:2 * c + 2]]
            if (per_loop[2 * c]["wave"] != c or
                    pointers["rs1"]["buffer"] != "activation_scales" or
                    pointers["rs1"]["byte_offset"] != 0 or
                    pointers["rs2"]["buffer"] != "weight_scales" or
                    pointers["rs2"]["byte_offset"] != c * nc or
                    strides["rs1"]["immediate"] != 128 or
                    strides["rs2"]["immediate"] != 128):
                raise ValueError("native loop-managed scale pointers differ")
    else:
        expected_scales = []
        for c in range(chunks):
            for buffer, offset, count, dest, selector in (
                    ("activation_scales", 0, 128, c * 4 * 128, 0),
                    ("weight_scales", c * nc, nc, c * 4 * nc, 1)):
                expected_scales.append((c, buffer, offset, 128 << 40,
                                        (4 << 46) | (dest << 33) |
                                        (selector << 32) | count))
        actual_scales = [(step["wave"], step["command"]["rs1"]["buffer"],
                          step["command"]["rs1"]["byte_offset"],
                          step["command"]["rs1"]["or_bits"],
                          step["command"]["rs2"]["immediate"])
                         for step in scales]
        if (actual_scales != expected_scales or len(selectors) != 1 or
                selectors[0].get("funct") != 26 or
                selectors[0]["rs1"]["or_bits"] !=
                ((8 * chunks << 51) | ((8 // chunks) << 42) | (8 << 33)) or
                selectors[0]["rs2"]["immediate"] !=
                (1 | ((1 << 16) if scale_wait else 0)) or
                sum(step["command"] == {} for step in steps
                    if step["phase"] == "upload_chunk_scales") !=
                (0 if scale_wait else 1)):
            raise ValueError("native column-loop scales differ from source")
    return {"schema": ("mx_gemmini.nicolas_fp8_native_dram_ls_audit.v1"
                       if loop_scales else
                       "mx_gemmini.nicolas_fp8_native_dram_nc_audit.v1"),
            "source_driver_sha256": case.source_sha256,
            "profile_sha256": profile_sha256(profile),
            "physical_program_sha256": _sha(object_dir / "physical_program.json"),
            "native_chunk_count": chunks,
            "native_command_functs": [command["funct"] for command in native[:-1]],
            "scale_uploads": len(scales),
            **({"store_activation": "relu",
                "derived_expected_bf16_sha256":
                physical["derived_expected_bf16_sha256"]} if store_relu else {}),
            **({"scale_config_wait_bit": True} if scale_wait else {}),
            **({"loop_scale_configurations": 2 * chunks}
               if loop_scales else {}),
            "output_column_bytes_per_chunk": nc * 2,
            "b_spad_ids": [1 + (c & 1) for c in range(chunks)],
            "explicit_operand_dma_commands": 0,
            "explicit_output_dma_commands": 0}


def _audit_native_dram_kt_fp8(object_dir: Path, profile: dict,
                              case: Case) -> dict:
    """Audit four native launches over two N chunks and two K tiles."""
    if (case.native_dram_chunks != 2 or case.native_dram_k_tiles != 2 or
            case.native_dram_scale_mode != "loop" or
            profile["geometry"]["mesh_columns"] != 16):
        raise ValueError("K-tiled native audit needs Nicolas's DIM16 recipe")
    physical = json.loads((object_dir / "physical_program.json").read_text())
    plan = physical["plan"]
    steps = physical["steps"]
    if (physical["shape_mnk"] != [128, 128, 128] or
            plan.get("execution_transport") != "native_dram_loop" or
            plan.get("native_dram_n_chunks") != 2 or
            plan.get("native_dram_k_tiles") != 2 or
            plan.get("native_dram_scale_mode") != "loop_managed" or
            any(step["phase"] in {"move_activation", "move_weight", "compute",
                                   "readout", "upload_scales", "select_scales"}
                for step in steps)):
        raise ValueError("compiler did not select native K-tiled DRAM loop")
    ordered = [(step["phase"], step["command"].get("funct")) for step in steps
               if step["phase"] in {"loop_scales", "native_dram_loop"}]
    # Keep the source's scale pointer pair immediately before each loop.
    expected_order = []
    for _ in range(4):
        expected_order.extend([("loop_scales", 31), ("loop_scales", 32)])
        expected_order.extend(("native_dram_loop", funct)
                              for funct in (9, 10, 11, 12, 13, 8))
    expected_order.append(("native_dram_loop", None))
    if ordered != expected_order:
        raise ValueError("native K-tile loop issue order differs from source")
    scale = [step["command"] for step in steps if step["phase"] == "loop_scales"]
    native = [step["command"] for step in steps
              if step["phase"] == "native_dram_loop" and
              step["command"].get("funct") is not None]
    for c in range(2):
        for t in range(2):
            index = c * 2 + t
            pointers, pitch = scale[2 * index:2 * index + 2]
            if (pointers["rs1"]["buffer"] != "activation_scales" or
                    pointers["rs1"]["byte_offset"] != t * 256 or
                    pointers["rs2"]["buffer"] != "weight_scales" or
                    pointers["rs2"]["byte_offset"] != t * 256 + c * 64 or
                    pitch["rs1"]["immediate"] != 128 or
                    pitch["rs2"]["immediate"] != 128):
                raise ValueError("native K-tile scale slice differs from source")
            command = native[6 * index:6 * index + 6]
            expected = (
                (0, (4 << 32) | (4 << 16) | 8),
                ("activation", "weight"),
                (0, "output_bf16" if t == 1 else 0),
                (128, 128),
                (0, 128),
                ((1 << 18) | ((1 + (index & 1)) << 16) | int(t > 0), 0),
            )
            actual = tuple(tuple(item[register]["buffer"] if
                                 item[register]["buffer"] is not None else
                                 item[register]["immediate"]
                                 for register in ("rs1", "rs2")) for item in command)
            if (actual != expected or
                    command[1]["rs1"]["byte_offset"] != t * 64 or
                    command[1]["rs2"]["byte_offset"] != t * 64 * 128 + c * 64 or
                    (t == 1 and command[2]["rs2"]["byte_offset"] != c * 128)):
                raise ValueError("native K-tile pointer, accumulation, or bank differs")
    return {"schema": "mx_gemmini.nicolas_fp8_native_dram_kt_audit.v1",
            "source_driver_sha256": case.source_sha256,
            "profile_sha256": profile_sha256(profile),
            "physical_program_sha256": _sha(object_dir / "physical_program.json"),
            "native_chunk_count": 2, "k_tiles_per_chunk": 2,
            "loop_scale_configurations": len(scale),
            "native_command_functs": [command["funct"] for command in native],
            "accumulating_loops": 2, "store_loops": 2,
            "explicit_operand_dma_commands": 0,
            "explicit_output_dma_commands": 0}


def _run_alternate_fp8_source(rtl_root: Path, riscv_root: Path,
                              out_dir: Path, source: Path) -> dict:
    """Run the pinned handwritten program as an independent source oracle."""
    from tools.qualify_nicolas_spad_requant_fp4 import _compile_program

    software = rtl_root / "software/gemmini-rocc-tests"
    build = out_dir / "source_baseline"
    elf = _compile_program(build, source, software,
                           riscv_root / "bin/riscv64-unknown-elf-gcc", None)
    so = out_dir / "run/libgemmini.so"
    result = subprocess.run([str(riscv_root / "bin/spike"), f"--extlib={so}",
                             "--extension=gemmini", str(elf)],
                            cwd=build, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log = build / "spike.log"
    log.write_text(result.stdout)
    if (result.returncode or
            "fp8 WS matmul test PASSED (no mismatches)." not in result.stdout):
        raise ValueError("Nicolas alternate FP8 source executable failed on pinned Spike")
    return {"source_elf_sha256": _sha(elf), "source_spike_log_sha256": _sha(log),
            "source_spike_exit_code": result.returncode,
            "source_golden_bf16_values_checked": 1024}


def _run_chunked_fp8_source(rtl_root: Path, riscv_root: Path,
                            out_dir: Path, source: Path) -> dict:
    """Run the pinned chunked handwritten program as an independent oracle."""
    from tools.qualify_nicolas_spad_requant_fp4 import _compile_program

    software = rtl_root / "software/gemmini-rocc-tests"
    build = out_dir / "source_baseline"
    elf = _compile_program(build, source, software,
                           riscv_root / "bin/riscv64-unknown-elf-gcc", None)
    so = out_dir / "run/libgemmini.so"
    result = subprocess.run([str(riscv_root / "bin/spike"), f"--extlib={so}",
                             "--extension=gemmini", str(elf)],
                            cwd=build, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log = build / "spike.log"
    log.write_text(result.stdout)
    if (result.returncode or
            "fp8 WS chunked matmul test PASSED (no mismatches)." not in result.stdout):
        raise ValueError("Nicolas chunked FP8 source failed on pinned Spike")
    return {"source_elf_sha256": _sha(elf), "source_spike_log_sha256": _sha(log),
            "source_spike_exit_code": result.returncode,
            "source_golden_bf16_values_checked": 128 * 128}


def _run_native_fp8_source(rtl_root: Path, riscv_root: Path,
                           out_dir: Path, source: Path, chunks: int) -> dict:
    """Run Nicolas's pinned native DRAM-loop program on the same Spike."""
    from tools.qualify_nicolas_spad_requant_fp4 import _compile_program

    software = rtl_root / "software/gemmini-rocc-tests"
    build = out_dir / "source_baseline"
    elf = _compile_program(build, source, software,
                           riscv_root / "bin/riscv64-unknown-elf-gcc", None)
    so = out_dir / "run/libgemmini.so"
    result = subprocess.run([str(riscv_root / "bin/spike"), f"--extlib={so}",
                             "--extension=gemmini", str(elf)],
                            cwd=build, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log = build / "spike.log"
    log.write_text(result.stdout)
    success = ("fp8 WS native-loop matmul test PASSED (no mismatches)."
               if chunks == 1 else
               f"fp8 WS native multi-loop ({chunks} chunks) test PASSED (no mismatches).")
    if result.returncode or success not in result.stdout:
        raise ValueError("Nicolas native FP8 DRAM-loop source failed on pinned Spike")
    return {"source_elf_sha256": _sha(elf), "source_spike_log_sha256": _sha(log),
            "source_spike_exit_code": result.returncode,
            "source_golden_bf16_values_checked": 128 * 128}


def _run_plain_fp6_source(rtl_root: Path, riscv_root: Path,
                          out_dir: Path, source: Path) -> dict:
    """Run Nicolas's original FP6 128³ source as an independent oracle."""
    from tools.qualify_nicolas_spad_requant_fp4 import _compile_program

    software = rtl_root / "software/gemmini-rocc-tests"
    build = out_dir / "source_baseline"
    elf = _compile_program(build, source, software,
                           riscv_root / "bin/riscv64-unknown-elf-gcc", None)
    so = out_dir / "run/libgemmini.so"
    result = subprocess.run([str(riscv_root / "bin/spike"), f"--extlib={so}",
                             "--extension=gemmini", str(elf)],
                            cwd=build, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log = build / "spike.log"
    log.write_text(result.stdout)
    if (result.returncode or
            "fp6 WS matmul test PASSED (no mismatches)." not in result.stdout):
        raise ValueError("Nicolas plain FP6 128-cubed source failed on pinned Spike")
    return {"source_elf_sha256": _sha(elf), "source_spike_log_sha256": _sha(log),
            "source_spike_exit_code": result.returncode,
            "source_golden_bf16_values_checked": 128 * 128}


def _run_smem_fp8_source(rtl_root: Path, riscv_root: Path,
                          out_dir: Path, source: Path, case: Case) -> dict:
    """Run the pinned zero-base SMEM source under its Spike path."""
    from tools.qualify_nicolas_spad_requant_fp4 import _compile_program

    software = rtl_root / "software/gemmini-rocc-tests"
    build = out_dir / "source_baseline"
    elf = _compile_program(build, source, software,
                           riscv_root / "bin/riscv64-unknown-elf-gcc", None)
    so = out_dir / "run/libgemmini.so"
    result = subprocess.run([str(riscv_root / "bin/spike"), f"--extlib={so}",
                             "--extension=gemmini", str(elf)],
                            cwd=build, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log = build / "spike.log"
    log.write_text(result.stdout)
    if (result.returncode or
            "fp8 WS matmul test PASSED (no mismatches)." not in result.stdout):
        raise ValueError("Nicolas zero-base SMEM source failed on pinned Spike")
    return {"source_elf_sha256": _sha(elf), "source_spike_log_sha256": _sha(log),
            "source_spike_exit_code": result.returncode,
            "source_golden_bf16_values_checked": case.shape[0] * case.shape[1],
            "source_execution_path": ("SPIKE_SIM_scratchpad_fallback" if
                                      case.dram_mvout_spike_fallback else
                                      "SPIKE_SIM_scratchpad_readback")}


def source_kernel(rtl_root: Path, case: Case) -> SourceGemm:
    """Admit only a Nicolas test whose header and golden were audited."""
    if _revision(rtl_root) != RTL_REVISION:
        raise ValueError("Nicolas FP8 qualifier requires the pinned MX RTL revision")
    for submodule in ("software/gemmini-rocc-tests", "software/libgemmini"):
        _require_gitlink(rtl_root, submodule)
    software = rtl_root / "software/gemmini-rocc-tests"
    driver = software / "bareMetalC" / case.source_name
    header = software / "include" / case.header_name
    if _sha(driver) != case.source_sha256 or _sha(header) != case.header_sha256:
        raise ValueError(f"Nicolas {case.precision} driver/header differs from pinned source")
    source = driver.read_text()
    native_wrappers = {
        "matmul_tiled_fp8_128x128_dramloop_nc4.c":
            '// 4-chunk variant of matmul_tiled_fp8_128x128_dramloop_nc.\n'
            '#define NCHUNKS 4\n'
            '#include "matmul_tiled_fp8_128x128_dramloop_nc.c"',
        "matmul_tiled_fp8_128x128_dramloop_ls.c":
            '// dramloop_nc with LOOP-MANAGED scales (2 chunks): no MX_LOAD_SCALES / CONFIG_SCALE_MEM / fence.\n'
            '#define NCHUNKS 2\n#define LOOP_SCALES 1\n'
            '#include "matmul_tiled_fp8_128x128_dramloop_nc.c"',
        "matmul_tiled_fp8_128x128_dramloop_ls4.c":
            '// dramloop_nc with LOOP-MANAGED scales (4 chunks): no MX_LOAD_SCALES / CONFIG_SCALE_MEM / fence.\n'
            '#define NCHUNKS 4\n#define LOOP_SCALES 1\n'
            '#include "matmul_tiled_fp8_128x128_dramloop_nc.c"',
        "matmul_tiled_fp8_128x128_dramloop_kt.c":
            '// dramloop_nc with native K-tiling: 2 chunks x 2 K-tiles of 64, accumulating loops (C on the last), loop scales.\n'
            '#define NCHUNKS 2\n#define K_TILES 2\n#define LOOP_SCALES 1\n'
            '#include "matmul_tiled_fp8_128x128_dramloop_nc.c"',
        "matmul_tiled_fp8_128x128_dramloop_nc_2d.c":
            '// matmul_tiled_fp8_128x128_dramloop_nc with 2-D MX_LOAD_SCALES (no host packing).\n'
            '#define SCALES_2D 1\n'
            '#include "matmul_tiled_fp8_128x128_dramloop_nc.c"',
        "matmul_tiled_fp8_128x128_dramloop_nc_wait.c":
            '// dramloop_nc with 2-D scale loads and NO fence: CONFIG_SCALE_MEM waits in HW (rs2[16]).\n'
            '#define SCALES_2D 1\n#define SCALE_WAIT 1\n'
            '#include "matmul_tiled_fp8_128x128_dramloop_nc.c"',
        "matmul_tiled_fp8_128x128_dramloop_relu.c":
            '// dramloop_ls with ReLU on the accumulated output (store-path activation); golden = relu(C_out_bf16).\n'
            '#define NCHUNKS 2\n#define LOOP_SCALES 1\n#define RELU_OUT 1\n'
            '#include "matmul_tiled_fp8_128x128_dramloop_nc.c"',
    }
    if case.native_dram and case.source_name in native_wrappers:
        if source.strip() != native_wrappers[case.source_name]:
            raise ValueError("Nicolas native column-loop wrapper changed")
        core = driver.parent / "matmul_tiled_fp8_128x128_dramloop_nc.c"
        if _sha(core) != CASES["fp8_128x128x128_native_dram_nc2"].source_sha256:
            raise ValueError("Nicolas native column-loop include changed")
        source = core.read_text()
    if case.i_chunks == 4 or case.key == _chunk2d_key:
        expected = (
            '// 4-chunk variant of matmul_tiled_fp8_128x128_chunked '
            '(32 rows per chunk; chunks alternate the 2 acc banks).\n'
            '#define NCHUNKS 4\n'
            '#include "matmul_tiled_fp8_128x128_chunked.c"'
            if case.i_chunks == 4 else
            '// matmul_tiled_fp8_128x128_chunked with 2-D MX_LOAD_SCALES '
            '(no host packing).\n'
            '#define SCALES_2D 1\n'
            '#include "matmul_tiled_fp8_128x128_chunked.c"')
        if source.strip() != expected:
            raise ValueError("Nicolas I-chunk wrapper changed")
        core = driver.parent / "matmul_tiled_fp8_128x128_chunked.c"
        if _sha(core) != CASES["fp8_128x128x128_chunked2"].source_sha256:
            raise ValueError("Nicolas I-chunk include changed")
        source = core.read_text()
    if case.i_chunks and case.i_chunks != 1 and (
            "#define NCHUNKS 2" not in source or
            "0x38 | 0x100" not in source or
            "NCHUNKS * tiles_K" not in source or
            "gemmini_mx_load_scales_2d" not in source):
        raise ValueError("Nicolas chunked FP8 schedule changed")
    alternate_mvin = case.key == "fp8_32x32x32_alternate_mvin"
    required = (f'#include "include/{case.header_name}"', "gemmini_mx_load_scales",
                "gemmini_loop_ws" if case.native_dram else
                "gemmini_loop_ws_spad",
                "C_hw" if case.native_dram else
                "gemmini_mx_read_smem" if alternate_mvin or
                case.smem_zero_readout else
                "gemmini_extended_mvout")
    if not case.quant_output:
        required += ("C_out_bf16",)
    if case.precision == "FP6":
        required += ("gemmini_mx_load_lut_dt",)
    if case.quant_output:
        required += (("gemmini_mxquant_config_mvout", "C_proj_hw[i][j]",
                      "C_scales_row[b][i]") if case.precision == "FP6" else
                     ("gemmini_mxquant_config_mvout", "C_scales_out", "C_out[i][j]"))
    if any(needle not in source for needle in required):
        raise ValueError("Nicolas source no longer has the audited compute/check path")
    if alternate_mvin and any(source.count(marker) != 1 for marker in (
            "elem_t *dram_ptr = ((elem_t*)B_in) + j * DIM * MATMUL_M + k * DIM;",
            "uint32_t sp_addr = b_base + (j * tiles_K + k) * DIM;",
            "gemmini_mx_read_smem(&C_hw[0][0], SPAD_DEST * 16, MATMUL_M * MATMUL_N);")):
        raise ValueError("Nicolas alternate FP8 transfer source changed")
    if case.native_dram and case.native_dram_chunks == 1 and any(marker not in source for marker in (
            "A_in, B_in, NULL, C_hw,", "MATMUL_K, MATMUL_N, 0, MATMUL_N,",
            "1, 1, false);", "gemmini_mxquant_config_mvout")):
        raise ValueError("Nicolas native DRAM-loop source changed")
    if case.native_dram and case.native_dram_chunks > 1 and any(
            marker not in source for marker in (
                "c == 0 ? (const void *) A_in : NULL",
                "(const uint8_t *) B_in + c * NC",
                "&C_hw[0][c * NC]", "1 + (c & 1)",
                "NCHUNKS * K", "gemmini_loop_ws(I, J, K")):
        raise ValueError("Nicolas native column-chunk loop source changed")
    if case.native_dram_store_activation == "relu" and any(
            marker not in source for marker in (
                "gemmini_extended_config_st(MATMUL_N * sizeof(uint16_t), RELU, ACC_SCALE_IDENTITY)",
                "(C_out_bf16[i][j] & 0x8000) ? 0 : C_out_bf16[i][j]")):
        raise ValueError("Nicolas native ReLU store/golden branch changed")
    smem_markers = (("uint32_t c_dest = 0;", "uint32_t c_flag = 0x38;",
                     "uint32_t c_dest = acc_addr;", "uint32_t c_flag = 0xb8;",
                     "gemmini_mx_read_smem(&C_hw[0][0], 0, MATMUL_M * MATMUL_N);",
                     "gemmini_mvout((void *) dram_ptr, acc_tile_addr);")
                    if case.dram_mvout_spike_fallback else
                    ("int SPAD_DEST = 0;",
                     "gemmini_mx_read_smem(&C_hw[0][0], 0, MATMUL_M * MATMUL_N);",
                     "gemmini_config_st(DIM * sizeof(elem_t));", "0x38);"))
    if case.smem_zero_readout and any(marker not in source for marker in smem_markers):
        raise ValueError("Nicolas zero-base SMEM readback source changed")
    header_text = header.read_text()
    for axis, extent in zip(("M", "N", "K"), case.shape):
        if re.search(rf"^#define MATMUL_{axis}\s+{extent}$", header_text, re.M) is None:
            raise ValueError("Nicolas source header shape changed")
    return SourceGemm(driver, header, case.shape, case.tile,
                      case.precision, case.quant_output, False, True)


def capture_handoff(model2mlir: Path, mxq_root: Path, kernel: SourceGemm,
                    directory: Path) -> tuple[str, str, dict]:
    sys.path[:0] = [str(model2mlir), str(mxq_root)]
    import m2m
    import mxq
    import torch
    import yaml
    from m2m.capture.external_quantization import ExternalQuantizationConfig
    from m2m.coverage import opaque_report
    from mx_gemmini_support.handoff import render_handoff, validate_handoff

    if (Path(m2m.__file__).resolve().parents[1] != model2mlir or
            Path(mxq.__file__).resolve().parents[1] != mxq_root or
            _revision(model2mlir) != MODEL2MLIR_REVISION or
            _revision(mxq_root) != MXQ_REVISION):
        raise ValueError("model2MLIR/MXQuant checkout differs from selected frontend")

    class Matmul(torch.nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b)

    m, n, k = kernel.shape
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        example = (torch.randn((m, k), dtype=torch.float32),
                   torch.randn((k, n), dtype=torch.float32))
    contract = ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
    policy = ROOT / "examples/default-policy.yaml"
    if kernel.datatype == "FP4":
        policy = ROOT / "examples/fp4-policy.yaml"
    if kernel.datatype == "FP6":
        fp6 = read_source_fp6_payload(kernel)
        policy = directory / "source_line0_policy.yaml"
        # MXQuant's structural capture requires a one-to-one codebook. The
        # source payload, bound after capture, retains every original LUT line
        # and packed alias index. Fill only missing codes in the capture's
        # line-zero witness when the source LUT has repeated values.
        def capture_codebook(source_line: tuple[int, ...]) -> list[int]:
            values = list(dict.fromkeys(source_line))
            values.extend(code for code in range(64) if code not in values)
            return values[:16]
        policy.write_text(yaml.safe_dump({
            "schema": "mx_gemmini.quantization_policy.v1",
            "default_format": "mxfp6", "module_overrides": {},
            "functional_overrides": {}, "output_chains": {},
            "fp6_codebooks": {"default": {
                "status": "reviewed",
                "activation": capture_codebook(fp6.activation_lut_line0),
                "weight": capture_codebook(fp6.weight_lut_line0)}}}, sort_keys=False))
    result = m2m.convert(
        Matmul().eval(), example,
        quantization=ExternalQuantizationConfig("mx_gemmini", contract, policy),
        backend="fx_importer", capture_trace=True)
    if not result.ok or opaque_report(result.mlir_text):
        raise ValueError(f"model2MLIR {kernel.datatype} matmul capture failed: {result.diagnostics}")
    sites = result.quantization_manifest["sites"]
    if [(site["site_id"], site["status"], site["format"], site["shape"])
            for site in sites] != [
                ("functional:matmul", "quantized",
                 {"FP8": "mxfp8", "FP4": "mxfp4",
                  "FP6": "mxfp6"}[kernel.datatype], [m, n, k])]:
        raise ValueError(f"model2MLIR did not capture the selected matmul: {sites}")
    contract_bytes, policy_bytes = contract.read_bytes(), policy.read_bytes()
    validate_handoff(result, contract_bytes, policy_bytes)
    return (result.mlir_text, render_handoff(result, contract_bytes, policy_bytes),
            result.quantization_manifest)


def _driver(case: Case) -> str:
    if case.quant_output:
        if case.precision not in {"FP8", "FP4", "FP6"}:
            raise ValueError("source quantized driver needs FP8, FP4, or FP6 output")
        output_rows = "MATMUL_M/2" if case.precision in {"FP4", "FP6"} else "MATMUL_M"
        code_label = "packed-byte" if case.precision in {"FP4", "FP6"} else "code"
        source_code = "C_proj_hw[i][j]" if case.precision == "FP6" else "C_out[i][j]"
        source_scale = "C_scales_row[g][i]" if case.precision == "FP6" else "C_scales_out[i][g]"
        return f"""#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/{case.header_name}"
#include "mx_issue.h"
static uint8_t C_hw[{output_rows}][MATMUL_N] __attribute__((aligned(64)));
static uint8_t scratch_output_scales[2048] __attribute__((aligned(64)));
int main(void) {{
  mx_issue({', '.join(case.call_arguments)});
  gemmini_fence();
  int code_errors = 0, scale_errors = 0;
  for (int i = 0; i < {output_rows}; ++i)
    for (int j = 0; j < MATMUL_N; ++j)
      if (C_hw[i][j] != {source_code}) {{
        if (code_errors < 8) printf("{code_label} mismatch %d,%d got %x want %x\\n",
                                    i,j,C_hw[i][j],{source_code});
        ++code_errors;
      }}
  for (int i = 0; i < MATMUL_M; ++i)
    for (int g = 0; g < MATMUL_GN; ++g)
      if (scratch_output_scales[i*MATMUL_GN+g] != {source_scale}) {{
        if (scale_errors < 8) printf("scale mismatch %d,%d got %x want %x\\n",
                                     i,g,scratch_output_scales[i*MATMUL_GN+g],
                                     {source_scale});
        ++scale_errors;
      }}
  printf("compiled Nicolas {case.label}: %d {code_label} mismatches / %d, %d scale mismatches / %d\\n",
         code_errors, {output_rows}*MATMUL_N, scale_errors, MATMUL_M*MATMUL_GN);
  return code_errors != 0 || scale_errors != 0;
}}
"""
    scratch_decl = ("static uint8_t scratch_output_scales[2048] "
                    "__attribute__((aligned(64)));\n"
                    if "scratch_output_scales" in case.buffer_abi else "")
    expected_bf16 = ("((C_out_bf16[i][j] & 0x8000) ? 0 : C_out_bf16[i][j])"
                     if case.native_dram_store_activation == "relu" else
                     "C_out_bf16[i][j]")
    return f"""#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/{case.header_name}"
#include "mx_issue.h"
static uint16_t C_hw[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));
{scratch_decl}int main(void) {{
  mx_issue({', '.join(case.call_arguments)});
  gemmini_fence();
  int errors = 0;
  for (int i = 0; i < MATMUL_M; ++i)
    for (int j = 0; j < MATMUL_N; ++j)
      if (C_hw[i][j] != {expected_bf16}) {{
        if (errors < 8) printf("mismatch %d,%d got %x want %x\\n",
                               i,j,C_hw[i][j],{expected_bf16});
        ++errors;
      }}
  printf("compiled Nicolas {case.label}: %d mismatches / %d BF16 values\\n",
         errors, MATMUL_M * MATMUL_N);
  return errors != 0;
}}
"""


def run_spike(rtl_root: Path, riscv_root: Path, object_dir: Path,
              out_dir: Path, case: Case, mesh_dim: int) -> tuple[int, str, Path]:
    if mesh_dim not in {8, 16, 32}:
        raise ValueError("Nicolas source qualifier needs a supported MX mesh dimension")
    software = rtl_root / "software/gemmini-rocc-tests"
    bench = software / "riscv-tests/benchmarks/common"
    build = out_dir / "run"
    build.mkdir()
    abi = json.loads((object_dir / "object_manifest.json").read_text())["buffer_abi"]
    scratch_slots = [slot for slot in abi
                     if slot["name"] == "scratch_output_scales"]
    if ([slot["name"] for slot in abi] != list(case.buffer_abi) or
            [slot["position"] for slot in abi] != list(range(len(abi))) or
            any(slot["minimum_bytes"] > 2048 for slot in scratch_slots)):
        raise ValueError("compiled MX object buffer ABI differs from checked driver")
    (build / "mx_driver.c").write_text(_driver(case))
    cc = riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = riscv_root / "bin/spike"
    if not all(path.is_file() for path in (cc, spike, bench / "test.ld")):
        raise ValueError("RISC-V GCC, Spike, or Nicolas benchmark linker is missing")
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"),
             "-I", str(software), "-I", str(bench), "-I", str(object_dir)]
    sources = [build / "mx_driver.c", *sorted(bench.glob("*.c")),
               *sorted(bench.glob("*.S"))]
    objects = []
    for index, source in enumerate(sources):
        obj = build / f"driver_{index}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(obj)],
             cwd=build, log=build / f"compile_{index}.log")
        objects.append(obj)
    elf = build / "mx_program.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), str(object_dir / "mx_issue.o"),
          *(str(obj) for obj in objects), "-lm", "-lgcc", "-o", str(elf)],
         cwd=build, log=build / "link.log")
    extension = rtl_root / "software/libgemmini"
    extension_sources = [extension / "gemmini.cc", extension / "gemmini_perf.cc",
                         *sorted((extension / "perf").rglob("*.cc"))]
    so = build / "libgemmini.so"
    _run(["g++", *([f"-DGEMMINI_DIM={mesh_dim}"] if mesh_dim != 16 else []),
          "-L", str(riscv_root / "lib"),
          f"-Wl,-rpath,{riscv_root / 'lib'}", "-shared", "-o", str(so),
          "-std=c++17", "-I", str(riscv_root / "include"), "-fPIC", "-O3",
          *(str(path) for path in extension_sources)],
         cwd=build, log=build / "extension.log")
    extension_name = "gemmini" if mesh_dim == 16 else f"gemmini_dim{mesh_dim}"
    run = subprocess.run([str(spike), f"--extlib={so}",
                          f"--extension={extension_name}",
                          str(elf)], cwd=build, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=False)
    (build / "spike.log").write_text(run.stdout)
    return run.returncode, run.stdout, elf


def main(default_case: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES,
                        required=default_case is None, default=default_case)
    for name in ("model2mlir-root", "mxq-root", "rtl-root", "profile",
                 "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    case = CASES[args.case]
    rtl_root = args.rtl_root.resolve()
    kernel = source_kernel(rtl_root, case)
    profile = load_profile(args.profile, rtl_root=rtl_root)
    if profile["name"] != case.profile_name:
        raise ValueError(f"Nicolas {case.precision} source needs {case.profile_name}")
    args.out_dir.mkdir(parents=True)
    source_mlir, handoff, quant_manifest = capture_handoff(
        args.model2mlir_root.resolve(), args.mxq_root.resolve(), kernel, args.out_dir)
    (args.out_dir / "model2mlir.mlir").write_text(source_mlir)
    (args.out_dir / "handoff.mlir").write_text(handoff)
    (args.out_dir / "quantization_manifest.json").write_text(
        json.dumps(quant_manifest, indent=2, sort_keys=True) + "\n")
    bundle = args.out_dir / "bundle"
    manifest = write_bundle(bundle, kernel, site_id="functional:matmul",
                            profile_sha256=profile_sha256(profile),
                            source_origin=NICOLAS_SOURCE_HEADER_ORIGIN)
    if case.quant_output:
        _, resources = load_bundle(bundle)
        output_format = {"FP8": "fp8_e4m3", "FP4": "fp4_e2m1",
                         "FP6": "fp6_e3m2"}[case.precision]
        source_code = {"FP8": "golden_fp8", "FP4": "source_fp4_packed",
                       "FP6": "source_fp6_packed"}[case.precision]
        target_code = {"FP8": "nicolas_fp8", "FP4": "nicolas_fp4",
                       "FP6": "nicolas_fp6"}[case.precision]
        if (manifest.get("output_format") != output_format or
                resources[source_code] != resources[target_code] or
                resources["golden_output_scales"] != resources["nicolas_output_scales"]):
            raise ValueError("Nicolas source codes/scales differ from target quantization")
    bound = bind_handoff(handoff, profile)
    bound = bind_payload(bound, profile, manifest)
    if not case.quant_output:
        bound = select_bf16_output_layout(bound, profile, manifest)
    if case.i_chunks > 1:
        bound = bind_i_chunks(bound, profile, manifest, case.i_chunks)
    if case.native_dram:
        bound = bind_native_dram(bound, profile, manifest,
                                 case.native_dram_chunks,
                                 scale_mode=case.native_dram_scale_mode,
                                 k_tiles=case.native_dram_k_tiles,
                                 store_activation=case.native_dram_store_activation)
    if case.smem_zero_readout:
        bound = bind_smem_zero_readout(bound, profile, manifest)
    mlir = args.out_dir / "payload_bound.mlir"
    mlir.write_text(bound)
    object_dir = args.out_dir / "object"
    _run([sys.executable, "-m", "tools.compile_object", "--mlir", str(mlir),
          "--bundle", str(bundle), "--profile", str(args.profile.resolve()),
          "--rtl-root", str(rtl_root), "--riscv-root", str(args.riscv_root.resolve()),
          "--mx-opt", str(args.mx_opt.resolve()), "--out-dir", str(object_dir)],
         cwd=ROOT, log=args.out_dir / "object_compile.log")
    transfer_audit = None
    if case.key == "fp8_32x32x32_alternate_mvin":
        transfer_audit = _audit_alternate_fp8_mvin(
            object_dir, profile, case.source_sha256)
        (args.out_dir / "transfer_equivalence.json").write_text(
            json.dumps(transfer_audit, indent=2, sort_keys=True) + "\n")
    chunk_audit = None
    if case.i_chunks > 1:
        chunk_audit = _audit_chunked_fp8(object_dir, profile, case)
        (args.out_dir / "chunk_equivalence.json").write_text(
            json.dumps(chunk_audit, indent=2, sort_keys=True) + "\n")
    native_audit = None
    if case.native_dram:
        native_audit = _audit_native_dram_fp8(object_dir, profile, case)
        (args.out_dir / "native_dram_equivalence.json").write_text(
            json.dumps(native_audit, indent=2, sort_keys=True) + "\n")
    smem_audit = None
    if case.smem_zero_readout:
        smem_audit = _audit_smem_zero_readout(object_dir, profile, case)
        (args.out_dir / "smem_readout_equivalence.json").write_text(
            json.dumps(smem_audit, indent=2, sort_keys=True) + "\n")
    mesh_dim = profile["geometry"]["mesh_columns"]
    if profile["geometry"]["mesh_rows"] != mesh_dim:
        raise ValueError("Nicolas source qualifier needs a square MX mesh")
    returncode, output, elf = run_spike(rtl_root, args.riscv_root.resolve(),
                                        object_dir, args.out_dir, case, mesh_dim)
    source_baseline = (_run_alternate_fp8_source(
        rtl_root, args.riscv_root.resolve(), args.out_dir, kernel.driver)
        if case.key == "fp8_32x32x32_alternate_mvin" else
        _run_chunked_fp8_source(rtl_root, args.riscv_root.resolve(),
                                args.out_dir, kernel.driver)
        if case.i_chunks > 1 else
        _run_native_fp8_source(rtl_root, args.riscv_root.resolve(),
                               args.out_dir, kernel.driver,
                               case.native_dram_chunks)
        if case.native_dram else
        _run_plain_fp6_source(rtl_root, args.riscv_root.resolve(),
                              args.out_dir, kernel.driver)
        if case.key == "fp6_128x128x128" else
        _run_smem_fp8_source(rtl_root, args.riscv_root.resolve(),
                             args.out_dir, kernel.driver, case)
        if case.smem_zero_readout else None)
    m, n, _ = case.shape
    code_label = "packed-byte" if case.precision in {"FP4", "FP6"} else "code"
    code_count = m * n // 2 if case.precision in {"FP4", "FP6"} else m * n
    expected = (f"compiled Nicolas {case.label}: 0 {code_label} mismatches / {code_count}, "
                f"0 scale mismatches / {m * n // 32}" if case.quant_output else
                f"compiled Nicolas {case.label}: 0 mismatches / {m * n} BF16 values")
    passed = returncode == 0 and expected in output
    dispatch = json.loads((object_dir / "compile_manifest.json").read_text())
    receipt = {
        "schema": f"mx_gemmini.nicolas_plain_{case.precision.lower()}_typed_object_spike.v1",
        "status": "source_golden_matched_on_pinned_spike" if passed else
                  "source_golden_failed_on_pinned_spike",
        "scope": (f"Nicolas {case.source_name}: packed {case.precision} source arrays, "
                  f"{m}x{n}x{case.shape[2]} " +
                  (f"source-header {case.precision} codes and E8M0 scales" if case.quant_output
                   else "BF16 output") + "; PyTorch model2MLIR capture, "
                  "typed source binding, public object compiler, full Spike comparison"),
        "source_driver_sha256": _sha(kernel.driver),
        "source_header_sha256": _sha(kernel.data_header),
        "rtl_revision": _revision(rtl_root),
        "software_revision": _revision(rtl_root / "software/gemmini-rocc-tests"),
        "model2mlir_revision": _revision(args.model2mlir_root),
        "mxq_revision": _revision(args.mxq_root),
        "compiler_revision": dispatch["compiler_revision"],
        "compiler_source_closure_sha256": dispatch["compiler_source_closure_sha256"],
        "profile_name": profile["name"], "profile_sha256": profile_sha256(profile),
        "mesh_dim": mesh_dim,
        "case": case.key,
        "source_mlir_sha256": _sha(args.out_dir / "model2mlir.mlir"),
        "handoff_mlir_sha256": _sha(args.out_dir / "handoff.mlir"),
        "payload_manifest_sha256": _sha(bundle / "manifest.json"),
        "bound_mlir_sha256": _sha(mlir),
        "object_sha256": _sha(object_dir / "mx_issue.o"),
        "object_dispatch_manifest_sha256": _sha(object_dir / "compile_manifest.json"),
        "object_manifest_sha256": _sha(object_dir / "object_manifest.json"),
        "elf_sha256": _sha(elf), "spike_log_sha256": _sha(args.out_dir / "run/spike.log"),
        "spike_sha256": _sha(args.riscv_root.resolve() / "bin/spike"),
        "spike_extension_sha256": _sha(args.out_dir / "run/libgemmini.so"),
        "driver_sha256": _sha(args.out_dir / "run/mx_driver.c"),
        "outputs_checked": code_count + m * n // 32 if case.quant_output else m * n,
        "codes_checked": m * n if case.quant_output and case.precision == "FP8" else None,
        "packed_bytes_checked": (code_count if case.quant_output and
                                 case.precision in {"FP4", "FP6"} else None),
        "scales_checked": m * n // 32 if case.quant_output else None,
        "mismatches": 0 if passed else None,
    }
    if transfer_audit is not None:
        receipt["transfer_equivalence_sha256"] = _sha(
            args.out_dir / "transfer_equivalence.json")
        receipt["source_baseline"] = source_baseline
    if case.key == "fp6_128x128x128":
        receipt["source_baseline"] = source_baseline
        receipt["fp6_capture_codebook_role"] = "unique_structural_witness_not_source_lut"
        receipt["fp6_capture_policy_sha256"] = _sha(
            args.out_dir / "source_line0_policy.yaml")
    if case.smem_zero_readout:
        receipt["source_baseline"] = source_baseline
        receipt["smem_readout_equivalence_sha256"] = _sha(
            args.out_dir / "smem_readout_equivalence.json")
        if case.dram_mvout_spike_fallback:
            receipt["qualification_scope"] = "SPIKE_SIM_scratchpad_fallback_only"
            receipt["hardware_accumulator_mvout_qualified"] = False
    if case.i_chunks > 1:
        receipt["i_chunks"] = case.i_chunks
        receipt["chunk_equivalence_sha256"] = _sha(
            args.out_dir / "chunk_equivalence.json")
        receipt["source_baseline"] = source_baseline
    if case.native_dram:
        receipt["native_dram_loop"] = True
        receipt["native_dram_equivalence_sha256"] = _sha(
            args.out_dir / "native_dram_equivalence.json")
        receipt["source_baseline"] = source_baseline
        if case.native_dram_scale_mode != "preload":
            receipt["native_dram_scale_mode"] = case.native_dram_scale_mode
        if case.native_dram_store_activation != "none":
            receipt["native_dram_store_activation"] = (
                case.native_dram_store_activation)
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                           sort_keys=True) + "\n")
    print(output)
    if not passed:
        raise SystemExit("compiled Nicolas source did not match its checked-in golden")


if __name__ == "__main__":
    main()
