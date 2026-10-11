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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                   text=True).strip()


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
    required = (f'#include "include/{case.header_name}"', "gemmini_mx_load_scales",
                "gemmini_loop_ws_spad", "gemmini_extended_mvout")
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
        policy.write_text(yaml.safe_dump({
            "schema": "mx_gemmini.quantization_policy.v1",
            "default_format": "mxfp6", "module_overrides": {},
            "functional_overrides": {}, "output_chains": {},
            "fp6_codebooks": {"default": {
                "status": "reviewed",
                "activation": list(fp6.activation_lut_line0),
                "weight": list(fp6.weight_lut_line0)}}}, sort_keys=False))
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
    return f"""#include <stdint.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#include "include/{case.header_name}"
#include "mx_issue.h"
static uint16_t C_hw[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));
static uint8_t scratch_output_scales[2048] __attribute__((aligned(64)));
int main(void) {{
  mx_issue({', '.join(case.call_arguments)});
  gemmini_fence();
  int errors = 0;
  for (int i = 0; i < MATMUL_M; ++i)
    for (int j = 0; j < MATMUL_N; ++j)
      if (C_hw[i][j] != C_out_bf16[i][j]) {{
        if (errors < 8) printf("mismatch %d,%d got %x want %x\\n",
                               i,j,C_hw[i][j],C_out_bf16[i][j]);
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
    if ([slot["name"] for slot in abi] != list(case.buffer_abi) or
            [slot["position"] for slot in abi] != list(range(len(abi))) or
            next(slot for slot in abi if slot["name"] == "scratch_output_scales")[
                "minimum_bytes"] > 2048):
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
    mlir = args.out_dir / "payload_bound.mlir"
    mlir.write_text(bound)
    object_dir = args.out_dir / "object"
    _run([sys.executable, "-m", "tools.compile_object", "--mlir", str(mlir),
          "--bundle", str(bundle), "--profile", str(args.profile.resolve()),
          "--rtl-root", str(rtl_root), "--riscv-root", str(args.riscv_root.resolve()),
          "--mx-opt", str(args.mx_opt.resolve()), "--out-dir", str(object_dir)],
         cwd=ROOT, log=args.out_dir / "object_compile.log")
    mesh_dim = profile["geometry"]["mesh_columns"]
    if profile["geometry"]["mesh_rows"] != mesh_dim:
        raise ValueError("Nicolas source qualifier needs a square MX mesh")
    returncode, output, elf = run_spike(rtl_root, args.riscv_root.resolve(),
                                        object_dir, args.out_dir, case, mesh_dim)
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
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2,
                                                           sort_keys=True) + "\n")
    print(output)
    if not passed:
        raise SystemExit("compiled Nicolas source did not match its checked-in golden")


if __name__ == "__main__":
    main()
