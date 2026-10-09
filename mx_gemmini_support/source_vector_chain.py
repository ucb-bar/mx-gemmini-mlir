"""Capture Nicolas's VPU→SPAD_REQUANT seam as profile-bound typed MLIR.

This is a source audit of the checked-in target reference. It starts from the
reference BF16 tile, so it does not claim to compile the preceding matmul or
the following resident matmul. Both are separate completion gates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .quant_reference import exact_bf16_x2, quantize_bf16_fp8_output
from .source_fp6 import _array, _bytes
from .target_profile import profile_sha256
from .verify_profile_ir import verify_ir


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _define(source: str, name: str) -> int:
    match = re.search(rf"^#define {name}\s+(0x[0-9a-fA-F]+|\d+)\b", source,
                      re.MULTILINE)
    if match is None:
        raise ValueError(f"Nicolas vector chain lacks literal {name}")
    return int(match.group(1), 0)


def capture_nicolas_vpu_requant(source_path: Path, header_path: Path,
                                profile: dict, *, include_resident_matmul: bool = False,
                                first_source_path: Path | None = None
                                ) -> tuple[str, dict[str, bytes], dict]:
    """Audit Nicolas's chain and check its source-derived output goldens."""
    source_bytes, header_bytes = source_path.read_bytes(), header_path.read_bytes()
    source, header = source_bytes.decode("ascii"), header_bytes.decode("ascii")
    if (source_path.name != "chain_vpu_spad_requant.c" or
            header_path.name != "matmul_fp8_64x64_chain.h" or
            '"include/matmul_fp8_64x64_chain.h"' not in source):
        raise ValueError("selected Nicolas VPU/requant source pair differs")
    m, n = _define(header, "MATMUL_M"), _define(header, "MATMUL_N")
    if (m, n, _define(header, "MATMUL_GN")) != (64, 64, 2):
        raise ValueError("Nicolas vector chain requires its checked 64x64 header")
    for marker in ("#define M MATMUL_M", "#define N MATMUL_N"):
        if marker not in source:
            raise ValueError("Nicolas vector chain dimensions changed")
    sp_bf16, sp_c1, two = (_define(source, name)
                             for name in ("SP_BF16", "SP_C1", "BF16_TWO"))
    if (sp_bf16, sp_c1, two) != (0x1000, 128, 0x4000):
        raise ValueError("Nicolas vector chain scratchpad placement or scalar changed")
    for marker in (
            "gemmini_vpu_scalar(VPU_MULS, SP_BF16, SP_BF16, BF16_TWO, M * N / 8);",
            "gemmini_spad_requant(SP_C1, SP_BF16, M, N, 1, (uint64_t)c1_scales, 1);"):
        if marker not in source:
            raise ValueError("Nicolas vector chain VPU/requant command changed")
    bf16 = _bytes(_array(header, name="C1_out_bf16", ctype="uint16_t",
                         dimensions="[MATMUL_M][MATMUL_N]", count=m * n,
                         maximum=65535), 2)
    source_codes = bytes(_array(header, name="C1_out", ctype="uint8_t",
                                dimensions="[MATMUL_M][MATMUL_N]", count=m * n,
                                maximum=255))
    source_scales = bytes(_array(header, name="C1_scales_out", ctype="uint8_t",
                                 dimensions="[MATMUL_M][MATMUL_GN]", count=m * n // 32,
                                 maximum=255))
    codes, scales = quantize_bf16_fp8_output(exact_bf16_x2(bf16), m, n)
    if codes != source_codes or any(got != original + 1
                                    for got, original in zip(scales, source_scales)):
        raise ValueError("Nicolas VPU x2/requant reference differs from source golden")
    resources = {"c1_bf16": bf16, "c1_codes_ref": codes,
                 "c1_scales_ref": scales}
    resident_op = ""
    if include_resident_matmul:
        from .resident_lowering import validate_resident_contract
        k = _define(header, "MATMUL_K")
        sp_c2 = _define(source, "SP_C2")
        attrs = {"activation_row": sp_c1,
                 "weight_row": profile["resources"]["scratchpad_bytes"] // 16 - k * n // 16,
                 "output_row": sp_c2, "m": m, "n": n, "k": k,
                 "activation_format": "fp8_e4m3", "weight_format": "fp8_e4m3",
                 "output_format": "fp8_e4m3", "weight_buffer": "b2_weight",
                 "weight_scales_buffer": "b2_scales",
                 "output_scales_buffer": "c2_scales"}
        validate_resident_contract(profile, attrs)
        for marker in (
                "gemmini_mx_load_scales((uint64_t)&B2_scales_col, sizeof(B2_scales_col), 1);",
                "gemmini_config_ld(N * sizeof(uint8_t));",
                "gemmini_extended_mvin((uint8_t *)B2_in + j * DIM * N + k * DIM, b_base + (j * tiles_K + k) * DIM, DIM, DIM);",
                "gemmini_mxquant_config_mvout_resident((uint64_t)c2_scales, tiles_I, tiles_J, tiles_K, 0, 0, 1);",
                "gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K, 0, 0, 0, SP_C1, BANK_NUM * BANK_ROWS, 0, SP_C2,",
                "false, false, false, false, false, NO_ACTIVATION, 0, 0, false, CHAIN_FLAGS);",
                "#define CHAIN_FLAGS (0x38 | LOOP_WS_REQUANT_TILED)"):
            if marker not in source:
                raise ValueError("Nicolas resident second contraction changed")
        b2 = bytes(_array(header, name="B2_in", ctype="uint8_t",
                          dimensions="[MATMUL_K][MATMUL_N]", count=k * n, maximum=255))
        b2_scales = bytes(_array(header, name="B2_scales_col", ctype="uint8_t",
                                 dimensions="[MATMUL_GK][MATMUL_N]", count=k * n // 32,
                                 maximum=255))
        c2_bf16 = _bytes(_array(header, name="C2_out_bf16", ctype="uint16_t",
                                dimensions="[MATMUL_M][MATMUL_N]", count=m * n,
                                maximum=65535), 2)
        c2_source_codes = bytes(_array(header, name="C2_out", ctype="uint8_t",
                                       dimensions="[MATMUL_M][MATMUL_N]", count=m * n,
                                       maximum=255))
        c2_source_scales = bytes(_array(header, name="C2_scales_out", ctype="uint8_t",
                                        dimensions="[MATMUL_M][MATMUL_GN]", count=m * n // 32,
                                        maximum=255))
        c2_codes, c2_scales = quantize_bf16_fp8_output(
            exact_bf16_x2(c2_bf16), m, n)
        if c2_codes != c2_source_codes or c2_scales != bytes(
                min(value + 1, 255) for value in c2_source_scales):
            raise ValueError("Nicolas second contraction golden differs from x2 source")
        resources.update({"b2_weight": b2, "b2_scales": b2_scales,
                          "c2_codes_ref": c2_codes, "c2_scales_ref": c2_scales})
    if first_source_path is not None:
        if not include_resident_matmul or first_source_path.name != "matmul_tiled_fp8_64x64_chain.c":
            raise ValueError("first matrix requires Nicolas's checked FP8 chain source")
        first_bytes = first_source_path.read_bytes()
        first = first_bytes.decode("ascii")
        for marker in (
                "gemmini_mx_load_scales((uint64_t)&A_scales_row, sizeof(A_scales_row), 0);",
                "gemmini_mx_load_scales((uint64_t)&B_scales_col, sizeof(B_scales_col), 1);",
                "gemmini_extended_mvin((void *) dram_ptr, a_base + (i * tiles_K + k) * DIM, DIM, DIM);",
                "gemmini_extended_mvin((void *) dram_ptr, b_base + (j * tiles_K + k) * DIM, DIM, DIM);",
                "gemmini_loop_ws_spad("):
            if marker not in first:
                raise ValueError("Nicolas first contraction source changed")
        resources.update({
            "a1_activation": bytes(_array(header, name="A_in", ctype="uint8_t",
                                           dimensions="[MATMUL_M][MATMUL_K]",
                                           count=m * k, maximum=255)),
            "b1_weight": bytes(_array(header, name="B_in", ctype="uint8_t",
                                       dimensions="[MATMUL_K][MATMUL_N]",
                                       count=k * n, maximum=255)),
            "a1_scales": bytes(_array(header, name="A_scales_row", ctype="uint8_t",
                                        dimensions="[MATMUL_GK][MATMUL_M]",
                                        count=k * m // 32, maximum=255)),
            "b1_scales": bytes(_array(header, name="B_scales_col", ctype="uint8_t",
                                        dimensions="[MATMUL_GK][MATMUL_N]",
                                        count=k * n // 32, maximum=255)),
        })
    facts = {"shape_mn": [m, n], "sp_bf16": sp_bf16,
             "sp_c1": sp_c1, "bf16_scalar": two,
             "source_sha256": _sha(source_bytes), "header_sha256": _sha(header_bytes),
             "profile_sha256": profile_sha256(profile),
             "resource_sha256": {name: _sha(data) for name, data in resources.items()},
             "source_scope": ("first FP8 matrix from model2MLIR site and Nicolas A1/B1, "
                              "VPU x2, resident requant, resident FP8 second matmul"
                              if first_source_path is not None else
                              "BF16 C1 preload, VPU x2, tiled resident FP8 requant, "
                              "resident FP8 second matmul; excludes first matmul"
                              if include_resident_matmul else
                              "BF16 C1 preload, VPU x2, tiled resident FP8 requant; excludes both matmuls")}
    if first_source_path is not None:
        facts["first_source_sha256"] = _sha(first_bytes)
    contract = _sha(json.dumps(facts, sort_keys=True, separators=(",", ":")).encode())
    policy = _sha(b"bf16_exact_x2;fp8_e4m3_po2_rne;spad_requant_tiled_resident")
    manifest = _sha(header_bytes)
    binding = (f'contract_sha256 = "{contract}", policy_sha256 = "{policy}", '
               f'manifest_sha256 = "{manifest}", profile_sha256 = "{facts["profile_sha256"]}"')
    first_site = "functional:matmul" if first_source_path is not None else "nicolas:chain:c1"
    second_site = "functional:matmul_1" if first_source_path is not None else "nicolas:chain:c2"
    if include_resident_matmul:
        resident_op = f'''    "mx_gemmini.resident_contract"() {{site_id = "{second_site}",
      activation_row = {attrs["activation_row"]} : i32,
      weight_row = {attrs["weight_row"]} : i32, output_row = {attrs["output_row"]} : i32,
      m = {m} : i32, n = {n} : i32, k = {k} : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales", output_scales_buffer = "c2_scales",
      {binding}}} : () -> ()
'''
    mlir = f'''module attributes {{mx.contract_sha256 = "{contract}",
  mx.policy_sha256 = "{policy}", prov.quantization_manifest_sha256 = "{manifest}",
  mx.profile_sha256 = "{facts["profile_sha256"]}"}} {{
  func.func @nicolas_c1_vpu_requant() {{
    "mx_gemmini.vpu_execute"() {{site_id = "{first_site}", kind = "muls",
      src1_row = {sp_bf16} : i32, src2_row = 0 : i32, dst_row = {sp_bf16} : i32,
      rows = {m * n // 8} : i32, reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = {two} : i32, {binding}}} : () -> ()
    "mx_gemmini.spad_requant"() {{site_id = "{first_site}",
      source_row = {sp_bf16} : i32, destination_row = {sp_c1} : i32,
      m = {m} : i32, n = {n} : i32, output_format = "fp8_e4m3",
      tiled = true, resident = true, scale_dram_address = 0 : i64,
      scale_buffer = "c1_scales", {binding}}} : () -> ()
{resident_op}    func.return
  }}
}}
'''
    checked = verify_ir(mlir, profile)
    if (checked["contracts"], checked["vpu_commands"], checked["spad_requants"],
            checked["resident_contracts"]) != (0, 1, 1, int(include_resident_matmul)):
        raise ValueError("Nicolas vector chain capture has unexpected target operations")
    facts["typed_mlir_sha256"] = _sha(mlir.encode())
    facts["contract_sha256"] = contract
    facts["policy_sha256"] = policy
    return mlir, resources, facts
