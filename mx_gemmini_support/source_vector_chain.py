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

from .physical_program import _exact_bf16_x2
from .quant_reference import quantize_bf16_fp8_output
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
                                profile: dict) -> tuple[str, dict[str, bytes], dict]:
    """Audit two source commands and independently check their output golden."""
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
    codes, scales = quantize_bf16_fp8_output(_exact_bf16_x2(bf16), m, n)
    if codes != source_codes or any(got != original + 1
                                    for got, original in zip(scales, source_scales)):
        raise ValueError("Nicolas VPU x2/requant reference differs from source golden")
    resources = {"c1_bf16": bf16, "c1_codes_ref": codes,
                 "c1_scales_ref": scales}
    facts = {"shape_mn": [m, n], "sp_bf16": sp_bf16,
             "sp_c1": sp_c1, "bf16_scalar": two,
             "source_sha256": _sha(source_bytes), "header_sha256": _sha(header_bytes),
             "profile_sha256": profile_sha256(profile),
             "resource_sha256": {name: _sha(data) for name, data in resources.items()},
             "source_scope": "BF16 C1 preload, VPU x2, tiled resident FP8 requant; excludes both matmuls"}
    contract = _sha(json.dumps(facts, sort_keys=True, separators=(",", ":")).encode())
    policy = _sha(b"bf16_exact_x2;fp8_e4m3_po2_rne;spad_requant_tiled_resident")
    manifest = _sha(header_bytes)
    binding = (f'contract_sha256 = "{contract}", policy_sha256 = "{policy}", '
               f'manifest_sha256 = "{manifest}", profile_sha256 = "{facts["profile_sha256"]}"')
    mlir = f'''module attributes {{mx.contract_sha256 = "{contract}",
  mx.policy_sha256 = "{policy}", prov.quantization_manifest_sha256 = "{manifest}",
  mx.profile_sha256 = "{facts["profile_sha256"]}"}} {{
  func.func @nicolas_c1_vpu_requant() {{
    "mx_gemmini.vpu_execute"() {{site_id = "nicolas:chain:c1", kind = "muls",
      src1_row = {sp_bf16} : i32, src2_row = 0 : i32, dst_row = {sp_bf16} : i32,
      rows = {m * n // 8} : i32, reduction_length = 1 : i32,
      broadcast = false, immediate_bf16 = {two} : i32, {binding}}} : () -> ()
    "mx_gemmini.spad_requant"() {{site_id = "nicolas:chain:c1",
      source_row = {sp_bf16} : i32, destination_row = {sp_c1} : i32,
      m = {m} : i32, n = {n} : i32, output_format = "fp8_e4m3",
      tiled = true, resident = true, scale_dram_address = 0 : i64,
      scale_buffer = "c1_scales", {binding}}} : () -> ()
    func.return
  }}
}}
'''
    checked = verify_ir(mlir, profile)
    if (checked["contracts"], checked["vpu_commands"], checked["spad_requants"]) != (0, 1, 1):
        raise ValueError("Nicolas vector chain capture has unexpected target operations")
    facts["typed_mlir_sha256"] = _sha(mlir.encode())
    facts["contract_sha256"] = contract
    facts["policy_sha256"] = policy
    return mlir, resources, facts
