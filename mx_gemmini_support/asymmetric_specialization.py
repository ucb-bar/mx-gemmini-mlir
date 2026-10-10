"""Explicit source specialization of a captured matmul for asymmetric MX.

The current model2MLIR MX policy captures the matmul site but describes
symmetric quantization. A recipe naming Nicolas's packed source operands is
therefore required before changing either operand's format or projection.
"""

from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path

from .target_profile import profile_sha256, require_compute
from .verify_profile_ir import _int_attr, _operation_name, _text_attr, verify_ir


ASYM_CELL = {
    "activation_format": "fp8_e4m3",
    "activation_projection": "lut",
    "weight_format": "fp4_e2m1",
    "weight_projection": "direct",
    "pe_mode": 10,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_recipe(source: Path, header: Path, profile: dict) -> dict:
    """Bind one checked-in Nicolas test and its data to a legal profile mode."""
    if source.name != "matmul_tiled_asym_e4m3_fp4_64x64.c" or \
            header.name != "matmul_data_asym_e4m3_fp4.h":
        raise ValueError("selected asymmetric specialization needs Nicolas's 64-cubed test")
    if not source.is_file() or not header.is_file():
        raise ValueError("selected asymmetric source or data header is absent")
    require_compute(profile, ASYM_CELL["activation_format"], ASYM_CELL["weight_format"],
                    pe_mode=ASYM_CELL["pe_mode"],
                    activation_projection=ASYM_CELL["activation_projection"],
                    weight_projection=ASYM_CELL["weight_projection"])
    if (profile["geometry"] != {"mesh_rows": 16, "mesh_columns": 16,
                                "tile_rows": 1, "tile_columns": 1} or
            profile["resources"]["scratchpad_bytes"] != 262144 or
            profile["resources"]["lut_config"]["activation_code_bits"] != 8 or
            profile["resources"]["lut_config"]["address_bits"] != 4):
        raise ValueError("selected asymmetric source requires DIM16, 256 KiB SPAD, 4-bit LUT indices")
    source_text, header_text = source.read_text(), header.read_text()
    for marker in ("#include \"include/matmul_data_asym_e4m3_fp4.h\"",
                   "#define USE_LUT 1", "#define MX_ALTFMT 0",
                   "gemmini_loop_ws_spad(tiles_I, tiles_J, tiles_K"):
        if marker not in source_text:
            raise ValueError(f"Nicolas source command contract changed: {marker}")
    for marker in ("#define MATMUL_M   64", "#define MATMUL_K   64",
                   "#define MATMUL_N   64", "A_in_hw[32][64]",
                   "B_in[MATMUL_K][MATMUL_N / 2]", "A_lut[32][4]",
                   "A_scales_row[MATMUL_GK][MATMUL_M]",
                   "B_scales_col[MATMUL_GK][MATMUL_N]",
                   "C_out_bf16[MATMUL_M][MATMUL_N]"):
        if marker not in header_text:
            raise ValueError(f"Nicolas source data contract changed: {marker}")
    return {"schema": "mx_gemmini.asymmetric_source_recipe.v1",
            "site_id": "functional:matmul", "shape": [64, 64, 64],
            "frontend_capture_format": "mxfp8",
            "source_driver_sha256": sha256(source),
            "source_header_sha256": sha256(header),
            "profile_sha256": profile_sha256(profile), "compute": ASYM_CELL.copy()}


def specialize_handoff(mlir_text: str, profile: dict, recipe: dict) -> str:
    """Change a captured site's MX semantics only with an explicit source recipe."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin, IntegerAttr, StringAttr
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser
    from xdsl.printer import Printer

    if (recipe.get("schema") != "mx_gemmini.asymmetric_source_recipe.v1" or
            recipe.get("site_id") != "functional:matmul" or
            recipe.get("shape") != [64, 64, 64] or
            recipe.get("frontend_capture_format") != "mxfp8" or
            recipe.get("profile_sha256") != profile_sha256(profile) or
            recipe.get("compute") != ASYM_CELL or
            any(not isinstance(recipe.get(key), str) or len(recipe[key]) != 64 or
                any(ch not in "0123456789abcdef" for ch in recipe[key])
                for key in ("source_driver_sha256", "source_header_sha256"))):
        raise ValueError("asymmetric recipe does not identify the selected source and profile")
    require_compute(profile, ASYM_CELL["activation_format"], ASYM_CELL["weight_format"],
                    pe_mode=ASYM_CELL["pe_mode"],
                    activation_projection=ASYM_CELL["activation_projection"],
                    weight_projection=ASYM_CELL["weight_projection"])
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    if "mx.profile_sha256" in module.attributes:
        raise ValueError("asymmetric specialization requires an unbound frontend handoff")
    operations = [op for op in module.walk() if _operation_name(op).startswith("mx_gemmini.")]
    if [_operation_name(op) for op in operations] != ["mx_gemmini.contract", "mx_gemmini.readout_bf16"]:
        raise ValueError("asymmetric specialization needs one captured BF16 matmul site")
    contract, readout = operations
    if (_text_attr(contract, "site_id") != recipe["site_id"] or
            _text_attr(readout, "site_id") != recipe["site_id"] or
            _text_attr(contract, "format") != recipe["frontend_capture_format"]):
        raise ValueError("captured matmul site or frontend format differs from recipe")
    digest = profile_sha256(profile)
    recipe_digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    module.attributes["mx.profile_sha256"] = StringAttr(digest)
    module.attributes["mx.asymmetric_recipe_sha256"] = StringAttr(recipe_digest)
    module.attributes["mx.source_driver_sha256"] = StringAttr(recipe["source_driver_sha256"])
    module.attributes["mx.source_header_sha256"] = StringAttr(recipe["source_header_sha256"])
    del contract.attributes["format"]
    for key, value in ASYM_CELL.items():
        contract.attributes[key] = IntegerAttr(value, 32) if key == "pe_mode" else StringAttr(value)
    for op in operations:
        op.attributes["profile_sha256"] = StringAttr(digest)
    stream = StringIO()
    Printer(stream=stream).print_op(module)
    result = stream.getvalue() + "\n"
    verify_ir(result, profile)
    return result


def emit_baremetal(mlir_text: str, profile: dict, recipe: dict, *, source: Path,
                   header: Path) -> str:
    """Emit the bounded physical schedule from a verified asymmetric contract."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    if recipe != source_recipe(source, header, profile):
        raise ValueError("asymmetric source bytes, profile, or recipe changed")
    checked = verify_ir(mlir_text, profile)
    if checked["contracts"] != 1 or checked["encodes"] or checked["requantizes"]:
        raise ValueError("physical asymmetric schedule needs one BF16 contract")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    recipe_digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (_text_attr(module, "mx.asymmetric_recipe_sha256") != recipe_digest or
            _text_attr(module, "mx.source_driver_sha256") != recipe["source_driver_sha256"] or
            _text_attr(module, "mx.source_header_sha256") != recipe["source_header_sha256"]):
        raise ValueError("profile-bound MLIR lacks matching asymmetric source provenance")
    operations = [op for op in module.walk() if _operation_name(op).startswith("mx_gemmini.")]
    if ([_operation_name(op) for op in operations] !=
            ["mx_gemmini.contract", "mx_gemmini.readout_bf16"] or
            any(_text_attr(op, "site_id") != recipe["site_id"] for op in operations) or
            any(_text_attr(operations[0], key) != value for key, value in ASYM_CELL.items()
                if key != "pe_mode") or
            _int_attr(operations[0], "pe_mode") != ASYM_CELL["pe_mode"]):
        raise ValueError("bound asymmetric MLIR compute tuple or site differs from recipe")
    return f'''// Generated from model2MLIR site {recipe["site_id"]} and explicit source recipe.
// Source driver SHA-256: {recipe["source_driver_sha256"]}
// Source header SHA-256: {recipe["source_header_sha256"]}
// MX profile SHA-256: {recipe["profile_sha256"]}
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "include/gemmini_testutils.h"
#include "include/{header.name}"

_Static_assert(DIM == 16 && BANK_NUM == 4 && BANK_ROWS == 4096,
               "selected asymmetric MX software geometry changed");
static uint16_t C_hw[MATMUL_M][MATMUL_N] __attribute__((aligned(64)));
static uint32_t output_scales[512] __attribute__((aligned(64)));

int main(void) {{
  const int tiles_i = MATMUL_M / 32;
  const int tiles_j = MATMUL_N / 32;
  const int tiles_k = MATMUL_K / DIM;
  const uint32_t a_base = 0;
  const uint32_t b_end = BANK_NUM * BANK_ROWS / 2;
  const uint32_t b_base = b_end - tiles_k * tiles_j * DIM;
  const uint32_t c_base = 128;
  memset(C_hw, 0, sizeof C_hw);
  gemmini_flush(0);
  // E4M3 activation via 4-bit LUT index, direct FP4 weight, BF16 output.
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
                              1, 1, 0, 0, false, 0, 2, 3, true);
  gemmini_mx_load_lut_dt((uint64_t)B_lut, MATMUL_N / 2, 0, 8);
  gemmini_mx_load_lut_dt((uint64_t)A_lut, MATMUL_M / 2, 1, 8);
  gemmini_mx_load_lut_dt((uint64_t)C_lut, MATMUL_M / 2, 2, 8);
  gemmini_mx_load_scales((uint64_t)A_scales_row, sizeof A_scales_row, 0);
  gemmini_mx_load_scales((uint64_t)B_scales_col, sizeof B_scales_col, 1);
  gemmini_fence();
  gemmini_config_ld(MATMUL_K);
  for (int i = 0; i < tiles_i; ++i)
    for (int k = 0; k < tiles_k; ++k)
      gemmini_extended_mvin((void *)&A_in_hw[i * DIM][k * DIM],
                            a_base + (i * tiles_k + k) * DIM, DIM, DIM);
  gemmini_config_ld(MATMUL_N / 2);
  for (int k = 0; k < tiles_k; ++k)
    for (int j = 0; j < tiles_j; ++j)
      gemmini_extended_mvin((void *)&B_in[k * DIM][j * DIM],
                            b_base + (k * tiles_j + j) * DIM, DIM, DIM);
  gemmini_fence();
  gemmini_config_st(MATMUL_N * sizeof(uint16_t));
  gemmini_mxquant_config_mvout((uint64_t)output_scales,
                                tiles_i, tiles_j, tiles_k, 0, 0, 1);
  gemmini_loop_ws_spad(tiles_i, tiles_j, tiles_k,
      0, 0, 0, a_base, b_end, 0, c_base,
      false, false, false, false, false, NO_ACTIVATION,
      0, 0, false, 0x38);
  gemmini_fence();
  gemmini_config_st(DIM);
  uint8_t *dst = (uint8_t *)C_hw;
  for (int row = 0; row < MATMUL_M * MATMUL_N * 2 / DIM; row += DIM)
    gemmini_extended_mvout(dst + row * DIM, c_base + row, DIM, DIM);
  gemmini_fence();
  int errors = 0;
  for (int i = 0; i < MATMUL_M; ++i)
    for (int j = 0; j < MATMUL_N; ++j)
      if (C_hw[i][j] != C_out_bf16[i][j]) {{
        if (errors < 8)
          printf("MISMATCH (%d,%d): got=0x%04x expected=0x%04x\\n",
                 i, j, C_hw[i][j], C_out_bf16[i][j]);
        ++errors;
      }}
  printf("lowered asymmetric E4M3xFP4 64x64x64: %d BF16 mismatches\\n", errors);
  return errors != 0;
}}
'''
