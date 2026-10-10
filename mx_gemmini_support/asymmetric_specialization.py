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
import re

from .target_profile import profile_sha256, require_compute
from .verify_profile_ir import _int_attr, _operation_name, _text_attr, verify_ir


ASYM_CELL = {
    "activation_format": "fp8_e4m3",
    "activation_projection": "lut",
    "weight_format": "fp4_e2m1",
    "weight_projection": "direct",
    "pe_mode": 10,
}
DIRECT_CELL = {**ASYM_CELL, "activation_projection": "direct", "pe_mode": 6}
FP6_FP4_CELL = {**ASYM_CELL, "activation_format": "fp6_e3m2", "pe_mode": 3}
FP4_FP6_CELL = {"activation_format": "fp4_e2m1",
                "activation_projection": "direct",
                "weight_format": "fp6_e3m2",
                "weight_projection": "lut", "pe_mode": 1}
E4M3_E2M3_CELL = {"activation_format": "fp8_e4m3",
                  "activation_projection": "lut",
                  "weight_format": "fp6_e2m3",
                  "weight_projection": "lut", "pe_mode": 9}
E5M2_FP4_CELL = {"activation_format": "fp8_e5m2",
                 "activation_projection": "lut",
                 "weight_format": "fp4_e2m1",
                 "weight_projection": "direct", "pe_mode": 3}

_VARIANTS = {
    "matmul_tiled_asym_e4m3_fp4_64x64.c": {
        "header": "matmul_data_asym_e4m3_fp4.h", "cell": ASYM_CELL,
        "activation_array": "A_in_hw[32][64]", "use_lut": True,
        "lut_words_per_line": 4, "lut_entry_bits": 8,
    },
    "matmul_tiled_asym_e4m3s_fp4_64x64.c": {
        "header": "matmul_data_asym_e4m3s_fp4.h", "cell": DIRECT_CELL,
        "activation_array": "A_in[MATMUL_M][MATMUL_K]", "use_lut": False,
        "lut_words_per_line": 0, "lut_entry_bits": 0,
    },
    "matmul_tiled_asym_fp6_fp4_64x64.c": {
        "header": "matmul_data_asym_fp6_fp4.h", "cell": FP6_FP4_CELL,
        "activation_array": "A_in_hw[32][64]", "use_lut": True,
        "lut_words_per_line": 3, "lut_entry_bits": 6,
    },
    "matmul_tiled_asym_fp4_fp6_64x64.c": {
        "header": "matmul_data_asym_fp4_fp6.h", "cell": FP4_FP6_CELL,
        "activation_array": "A_in_hw[32][64]", "use_lut": True,
        "lut_words_per_line": 3, "lut_entry_bits": 6,
    },
    "matmul_tiled_asym_e4m3_e2m3_64x64.c": {
        "header": "matmul_data_asym_e4m3_e2m3.h", "cell": E4M3_E2M3_CELL,
        "activation_array": "A_in_hw[32][64]", "use_lut": True,
        "lut_words_per_line": 4, "lut_entry_bits": 8,
    },
    "matmul_tiled_asym_e5m2_fp4_64x64.c": {
        "header": "matmul_data_asym_e5m2_fp4.h", "cell": E5M2_FP4_CELL,
        "activation_array": "A_in_hw[32][64]", "use_lut": True,
        "lut_words_per_line": 4, "lut_entry_bits": 8,
    },
}
_VARIANTS_BY_HEADER = {variant["header"]: variant for variant in _VARIANTS.values()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_recipe(source: Path, header: Path, profile: dict) -> dict:
    """Bind one checked-in Nicolas test and its data to a legal profile mode."""
    variant = _VARIANTS.get(source.name)
    if variant is None or header.name != variant["header"]:
        raise ValueError("selected asymmetric specialization needs a named Nicolas 64-cubed test")
    if not source.is_file() or not header.is_file():
        raise ValueError("selected asymmetric source or data header is absent")
    cell = variant["cell"]
    require_compute(profile, cell["activation_format"], cell["weight_format"],
                    pe_mode=cell["pe_mode"],
                    activation_projection=cell["activation_projection"],
                    weight_projection=cell["weight_projection"])
    if (profile["geometry"] != {"mesh_rows": 16, "mesh_columns": 16,
                                "tile_rows": 1, "tile_columns": 1} or
            profile["resources"]["scratchpad_bytes"] != 262144 or
            profile["resources"]["lut_config"]["activation_code_bits"] !=
            (variant["lut_entry_bits"] or 8) or
            profile["resources"]["lut_config"]["address_bits"] != 4):
        raise ValueError("selected asymmetric source requires DIM16, 256 KiB SPAD, 4-bit LUT indices")
    source_text, header_text = source.read_text(), header.read_text()
    for marker in (f'#include "include/{header.name}"',
                   f'#define USE_LUT {int(variant["use_lut"])}'):
        if marker not in source_text:
            raise ValueError(f"Nicolas source command contract changed: {marker}")
    if not re.search(r"gemmini_loop_ws_spad\(\s*tiles_I,\s*tiles_J,\s*tiles_K", source_text):
        raise ValueError("Nicolas source loop schedule changed")
    for marker in ("#define MATMUL_M   64", "#define MATMUL_K   64",
                   "#define MATMUL_N   64", variant["activation_array"],
                   "B_in[MATMUL_K][MATMUL_N / 2]",
                   "A_scales_row[MATMUL_GK][MATMUL_M]",
                   "B_scales_col[MATMUL_GK][MATMUL_N]",
                   "C_out_bf16[MATMUL_M][MATMUL_N]"):
        if marker not in header_text:
            raise ValueError(f"Nicolas source data contract changed: {marker}")
    if cell["activation_format"] in {"fp8_e4m3", "fp8_e5m2"}:
        altfmt = int(cell["activation_format"] == "fp8_e5m2")
        if f"#define MX_ALTFMT {altfmt}" not in source_text:
            raise ValueError("Nicolas FP8 source alternate format selection changed")
    if cell["weight_format"] == "fp6_e2m3":
        for marker in ("((uint64_t)(1) << 31)", "((uint64_t)(1) << 12)"):
            if marker not in source_text:
                raise ValueError(f"Nicolas E2M3 weight encoding changed: {marker}")
    if variant["use_lut"] and f'A_lut[32][{variant["lut_words_per_line"]}]' not in header_text:
        raise ValueError("Nicolas source activation LUT changed")
    return {"schema": "mx_gemmini.asymmetric_source_recipe.v1",
            "site_id": "functional:matmul", "shape": [64, 64, 64],
            "frontend_capture_format": "mxfp8",
            "source_driver_sha256": sha256(source),
            "source_header_sha256": sha256(header),
            "profile_sha256": profile_sha256(profile), "compute": cell.copy()}


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
            recipe.get("compute") not in (ASYM_CELL, DIRECT_CELL, FP6_FP4_CELL,
                                          FP4_FP6_CELL, E4M3_E2M3_CELL,
                                          E5M2_FP4_CELL) or
            any(not isinstance(recipe.get(key), str) or len(recipe[key]) != 64 or
                any(ch not in "0123456789abcdef" for ch in recipe[key])
                for key in ("source_driver_sha256", "source_header_sha256"))):
        raise ValueError("asymmetric recipe does not identify the selected source and profile")
    cell = recipe["compute"]
    require_compute(profile, cell["activation_format"], cell["weight_format"],
                    pe_mode=cell["pe_mode"],
                    activation_projection=cell["activation_projection"],
                    weight_projection=cell["weight_projection"])
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
    for key, value in cell.items():
        contract.attributes[key] = IntegerAttr(value, 32) if key == "pe_mode" else StringAttr(value)
    for op in operations:
        op.attributes["profile_sha256"] = StringAttr(digest)
    stream = StringIO()
    Printer(stream=stream).print_op(module)
    result = stream.getvalue() + "\n"
    verify_ir(result, profile)
    return result


def _validate_bound_site(mlir_text: str, profile: dict, recipe: dict, *,
                         source: Path, header: Path) -> dict:
    """Check the typed site, exact profile mode, and named source data."""
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
    cell = recipe["compute"]
    if ([_operation_name(op) for op in operations] !=
            ["mx_gemmini.contract", "mx_gemmini.readout_bf16"] or
            any(_text_attr(op, "site_id") != recipe["site_id"] for op in operations) or
            any(_text_attr(operations[0], key) != value for key, value in cell.items()
                if key != "pe_mode") or
            _int_attr(operations[0], "pe_mode") != cell["pe_mode"]):
        raise ValueError("bound asymmetric MLIR compute tuple or site differs from recipe")
    return cell


def emit_baremetal(mlir_text: str, profile: dict, recipe: dict, *, source: Path,
                   header: Path) -> str:
    """Emit the bounded historical C diagnostic from a verified contract."""
    cell = _validate_bound_site(mlir_text, profile, recipe, source=source, header=header)
    if cell["activation_format"] != "fp8_e4m3":
        raise ValueError("historical asymmetric C diagnostic only supports E4M3 activation")
    use_lut = cell["activation_projection"] == "lut"
    activation_array = "A_in_hw" if use_lut else "A_in"
    tile_rows = 32 if use_lut else 16
    lut_setup = (
        "  gemmini_mx_load_lut_dt((uint64_t)B_lut, MATMUL_N / 2, 0, 8);\n"
        "  gemmini_mx_load_lut_dt((uint64_t)A_lut, MATMUL_M / 2, 1, 8);\n"
        "  gemmini_mx_load_lut_dt((uint64_t)C_lut, MATMUL_M / 2, 2, 8);\n"
        if use_lut else "  gemmini_mx_lut_disable();\n"
    )
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
  const int tiles_i = MATMUL_M / {tile_rows};
  const int tiles_j = MATMUL_N / 32;
  const int tiles_k = MATMUL_K / DIM;
  const uint32_t a_base = 0;
  const uint32_t b_end = BANK_NUM * BANK_ROWS / 2;
  const uint32_t b_base = b_end - tiles_k * tiles_j * DIM;
  const uint32_t c_base = 128;
  memset(C_hw, 0, sizeof C_hw);
  gemmini_flush(0);
  // E4M3 activation ({cell["activation_projection"]}), direct FP4 weight, BF16 output.
  gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY,
                              1, 1, 0, 0, false, 0, 2, 3, {str(use_lut).lower()});
{lut_setup}  gemmini_mx_load_scales((uint64_t)A_scales_row, sizeof A_scales_row, 0);
  gemmini_mx_load_scales((uint64_t)B_scales_col, sizeof B_scales_col, 1);
  gemmini_fence();
  gemmini_config_ld(MATMUL_K);
  for (int i = 0; i < tiles_i; ++i)
    for (int k = 0; k < tiles_k; ++k)
      gemmini_extended_mvin((void *)&{activation_array}[i * DIM][k * DIM],
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


def read_asymmetric_resources(header: Path, recipe: dict) -> dict[str, bytes]:
    """Export Nicolas's packed input and BF16 golden arrays as binary resources."""
    from .source_fp6 import _array, _bytes

    if sha256(header) != recipe.get("source_header_sha256"):
        raise ValueError("asymmetric source header differs from recipe")
    text = header.read_text(encoding="ascii")
    cell = recipe["compute"]
    lut = cell["activation_projection"] == "lut" or cell["weight_projection"] == "lut"
    packed_activation = cell["activation_format"] != "fp8_e4m3" or lut
    variant = _VARIANTS_BY_HEADER[header.name]
    lut_words = variant["lut_words_per_line"]
    a_name = "A_in_hw" if packed_activation else "A_in"
    a_shape = "[32][64]" if packed_activation else "[MATMUL_M][MATMUL_K]"
    resources = {
        "activation": bytes(_array(text, name=a_name, ctype="uint8_t",
                                    dimensions=a_shape, count=2048 if packed_activation else 4096,
                                    maximum=255)),
        "weight": bytes(_array(text, name="B_in", ctype="uint8_t",
                                dimensions="[MATMUL_K][MATMUL_N / 2]",
                                count=2048, maximum=255)),
        "activation_scales": bytes(_array(text, name="A_scales_row", ctype="uint8_t",
                                            dimensions="[MATMUL_GK][MATMUL_M]",
                                            count=128, maximum=255)),
        "weight_scales": bytes(_array(text, name="B_scales_col", ctype="uint8_t",
                                        dimensions="[MATMUL_GK][MATMUL_N]",
                                        count=128, maximum=255)),
        "golden_bf16": _bytes(_array(text, name="C_out_bf16", ctype="uint16_t",
                                      dimensions="[MATMUL_M][MATMUL_N]",
                                      count=4096, maximum=0xffff), 2),
    }
    if lut:
        for c_name, resource in (("A_lut", "activation_lut"),
                                 ("B_lut", "weight_lut"),
                                 ("C_lut", "output_lut")):
            resources[resource] = _bytes(_array(text, name=c_name, ctype="uint32_t",
                                               dimensions=f"[32][{lut_words}]", count=32 * lut_words,
                                               maximum=0xffffffff), 4)
    return resources


def _resource_manifest(recipe: dict, resources: dict[str, bytes], header: Path) -> dict:
    """Describe every Nicolas source byte array with its physical shape."""
    from .source_payload import Resource

    variant = _VARIANTS_BY_HEADER[header.name]
    packed_activation = variant["activation_array"] == "A_in_hw[32][64]"
    shapes = {
        "activation": ((32, 64) if packed_activation else (64, 64), 8,
                       "packed_even_odd_m_nibbles" if packed_activation else "row_major_codes"),
        "weight": ((64, 32), 8, "packed_even_odd_n_nibbles"),
        "activation_scales": ((2, 64), 8, "k_group_row_e8m0"),
        "weight_scales": ((2, 64), 8, "k_group_column_e8m0"),
        "golden_bf16": ((64, 64), 16, "row_major_bf16"),
    }
    if variant["use_lut"]:
        words = variant["lut_words_per_line"]
        bits = variant["lut_entry_bits"]
        shapes.update({
            "activation_lut": ((32, words), 32, f"row_pair_lut_{bits}bit"),
            "weight_lut": ((32, words), 32, f"column_pair_lut_{bits}bit"),
            "output_lut": ((32, words), 32, f"output_pair_lut_{bits}bit"),
        })
    if set(resources) != set(shapes):
        raise ValueError("asymmetric source resource set differs from selected mode")
    return {
        "schema": "mx_gemmini.asymmetric_resource_manifest.v2",
        "site_id": recipe["site_id"],
        "profile_sha256": recipe["profile_sha256"],
        "origin": "nicolas_source_header_specialization",
        "recipe": recipe,
        "resources": {
            name: Resource(resources[name], *shapes[name]).descriptor(name)
            for name in sorted(resources)
        },
    }


def bind_asymmetric_payload(mlir_text: str, profile: dict, recipe: dict, *,
                            source: Path, header: Path) -> str:
    """Bind source operand, scale, LUT, and golden bytes to the typed MLIR site."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin, StringAttr
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser
    from xdsl.printer import Printer

    _validate_bound_site(mlir_text, profile, recipe, source=source, header=header)
    resources = read_asymmetric_resources(header, recipe)
    from .source_payload import manifest_sha256
    manifest = _resource_manifest(recipe, resources, header)
    digest = manifest_sha256(manifest)
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    if "mx.payload_manifest_sha256" in module.attributes:
        raise ValueError("asymmetric MLIR is already payload-bound")
    contract = next(op for op in module.walk()
                    if _operation_name(op) == "mx_gemmini.contract")
    module.attributes["mx.payload_manifest_sha256"] = StringAttr(digest)
    contract.attributes["payload_manifest_sha256"] = StringAttr(digest)
    contract.attributes["payload_origin"] = StringAttr(manifest["origin"])
    output = StringIO()
    Printer(stream=output).print_op(module)
    bound = output.getvalue() + "\n"
    verify_ir(bound, profile)
    return bound


def lower_asymmetric_physical(mlir_text: str, profile: dict, recipe: dict, *,
                              source: Path, header: Path):
    """Lower a source-bound DIM16 asymmetric mode to physical MX commands."""
    from .command_ir import Fence, Operand
    from .physical_program import (PhysicalProgram, PhysicalStep, _cmd, _config_ld,
                                   _config_st, _transfer)

    if recipe != source_recipe(source, header, profile):
        raise ValueError("asymmetric physical lowering source or target changed")
    _validate_bound_site(mlir_text, profile, recipe, source=source, header=header)
    resources = read_asymmetric_resources(header, recipe)
    from .source_payload import manifest_sha256
    resource_manifest = _resource_manifest(recipe, resources, header)
    payload_digest = manifest_sha256(resource_manifest)
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    contracts = [op for op in module.walk() if _operation_name(op) == "mx_gemmini.contract"]
    if (_text_attr(module, "mx.payload_manifest_sha256") != payload_digest or
            len(contracts) != 1 or
            _text_attr(contracts[0], "payload_manifest_sha256") != payload_digest or
            _text_attr(contracts[0], "payload_origin") != resource_manifest["origin"]):
        raise ValueError("asymmetric physical lowering payload differs from typed MLIR")
    cell = recipe["compute"]
    use_lut = cell["activation_projection"] == "lut" or cell["weight_projection"] == "lut"
    packed_activation = cell["activation_format"] != "fp8_e4m3" or use_lut
    ti, tj, tki = (2 if packed_activation else 4), 2, 4
    dim = 16
    a_base, b_end, c_base = 0, 8192, 128
    b_base = b_end - tki * tj * dim
    steps: list[PhysicalStep] = []

    def issue(phase: str, command) -> None:
        steps.append(PhysicalStep(phase, None, command))

    issue("configure", _cmd(7, 0, 0))
    activation_code = {"fp8_e4m3": 0, "fp8_e5m2": 0,
                       "fp6_e3m2": 1, "fp4_e2m1": 2}[cell["activation_format"]]
    weight_code = {"fp6_e3m2": 1, "fp6_e2m3": 1,
                   "fp4_e2m1": 2}[cell["weight_format"]]
    variant = _VARIANTS[source.name]
    lut_entry_bits = variant["lut_entry_bits"]
    activation_altfmt = int(cell["activation_format"] == "fp8_e5m2")
    weight_altfmt_diff = int(cell["weight_format"] == "fp6_e2m3")
    config_ex = (1 << 16) | (weight_code << 12) | (activation_code << 10) | \
                (3 << 14) | (int(use_lut) << 5) | (1 << 2) | \
                (activation_altfmt << 6) | (weight_altfmt_diff << 31)
    issue("configure", _cmd(0, config_ex, 1 << 48))
    if use_lut:
        for resource, selector in (("weight_lut", 0), ("activation_lut", 1),
                                   ("output_lut", 2)):
            issue("upload_lut", _cmd(29, Operand(buffer=resource),
                                     (lut_entry_bits << 34) | (selector << 32) | 32))
    else:
        issue("disable_lut", _cmd(30, 0, 0))
    issue("upload_scales", _cmd(27, Operand(buffer="activation_scales"), 128))
    issue("upload_scales", _cmd(27, Operand(buffer="weight_scales"), (1 << 32) | 128))
    issue("upload_scales", Fence())
    issue("move_activation", _config_ld(64))
    for i in range(ti):
        for k in range(tki):
            offset = i * dim * 64 + k * dim
            row = a_base + (i * tki + k) * dim
            issue("move_activation", _transfer(2, "activation", offset, row))
    issue("move_weight", _config_ld(32))
    for k in range(tki):
        for j in range(tj):
            offset = k * dim * 32 + j * dim
            row = b_base + (k * tj + j) * dim
            issue("move_weight", _transfer(2, "weight", offset, row))
    issue("move_weight", Fence())
    issue("configure", _config_st(128))
    selector_bits = (tki << 51) | (tj << 42) | (ti << 33)
    issue("select_scales", _cmd(26, Operand(buffer="scratch_output_scales",
                                            address_mask=(1 << 33) - 1,
                                            or_bits=selector_bits), 1))
    issue("compute", _cmd(9, 0, (tki << 32) | (tj << 16) | ti))
    issue("compute", _cmd(24, a_base, b_end))
    issue("compute", _cmd(8, 0, (c_base << 32) | 0x200 | 0x38))
    issue("compute", Fence())
    issue("readout", _config_st(dim))
    for row in range(0, 512, dim):
        issue("readout", _transfer(3, "output_bf16", row * dim, c_base + row))
    issue("readout", Fence())
    plan = {"shape_mnk": [64, 64, 64], "tile_mnk": [64, 64, 64],
            "activation_projection": recipe["compute"]["activation_projection"],
            "weight_projection": recipe["compute"]["weight_projection"],
            "pe_mode": recipe["compute"]["pe_mode"],
            "scratchpad_rows": profile["resources"]["scratchpad_bytes"] // dim,
            "a_row": a_base, "b_row": b_base, "c_row": c_base,
            "tiles_i": ti, "tiles_j": tj, "tiles_k": tki}
    return PhysicalProgram(profile_sha256(profile), payload_digest,
                           "spike_serial", (64, 64, 64), plan, tuple(steps)), resources, resource_manifest
