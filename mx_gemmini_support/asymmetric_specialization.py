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
E4M3_DIRECT_E3M2_CELL = {"activation_format": "fp8_e4m3",
                         "activation_projection": "direct",
                         "weight_format": "fp6_e3m2",
                         "weight_projection": "lut", "pe_mode": 7}
FP4_DIRECT_E4M3_CELL = {"activation_format": "fp4_e2m1",
                        "activation_projection": "direct",
                        "weight_format": "fp8_e4m3",
                        "weight_projection": "direct", "pe_mode": 2}
E2M3_E5M2_CELL = {"activation_format": "fp6_e2m3",
                  "activation_projection": "lut",
                  "weight_format": "fp8_e5m2",
                  "weight_projection": "lut", "pe_mode": 10}
E4M3_DIRECT_E5M2_CELL = {"activation_format": "fp8_e4m3",
                         "activation_projection": "direct",
                         "weight_format": "fp8_e5m2",
                         "weight_projection": "lut", "pe_mode": 7}
E2M3_E2M3_CELL = {"activation_format": "fp6_e2m3",
                  "activation_projection": "lut",
                  "weight_format": "fp6_e2m3",
                  "weight_projection": "lut", "pe_mode": 9}
E4M3_E4M3_CELL = {"activation_format": "fp8_e4m3",
                  "activation_projection": "lut",
                  "weight_format": "fp8_e4m3",
                  "weight_projection": "lut", "pe_mode": 9}
E5M2_E5M2_CELL = {"activation_format": "fp8_e5m2",
                  "activation_projection": "lut",
                  "weight_format": "fp8_e5m2",
                  "weight_projection": "lut", "pe_mode": 4}
FP4_FP4_CELL = {"activation_format": "fp4_e2m1",
                "activation_projection": "direct",
                "weight_format": "fp4_e2m1",
                "weight_projection": "direct", "pe_mode": 0}

_VARIANTS = {
    "matmul_tiled_fp4_64x64.c": {
        "header": "matmul_fp4_64x64.h", "cell": FP4_FP4_CELL,
        "activation_array": "A_in_hw[32][64]", "use_lut": False,
        "lut_words_per_line": 0, "lut_entry_bits": 0,
        "weight_decl": "B_in[64][32]",
        "activation_scales_decl": "A_scales_row[2][64]",
        "weight_scales_decl": "B_scales_col[2][64]",
        "golden_decl": "C_out_bf16[64][64]",
        "source_config_marker": ("gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, "
                                 "ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 2, 2, 3, 0)"),
    },
    "matmul_tiled_fp6_e2m3_lut_64x64.c": {
        "header": "matmul_data_mx_lut_e2m3_64x64.h",
        "cell": E2M3_E2M3_CELL,
        "activation_array": "A_in_hw[32][64]", "use_lut": True,
        "lut_words_per_line": 3, "lut_entry_bits": 6,
    },
    "matmul_tiled_fp8_e4m3_lut_64x64.c": {
        "header": "matmul_data_mx_lut_e4m3_64x64.h",
        "cell": E4M3_E4M3_CELL,
        "activation_array": "A_in_hw[32][64]", "use_lut": True,
        "lut_words_per_line": 4, "lut_entry_bits": 8,
    },
    "matmul_tiled_fp8_e5m2_64x64.c": {
        "header": "matmul_data_mx_lut_e5m2_64x64.h",
        "cell": E5M2_E5M2_CELL,
        "activation_array": "A_in_hw[32][64]", "use_lut": True,
        "lut_words_per_line": 4, "lut_entry_bits": 8,
    },
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
    "matmul_tiled_asym_e4m3s_fp4_128x128x256_dim32.c": {
        "header": "matmul_data_asym_e4m3s_fp4_128x128x256_dim32.h",
        "cell": DIRECT_CELL, "mesh_dim": 32, "shape": [128, 128, 256],
        "activation_array": "A_in[MATMUL_M][MATMUL_K]", "use_lut": False,
        "lut_words_per_line": 0, "lut_entry_bits": 0,
    },
    "matmul_tiled_asym_e4m3s_fp4_128x128.c": {
        "header": "matmul_data_asym_e4m3s_fp4_128x128.h",
        "cell": DIRECT_CELL, "mesh_dim": 16, "shape": [128, 128, 128],
        "activation_array": "A_in[MATMUL_M][MATMUL_K]", "use_lut": False,
        "lut_words_per_line": 0, "lut_entry_bits": 0,
    },
    "matmul_tiled_asym_e2m3_e5m2_128x128x256_dim32.c": {
        "header": "matmul_data_asym_e2m3_e5m2_128x128x256_dim32.h",
        "cell": E2M3_E5M2_CELL, "mesh_dim": 32, "shape": [128, 128, 256],
        "activation_array": "A_in_hw[64][256]", "use_lut": True,
        "lut_words_per_line": 4, "lut_entry_bits": 8, "lut_lines": 64,
    },
    "matmul_tiled_asym_e4m3s_e5m2_128x128x256_dim32.c": {
        "header": "matmul_data_asym_e4m3s_e5m2_128x128x256_dim32.h",
        "cell": E4M3_DIRECT_E5M2_CELL, "mesh_dim": 32, "shape": [128, 128, 256],
        "activation_array": "A_in[MATMUL_M][MATMUL_K]", "use_lut": True,
        "lut_arrays": ("B_lut",), "lut_words_per_line": 4,
        "lut_entry_bits": 8, "lut_lines": 64,
    },
    "matmul_tiled_asym_fp4_fp6_128x128x256_dim32.c": {
        "header": "matmul_data_asym_fp4_fp6_128x128x256_dim32.h",
        "cell": FP4_FP6_CELL, "mesh_dim": 32, "shape": [128, 128, 256],
        "activation_array": "A_in_hw[64][256]", "use_lut": True,
        "lut_words_per_line": 3, "lut_entry_bits": 6, "lut_lines": 64,
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
    "matmul_tiled_asym_e4m3s_e3m2_64x64.c": {
        "header": "matmul_data_asym_e4m3s_e3m2.h",
        "cell": E4M3_DIRECT_E3M2_CELL,
        "activation_array": "A_in[MATMUL_M][MATMUL_K]", "use_lut": True,
        "lut_arrays": ("B_lut",),
        "lut_words_per_line": 4, "lut_entry_bits": 8,
    },
    "matmul_tiled_asym_fp4_e4m3s_64x64.c": {
        "header": "matmul_data_asym_fp4_e4m3s.h",
        "cell": FP4_DIRECT_E4M3_CELL,
        "activation_array": "A_in_hw[32][64]", "use_lut": False,
        "weight_array": "B_in[MATMUL_K][MATMUL_N]",
        "lut_words_per_line": 0, "lut_entry_bits": 0,
    },
}
_VARIANTS_BY_HEADER = {variant["header"]: variant for variant in _VARIANTS.values()}
_FORMAT_TOKEN = {
    "e4m3": ("fp8_e4m3", "lut"), "e4m3s": ("fp8_e4m3", "direct"),
    "e5m2": ("fp8_e5m2", "lut"), "e5m2s": ("fp8_e5m2", "direct"),
    "e3m2": ("fp6_e3m2", "lut"), "e2m3": ("fp6_e2m3", "lut"),
    "fp6": ("fp6_e3m2", "lut"), "fp4": ("fp4_e2m1", "direct"),
}


def _source_variant(source: Path, header: Path, profile: dict,
                    header_text: str) -> dict:
    """Derive the source layout and legal mode from one named DIM16 test."""
    pinned = _VARIANTS.get(source.name)
    if pinned is not None:
        if header.name != pinned["header"]:
            raise ValueError("selected asymmetric source/header pair differs")
        return pinned
    match = re.fullmatch(r"matmul_tiled_asym_([a-z0-9]+)_([a-z0-9]+)_64x64(?:_dim(8|32))?\.c",
                         source.name)
    if match is None:
        raise ValueError("selected asymmetric specialization needs a named Nicolas 64-cubed test")
    left, right, dim_suffix = match.groups()
    mesh_dim = int(dim_suffix) if dim_suffix else 16
    if (left not in _FORMAT_TOKEN or right not in _FORMAT_TOKEN or
            header.name != f"matmul_data_asym_{left}_{right}" +
            (f"_dim{mesh_dim}.h" if dim_suffix else ".h")):
        raise ValueError("selected asymmetric source/header pair or format differs")
    activation_format, activation_projection = _FORMAT_TOKEN[left]
    weight_format, weight_projection = _FORMAT_TOKEN[right]
    cells = [cell for cell in profile["legal_compute"] if
             all(cell[key] == value for key, value in {
                 "activation_format": activation_format,
                 "activation_projection": activation_projection,
                 "weight_format": weight_format,
                 "weight_projection": weight_projection,
             }.items())]
    if len(cells) != 1:
        raise ValueError("selected asymmetric source has no unique legal profile mode")
    activation_array = ("A_in_hw[32][64]" if "A_in_hw[32][64]" in header_text else
                        "A_in[MATMUL_M][MATMUL_K]")
    weight_array = ("B_in[MATMUL_K][MATMUL_N]" if
                    "B_in[MATMUL_K][MATMUL_N]" in header_text else None)
    lut_shapes = {name: re.search(rf"\b{name}\[32\]\[(3|4)\]", header_text)
                  for name in ("A_lut", "B_lut", "C_lut")}
    lut_arrays = [name for name, shape in lut_shapes.items() if shape is not None]
    words = {int(shape.group(1)) for shape in lut_shapes.values() if shape is not None}
    use_lut = activation_projection == "lut" or weight_projection == "lut"
    if (len(words) > 1 or (use_lut and (not words or not lut_arrays)) or
            (not use_lut and lut_arrays)):
        raise ValueError("asymmetric source LUT arrays differ from legal projection")
    return {"header": header.name, "cell": cells[0], "mesh_dim": mesh_dim,
            "activation_array": activation_array, "use_lut": use_lut,
            "lut_arrays": lut_arrays,
            "lut_words_per_line": next(iter(words), 0),
            "lut_entry_bits": 2 * next(iter(words), 0),
            **({"weight_array": weight_array} if weight_array else {})}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_lut_fits(profile: dict, variant: dict) -> bool:
    """A direct mode needs no LUT RAM, regardless of a profile's LUT width."""
    bits = variant["lut_entry_bits"]
    if not variant["use_lut"]:
        return bits == 0
    config = profile["resources"].get("lut_config")
    return (isinstance(config, dict) and isinstance(bits, int) and bits > 0 and
            config.get("address_bits") == 4 and
            isinstance(config.get("read_data_bits"), int) and
            config["read_data_bits"] >= bits)


def source_recipe(source: Path, header: Path, profile: dict) -> dict:
    """Bind one checked-in Nicolas test and its data to a legal profile mode."""
    if not source.is_file() or not header.is_file():
        raise ValueError("selected asymmetric source or data header is absent")
    source_text, header_text = source.read_text(), header.read_text()
    variant = _source_variant(source, header, profile, header_text)
    cell = variant["cell"]
    mesh_dim = variant.get("mesh_dim", 16)
    m, n, k = variant.get("shape", [64, 64, 64])
    require_compute(profile, cell["activation_format"], cell["weight_format"],
                    pe_mode=cell["pe_mode"],
                    activation_projection=cell["activation_projection"],
                    weight_projection=cell["weight_projection"])
    if (profile["geometry"] != {"mesh_rows": mesh_dim, "mesh_columns": mesh_dim,
                                "tile_rows": 1, "tile_columns": 1} or
            profile["resources"]["scratchpad_bytes"] != 262144 or
            not _source_lut_fits(profile, variant)):
        raise ValueError("selected asymmetric source mesh, scratchpad, or LUT differs from profile")
    if mesh_dim in (8, 32) and f"#define DIM {mesh_dim}" not in source_text:
        raise ValueError("Nicolas source mesh dimension differs from selected profile")
    for marker in (f'#include "include/{header.name}"',):
        if marker not in source_text:
            raise ValueError(f"Nicolas source command contract changed: {marker}")
    lut_marker = f'#define USE_LUT {int(variant["use_lut"])}'
    config_marker = variant.get("source_config_marker")
    if lut_marker not in source_text and not (
            not variant["use_lut"] and "((uint64_t)(0) << 5)" in source_text) and not (
            config_marker and config_marker in source_text):
        raise ValueError(f"Nicolas source command contract changed: {lut_marker}")
    if not re.search(r"gemmini_loop_ws_spad\(\s*tiles_I,\s*tiles_J,\s*tiles_K", source_text):
        raise ValueError("Nicolas source loop schedule changed")
    for marker in (f"#define MATMUL_M   {m}", f"#define MATMUL_K   {k}",
                   f"#define MATMUL_N   {n}", variant["activation_array"],
                   variant.get("weight_decl", variant.get(
                       "weight_array", "B_in[MATMUL_K][MATMUL_N / 2]")),
                   variant.get("activation_scales_decl",
                               "A_scales_row[MATMUL_GK][MATMUL_M]"),
                   variant.get("weight_scales_decl",
                               "B_scales_col[MATMUL_GK][MATMUL_N]"),
                   variant.get("golden_decl", "C_out_bf16[MATMUL_M][MATMUL_N]")):
        if marker not in header_text:
            raise ValueError(f"Nicolas source data contract changed: {marker}")
    if cell["activation_format"] in {"fp8_e4m3", "fp8_e5m2"}:
        altfmt = int(cell["activation_format"] == "fp8_e5m2")
        if f"#define MX_ALTFMT {altfmt}" not in source_text:
            raise ValueError("Nicolas FP8 source alternate format selection changed")
    if cell["weight_format"] == "fp6_e2m3":
        if "((uint64_t)(1) << 12)" not in source_text:
            raise ValueError("Nicolas E2M3 weight format encoding changed")
    for lut_array in variant.get("lut_arrays", ("A_lut", "B_lut", "C_lut")
                                 if variant["use_lut"] else ()):
        if f'{lut_array}[{variant.get("lut_lines", 32)}][{variant["lut_words_per_line"]}]' not in header_text:
            raise ValueError(f"Nicolas source {lut_array} changed")
    altfmt_match = re.search(r"^#define MX_ALTFMT\s+([01])\b", source_text, re.M)
    config_altfmt = int(altfmt_match.group(1)) if altfmt_match else 0
    if cell["activation_format"] != "fp4_e2m1" and config_altfmt != int(
            cell["activation_format"] in {"fp8_e5m2", "fp6_e2m3"}):
        raise ValueError("Nicolas activation alternate format differs from selected mode")
    weight_alt = int(cell["weight_format"] in {"fp8_e5m2", "fp6_e2m3"})
    weight_altfmt_diff = (config_altfmt ^ weight_alt) if cell["weight_format"] != "fp4_e2m1" else 0
    if cell["weight_format"] != "fp4_e2m1":
        config_block = source_text.split("ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC,", 1)[-1].split("k_CONFIG);", 1)[0]
        coded_diff = re.search(r"\(\(uint64_t\)\(([01])\) << 31\)", config_block)
        source_diff = int(coded_diff.group(1)) if coded_diff else 0
        if source_diff != weight_altfmt_diff:
            raise ValueError("Nicolas weight alternate format differs from selected mode")
    source_layout = {key: value for key, value in variant.items() if key != "cell"}
    source_layout["lut_arrays"] = list(variant.get("lut_arrays", ("A_lut", "B_lut", "C_lut")
                                                     if variant["use_lut"] else ()))
    source_layout["mesh_dim"] = mesh_dim
    source_layout["config_altfmt"] = config_altfmt
    source_layout["weight_altfmt_diff"] = weight_altfmt_diff
    return {"schema": "mx_gemmini.asymmetric_source_recipe.v1",
            "site_id": "functional:matmul", "shape": [m, n, k],
            "frontend_capture_format": "mxfp8",
            "source_driver_sha256": sha256(source),
            "source_header_sha256": sha256(header),
            "profile_sha256": profile_sha256(profile), "compute": cell.copy(),
            "source_layout": source_layout}


_GENERATED_MODES = {
    "e2m3_e4m3s", "e3m2_e3m2", "e4m3s_e2m3",
    "e4m3s_e4m3", "e4m3_e4m3s",
}
_MESH_GENERATED_MODES = {
    "fp4_fp4", "fp4_e4m3", "fp4_e4m3s", "e2m3_e4m3s",
    "e2m3_e2m3", "e2m3_e4m3", "e3m2_e4m3", "e3m2_e3m2",
    "e3m2_e4m3s", "e4m3s_e2m3", "e4m3s_e4m3s",
    "e4m3s_e4m3", "e4m3_e4m3s", "e4m3_e4m3", "e5m2_e5m2",
}
_MESH_BASELINE_SHA256 = {
    8: "0ae643c6be0e288d9d9b3cb9c163ac8c669b0fb8941cee85cd1a7daa35c94b83",
    16: "b4c7f87d11a0e096bc8ccfe2234c88c992ac5d2bccdb7b4d09c9aca0d770d8dc",
    32: "13730bd97ec1306bb93d11f7cb7464dd5ce5b1b0a111f524c5679524c9a507b5",
}


def generated_header_recipe(generator: Path, header: Path, profile: dict) -> dict:
    """Bind a new Nicolas-model header without claiming a checked-in C driver.

    The generator is pinned gen_asym.py. The header bytes, baseline reproduction,
    and every required array are checked before the contraction is lowered.
    """
    if generator.name != "gen_asym.py" or not generator.is_file() or not header.is_file():
        raise ValueError("generated MX recipe needs Nicolas's generator and its header")
    match = re.fullmatch(
        r"matmul_data_asym_([a-z0-9]+)_([a-z0-9]+?)(?:_dim(8|32))?\.h",
        header.name)
    if match is None:
        raise ValueError("generated MX header names an unregistered format pair")
    dim = int(match[3]) if match[3] else 16
    name = f"{match[1]}_{match[2]}"
    if name not in (_GENERATED_MODES if dim == 16 else _MESH_GENERATED_MODES):
        raise ValueError("generated MX header names an unregistered format pair")
    generation_manifest = header.parent / (
        "mx_gemmini_generated_modes_manifest.json" if dim == 16 else
        f"mx_gemmini_generated_modes_dim{dim}_manifest.json")
    if not generation_manifest.is_file():
        raise ValueError("generated MX header lacks its pinned generation manifest")
    provenance = json.loads(generation_manifest.read_text())
    expected_schema = ("mx_gemmini.nicolas_generated_asymmetric_headers.v1"
                       if dim == 16 else "mx_gemmini.nicolas_generated_mesh_headers.v1")
    wrapper = ("generate_nicolas_missing_headers.py" if dim == 16 else
               "generate_nicolas_mesh_headers.py")
    if (provenance.get("schema") != expected_schema or
            (dim != 16 and provenance.get("mesh_dim") != dim) or
            provenance.get("rtl_revision") !=
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52" or
            provenance.get("software_revision") !=
            "350547f9843f46f485d4d1dd4c20b2f52f4844bf" or
            provenance.get("microxcaling_revision") !=
            "7bc41952de394f5cc5e782baf132e7c7542eb4e4" or
            provenance.get("microxcaling_elemwise_sha256") !=
            "57a825c801ec551d63a2aa6c827372b5ab420ff7b0f76c4d599d7d5050e83130" or
            provenance.get("baseline_sha256") != _MESH_BASELINE_SHA256[dim] or
            provenance.get("wrapper_sha256") != sha256(
                Path(__file__).resolve().parents[1] /
                f"tools/{wrapper}") or
            provenance.get("generator_sha256") != sha256(generator) or
            provenance.get("headers_sha256", {}).get(name) !=
            sha256(header)):
        raise ValueError("generated MX header differs from its pinned generation manifest")
    synthetic_name = (f"matmul_tiled_asym_{name}_64x64" +
                      (f"_dim{dim}" if dim != 16 else "") + ".c")
    variant = _source_variant(Path(synthetic_name), header, profile, header.read_text())
    cell = variant["cell"]
    require_compute(profile, cell["activation_format"], cell["weight_format"],
                    pe_mode=cell["pe_mode"],
                    activation_projection=cell["activation_projection"],
                    weight_projection=cell["weight_projection"])
    if (profile["geometry"] != {"mesh_rows": dim, "mesh_columns": dim,
                                "tile_rows": 1, "tile_columns": 1} or
            profile["resources"]["scratchpad_bytes"] != 262144 or
            not _source_lut_fits(profile, variant)):
        raise ValueError("generated MX header differs from selected mesh or LUT")
    text = header.read_text()
    for marker in ("#define MATMUL_M   64", "#define MATMUL_K   64",
                   "#define MATMUL_N   64", variant["activation_array"],
                   variant.get("weight_array", "B_in[MATMUL_K][MATMUL_N / 2]"),
                   "A_scales_row[MATMUL_GK][MATMUL_M]",
                   "B_scales_col[MATMUL_GK][MATMUL_N]",
                   "C_out_bf16[MATMUL_M][MATMUL_N]"):
        if marker not in text:
            raise ValueError(f"generated MX header lacks {marker}")
    recipe = {
        "schema": "mx_gemmini.asymmetric_source_recipe.v1",
        "origin": "nicolas_generated_header",
        "site_id": "functional:matmul", "shape": [64, 64, 64],
        "frontend_capture_format": "mxfp8",
        "source_generator_sha256": sha256(generator),
        "source_generation_manifest_sha256": sha256(generation_manifest),
        "source_header_sha256": sha256(header),
        "profile_sha256": profile_sha256(profile),
        "compute": cell.copy(),
        "source_layout": {**variant, "mesh_dim": dim,
                          "lut_arrays": list(variant.get("lut_arrays", ())),
                          "config_altfmt": int(cell["activation_format"] in
                                               {"fp8_e5m2", "fp6_e2m3"}),
                          "weight_altfmt_diff": int(
                              cell["activation_format"] in {"fp8_e5m2", "fp6_e2m3"}) ^
                              int(cell["weight_format"] in {"fp8_e5m2", "fp6_e2m3"})},
    }
    # Parse every named array now so malformed or wrong-size headers fail
    # before capture, rather than after the physical program is emitted.
    read_asymmetric_resources(header, recipe)
    return recipe


def _recheck_recipe(recipe: dict, source: Path, header: Path, profile: dict) -> dict:
    if recipe.get("origin") == "nicolas_generated_header":
        return generated_header_recipe(source, header, profile)
    return source_recipe(source, header, profile)


def specialize_handoff(mlir_text: str, profile: dict, recipe: dict) -> str:
    """Change a captured site's MX semantics only with an explicit source recipe."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin, IntegerAttr, StringAttr
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser
    from xdsl.printer import Printer

    if (recipe.get("schema") != "mx_gemmini.asymmetric_source_recipe.v1" or
            recipe.get("site_id") != "functional:matmul" or
            recipe.get("shape") not in ([64, 64, 64], [128, 128, 128],
                                        [128, 128, 256]) or
            recipe.get("frontend_capture_format") != "mxfp8" or
            recipe.get("profile_sha256") != profile_sha256(profile) or
            (recipe.get("compute") not in (ASYM_CELL, DIRECT_CELL, FP6_FP4_CELL,
                                           FP4_FP6_CELL, E4M3_E2M3_CELL,
                                           E5M2_FP4_CELL, E4M3_DIRECT_E3M2_CELL,
                                           FP4_DIRECT_E4M3_CELL) and
             not isinstance(recipe.get("source_layout"), dict)) or
            any(not isinstance(recipe.get(key), str) or len(recipe[key]) != 64 or
                any(ch not in "0123456789abcdef" for ch in recipe[key])
                for key in (("source_generator_sha256" if recipe.get("origin") ==
                             "nicolas_generated_header" else "source_driver_sha256"),
                            "source_header_sha256"))):
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
    source_key = ("source_generator_sha256" if recipe.get("origin") ==
                  "nicolas_generated_header" else "source_driver_sha256")
    module.attributes[f"mx.{source_key}"] = StringAttr(recipe[source_key])
    if recipe.get("origin") == "nicolas_generated_header":
        module.attributes["mx.source_generation_manifest_sha256"] = StringAttr(
            recipe["source_generation_manifest_sha256"])
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

    if recipe != _recheck_recipe(recipe, source, header, profile):
        raise ValueError("asymmetric source bytes, profile, or recipe changed")
    checked = verify_ir(mlir_text, profile)
    if checked["contracts"] != 1 or checked["encodes"] or checked["requantizes"]:
        raise ValueError("physical asymmetric schedule needs one BF16 contract")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    recipe_digest = hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source_key = ("source_generator_sha256" if recipe.get("origin") ==
                  "nicolas_generated_header" else "source_driver_sha256")
    if (_text_attr(module, "mx.asymmetric_recipe_sha256") != recipe_digest or
            _text_attr(module, f"mx.{source_key}") != recipe[source_key] or
            (recipe.get("origin") == "nicolas_generated_header" and
             _text_attr(module, "mx.source_generation_manifest_sha256") !=
             recipe["source_generation_manifest_sha256"]) or
            _text_attr(module, "mx.source_header_sha256") != recipe["source_header_sha256"]):
        raise ValueError("profile-bound MLIR lacks matching asymmetric source provenance")
    operations = [op for op in module.walk()
                  if _operation_name(op).startswith("mx_gemmini.") and
                  _operation_name(op) not in {"mx_gemmini.resource", "mx_gemmini.upload_lut"}]
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
    variant = recipe.get("source_layout") or _VARIANTS_BY_HEADER[header.name]
    if variant["header"] != header.name:
        raise ValueError("asymmetric source layout names a different data header")
    m, n, k = recipe["shape"]
    packed_activation = variant["activation_array"].startswith("A_in_hw[")
    lut_words = variant["lut_words_per_line"]
    a_name = "A_in_hw" if packed_activation else "A_in"
    a_shape = (variant["activation_array"].removeprefix("A_in_hw") if packed_activation
               else "[MATMUL_M][MATMUL_K]")
    weight_shape = (variant["weight_decl"].removeprefix("B_in") if
                    variant.get("weight_decl") else
                    "[MATMUL_K][MATMUL_N]" if variant.get("weight_array") else
                    "[MATMUL_K][MATMUL_N / 2]")
    activation_scales_shape = variant.get(
        "activation_scales_decl", "A_scales_row[MATMUL_GK][MATMUL_M]").removeprefix(
            "A_scales_row")
    weight_scales_shape = variant.get(
        "weight_scales_decl", "B_scales_col[MATMUL_GK][MATMUL_N]").removeprefix(
            "B_scales_col")
    golden_shape = variant.get(
        "golden_decl", "C_out_bf16[MATMUL_M][MATMUL_N]").removeprefix("C_out_bf16")
    weight_count = k * n if variant.get("weight_array") else k * n // 2
    resources = {
        "activation": bytes(_array(text, name=a_name, ctype="uint8_t",
                                    dimensions=a_shape, count=m * k // (2 if packed_activation else 1),
                                    maximum=255)),
        "weight": bytes(_array(text, name="B_in", ctype="uint8_t",
                                dimensions=weight_shape,
                                count=weight_count, maximum=255)),
        "activation_scales": bytes(_array(text, name="A_scales_row", ctype="uint8_t",
                                            dimensions=activation_scales_shape,
                                            count=k // 32 * m, maximum=255)),
        "weight_scales": bytes(_array(text, name="B_scales_col", ctype="uint8_t",
                                        dimensions=weight_scales_shape,
                                        count=k // 32 * n, maximum=255)),
        "golden_bf16": _bytes(_array(text, name="C_out_bf16", ctype="uint16_t",
                                      dimensions=golden_shape,
                                      count=m * n, maximum=0xffff), 2),
    }
    if variant["use_lut"]:
        lut_lines = variant.get("lut_lines", 32)
        for c_name, resource in (("A_lut", "activation_lut"),
                                 ("B_lut", "weight_lut"),
                                 ("C_lut", "output_lut")):
            if c_name not in variant.get("lut_arrays", ("A_lut", "B_lut", "C_lut")):
                continue
            resources[resource] = _bytes(_array(text, name=c_name, ctype="uint32_t",
                                               dimensions=f"[{lut_lines}][{lut_words}]",
                                               count=lut_lines * lut_words,
                                               maximum=0xffffffff), 4)
    return resources


def _resource_manifest(recipe: dict, resources: dict[str, bytes], header: Path) -> dict:
    """Describe every Nicolas source byte array with its physical shape."""
    from .source_payload import Resource

    variant = recipe.get("source_layout") or _VARIANTS_BY_HEADER[header.name]
    m, n, k = recipe["shape"]
    packed_activation = variant["activation_array"].startswith("A_in_hw[")
    shapes = {
        "activation": ((m // 2, k) if packed_activation else (m, k), 8,
                       "packed_even_odd_m_nibbles" if packed_activation else "row_major_codes"),
        "weight": ((k, n) if variant.get("weight_array") else (k, n // 2), 8,
                   "row_major_codes" if variant.get("weight_array") else
                   "packed_even_odd_n_nibbles"),
        "activation_scales": ((k // 32, m), 8, "k_group_row_e8m0"),
        "weight_scales": ((k // 32, n), 8, "k_group_column_e8m0"),
        "golden_bf16": ((m, n), 16, "row_major_bf16"),
    }
    if variant["use_lut"]:
        words = variant["lut_words_per_line"]
        bits = variant["lut_entry_bits"]
        lut_lines = variant.get("lut_lines", 32)
        all_luts = {
            "activation_lut": ((lut_lines, words), 32, f"row_pair_lut_{bits}bit"),
            "weight_lut": ((lut_lines, words), 32, f"column_pair_lut_{bits}bit"),
            "output_lut": ((lut_lines, words), 32, f"output_pair_lut_{bits}bit"),
        }
        names = {"A_lut": "activation_lut", "B_lut": "weight_lut",
                 "C_lut": "output_lut"}
        shapes.update({names[name]: all_luts[names[name]] for name in
                       variant.get("lut_arrays", ("A_lut", "B_lut", "C_lut"))})
    if set(resources) != set(shapes):
        raise ValueError("asymmetric source resource set differs from selected mode")
    return {
        "schema": "mx_gemmini.asymmetric_resource_manifest.v2",
        "site_id": recipe["site_id"],
        "profile_sha256": recipe["profile_sha256"],
        "origin": ("nicolas_generated_header_specialization" if recipe.get("origin") ==
                   "nicolas_generated_header" else "nicolas_source_header_specialization"),
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
    from .resource_ir import attach_source_resources
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
    attach_source_resources(module, contract, manifest)
    output = StringIO()
    Printer(stream=output).print_op(module)
    bound = output.getvalue() + "\n"
    verify_ir(bound, profile)
    return bound


def lower_asymmetric_physical(mlir_text: str, profile: dict, recipe: dict, *,
                              source: Path, header: Path):
    """Lower a source-bound asymmetric matmul to physical MX commands."""
    from .command_ir import Fence, Operand
    from .physical_program import (PhysicalProgram, PhysicalStep, _cmd, _config_ld,
                                   _config_st, _transfer)

    if recipe != _recheck_recipe(recipe, source, header, profile):
        raise ValueError("asymmetric physical lowering source or target changed")
    _validate_bound_site(mlir_text, profile, recipe, source=source, header=header)
    resources = read_asymmetric_resources(header, recipe)
    from .source_payload import manifest_json, manifest_sha256
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
            (_text_attr(module, "mx.payload_manifest_json") is not None and
             _text_attr(module, "mx.payload_manifest_json") !=
             manifest_json(resource_manifest)) or
            len(contracts) != 1 or
            _text_attr(contracts[0], "payload_manifest_sha256") != payload_digest or
            _text_attr(contracts[0], "payload_origin") != resource_manifest["origin"]):
        raise ValueError("asymmetric physical lowering payload differs from typed MLIR")
    cell = recipe["compute"]
    variant = recipe.get("source_layout") or _VARIANTS[source.name]
    use_lut = cell["activation_projection"] == "lut" or cell["weight_projection"] == "lut"
    packed_activation = variant["activation_array"].startswith("A_in_hw[")
    m, n, k_dim = recipe["shape"]
    weight_stride = n if variant.get("weight_array") else n // 2
    dim = variant.get("mesh_dim", 16)
    ti = m // (dim * (2 if packed_activation else 1))
    tj = n // (dim * (2 if weight_stride == n // 2 else 1))
    tki = k_dim // dim
    if (not ti or not tj or not tki or m % (dim * (2 if packed_activation else 1))
            or n % (dim * (2 if weight_stride == n // 2 else 1)) or k_dim % dim):
        raise ValueError("asymmetric source shape is not tileable on selected mesh")
    scratchpad_rows = profile["resources"]["scratchpad_bytes"] // dim
    a_base = 0
    b_end = 8192 if dim == 16 else min(16384, scratchpad_rows)
    c_base = 128 if recipe["shape"] == [64, 64, 64] and dim == 16 else ti * tki * dim
    b_base = b_end - tki * tj * dim
    if (b_base < ti * tki * dim or b_end > scratchpad_rows or
            c_base + m * n * 2 // dim > b_base):
        raise ValueError("asymmetric operand and BF16 readout rows exceed profile scratchpad")
    steps: list[PhysicalStep] = []

    def issue(phase: str, command) -> None:
        steps.append(PhysicalStep(phase, None, command))

    issue("configure", _cmd(7, 0, 0))
    format_code = {"fp8_e4m3": 0, "fp8_e5m2": 0,
                   "fp6_e3m2": 1, "fp6_e2m3": 1, "fp4_e2m1": 2}
    activation_code = format_code[cell["activation_format"]]
    weight_code = format_code[cell["weight_format"]]
    lut_entry_bits = variant["lut_entry_bits"]
    activation_altfmt = variant.get("config_altfmt", int(cell["activation_format"] == "fp8_e5m2"))
    weight_altfmt_diff = variant.get("weight_altfmt_diff",
                                     int(cell["weight_format"] == "fp6_e2m3"))
    config_ex = (1 << 16) | (weight_code << 12) | (activation_code << 10) | \
                (3 << 14) | (int(use_lut) << 5) | (1 << 2) | \
                (activation_altfmt << 6) | (weight_altfmt_diff << 31)
    issue("configure", _cmd(0, config_ex, 1 << 48))
    if use_lut:
        for resource, selector in (("weight_lut", 0), ("activation_lut", 1),
                                   ("output_lut", 2)):
            if resource not in resources:
                continue
            issue("upload_lut", _cmd(29, Operand(buffer=resource),
                                     (lut_entry_bits << 34) | (selector << 32) |
                                     variant.get("lut_lines", 32)))
    else:
        issue("disable_lut", _cmd(30, 0, 0))
    issue("upload_scales", _cmd(27, Operand(buffer="activation_scales"),
                                len(resources["activation_scales"])))
    issue("upload_scales", _cmd(27, Operand(buffer="weight_scales"),
                                (1 << 32) | len(resources["weight_scales"])))
    issue("upload_scales", Fence())
    issue("move_activation", _config_ld(k_dim, dim=dim))
    for i in range(ti):
        for k in range(tki):
            offset = i * dim * k_dim + k * dim
            row = a_base + (i * tki + k) * dim
            issue("move_activation", _transfer(2, "activation", offset, row, dim=dim))
    issue("move_weight", _config_ld(weight_stride, dim=dim))
    for k in range(tki):
        for j in range(tj):
            offset = k * dim * weight_stride + j * dim
            row = b_base + (k * tj + j) * dim
            issue("move_weight", _transfer(2, "weight", offset, row, dim=dim))
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
    for row in range(0, m * n * 2 // dim, dim):
        issue("readout", _transfer(3, "output_bf16", row * dim, c_base + row,
                                   dim=dim))
    issue("readout", Fence())
    plan = {"shape_mnk": [m, n, k_dim], "tile_mnk": [m, n, k_dim],
            "activation_projection": recipe["compute"]["activation_projection"],
            "weight_projection": recipe["compute"]["weight_projection"],
            "pe_mode": recipe["compute"]["pe_mode"],
            "scratchpad_rows": scratchpad_rows, "mesh_dim": dim,
            "a_row": a_base, "b_row": b_base, "c_row": c_base,
            "tiles_i": ti, "tiles_j": tj, "tiles_k": tki}
    return PhysicalProgram(profile_sha256(profile), payload_digest,
                           "spike_serial", (m, n, k_dim), plan, tuple(steps)), resources, resource_manifest
