"""Materialize source MX GEMM arrays as checked compiler input resources.

The bundle is independent of the C header after creation. Its manifest binds
every byte array to a shape, storage layout, and SHA-256 digest. It does not
assert that model2MLIR produced the source quantized bytes: the capture gives
the contraction structure, while this explicit source specialization supplies
the packed values used for source parity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from .source_fp6 import _array, _bytes, read_source_fp6_payload
from .source_gemm import SourceGemm
from .quant_reference import (exact_bf16_x2, quantize_bf16_fp4_output, quantize_bf16_fp8_output,
                              quantize_bf16_fp6_lut_output)


SCHEMA = "mx_gemmini.source_payload.v1"
ATTENTION_QK_CANDIDATE_ORIGIN = "radiance_source_derived_attention_qk_candidate"
ATTENTION_PV_PROXY_ORIGIN = "radiance_source_derived_attention_pv_proxy"
DERIVED_GEMM_FIXTURE_ORIGIN = "radiance_source_derived_gemm_fixture"
TARGET_MESH_REFERENCE_ORIGIN = "radiance_source_target_mesh_reference"


def validate_target_mesh_reference(manifest: dict) -> None:
    """Keep a derived DIM8/32 numerical reference distinct from source goldens."""
    from .mesh_reference import (_MODEL_CPP_SHA256, _MODEL_MATH_SHA256,
                                 _MODEL_FLOOR_MATH_SHA256)

    policy = manifest.get("target_mesh_reference")
    descriptor = manifest.get("resources", {}).get("golden_bf16", {})
    required = {"schema", "mesh_dim", "source_golden_sha256",
                "model_cpp_sha256", "model_math_sha256",
                "transformed_cpp_sha256", "target_golden_sha256"}
    floor_reference = isinstance(policy, dict) and policy.get("schema") == (
        "mx_gemmini.radiance_target_mesh_reference.v2")
    if floor_reference:
        required |= {"product_floor_exponent", "transformed_math_sha256"}
    if manifest.get("precision") == "FP6":
        required |= {"lut_granularity_shift", "unpacked_activation_lut_sha256",
                     "unpacked_weight_lut_sha256"}
    if (manifest.get("origin") != TARGET_MESH_REFERENCE_ORIGIN or
            manifest.get("precision") not in {"FP8", "FP4", "FP6"} or
            manifest.get("output_format") is not None or
            "source_derivation" in manifest or
            not isinstance(policy, dict) or
            set(policy) != required or
            policy.get("schema") not in {
                "mx_gemmini.radiance_target_mesh_reference.v1",
                "mx_gemmini.radiance_target_mesh_reference.v2"} or
            policy.get("mesh_dim") not in {8, 32} or
            policy.get("model_cpp_sha256") != _MODEL_CPP_SHA256 or
            policy.get("model_math_sha256") != _MODEL_MATH_SHA256 or
            policy.get("target_golden_sha256") != descriptor.get("sha256") or
            policy.get("source_golden_sha256") == descriptor.get("sha256")):
        raise ValueError("target mesh reference lacks pinned numerical provenance")
    if floor_reference and (policy.get("product_floor_exponent") != -16 or
                            policy.get("transformed_math_sha256") !=
                            _MODEL_FLOOR_MATH_SHA256):
        raise ValueError("target mesh reference lacks RTL product-floor provenance")
    if manifest["precision"] == "FP6" and (
            policy.get("lut_granularity_shift") != 1 or
            any(re.fullmatch(r"[0-9a-f]{64}", policy.get(f"unpacked_{name}_sha256", "")) is None
                for name in ("activation_lut", "weight_lut"))):
        raise ValueError("FP6 target mesh reference lacks unpacked LUT provenance")
    for name in ("source_golden_sha256", "transformed_cpp_sha256"):
        value = policy[name]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"target mesh reference lacks {name}")


def replace_source_golden_with_mesh_reference(directory: Path, target: bytes,
                                              policy: dict) -> dict:
    """Rewrite only the expected BF16 bytes in a freshly written source bundle."""
    manifest, resources = load_bundle(directory)
    if (manifest.get("origin") != "radiance_source_header_specialization" or
            manifest.get("output_format") is not None or
            len(target) != len(resources["golden_bf16"]) or
            policy.get("source_golden_sha256") != _sha(resources["golden_bf16"])):
        raise ValueError("target mesh reference does not match the source bundle")
    manifest["origin"] = TARGET_MESH_REFERENCE_ORIGIN
    manifest["target_mesh_reference"] = policy
    manifest["resources"]["golden_bf16"]["sha256"] = _sha(target)
    validate_target_mesh_reference(manifest)
    (directory / "golden_bf16.bin").write_bytes(target)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def validate_derived_gemm_fixture(manifest: dict) -> None:
    """Require explicit provenance for a generated, uncommitted MX GEMM case."""
    policy = manifest.get("source_derivation")
    if (manifest.get("origin") != DERIVED_GEMM_FIXTURE_ORIGIN or
            manifest.get("precision") != "FP4" or
            manifest.get("shape_mnk") != [256, 256, 256] or
            manifest.get("tile_mnk") != [128, 128, 128] or
            manifest.get("output_format") is not None or
            not isinstance(policy, dict) or
            set(policy) != {"schema", "transformation", "source_revision",
                            "base_driver_sha256", "source_generator_sha256",
                            "golden_model_sha256", "golden_cpp_sha256",
                            "derived_driver_sha256", "generated_header_sha256"} or
            policy.get("schema") != "mx_gemmini.radiance_generated_fp4_gemm_fixture.v1" or
            policy.get("transformation") !=
            "fp8_m256n256k256_tk256_to_fp4_tk128_with_activation_alias_v2" or
            policy.get("derived_driver_sha256") != manifest.get("source_driver_sha256") or
            policy.get("generated_header_sha256") != manifest.get("source_header_sha256")):
        raise ValueError("derived FP4 GEMM fixture lacks its explicit source transformation")
    for field in ("source_revision", "base_driver_sha256", "source_generator_sha256",
                  "golden_model_sha256", "golden_cpp_sha256",
                  "derived_driver_sha256", "generated_header_sha256"):
        value = policy[field]
        size = 40 if field == "source_revision" else 64
        if not isinstance(value, str) or len(value) != size or any(
                char not in "0123456789abcdef" for char in value):
            raise ValueError(f"derived FP4 GEMM fixture lacks {field}")


def validate_attention_qk_candidate(manifest: dict) -> None:
    """Keep a requantized attention probe distinct from exact source bytes."""
    if manifest.get("origin") != ATTENTION_QK_CANDIDATE_ORIGIN:
        raise ValueError("attention QK candidate has the wrong payload origin")
    policy = manifest.get("source_derivation")
    if (not isinstance(policy, dict) or
            policy.get("schema") != "mx_gemmini.attention_qk_shift.v1" or
            not isinstance(policy.get("stage"), str) or
            re.fullmatch(r"gqa_qk_head[0-7]_block[01]", policy["stage"]) is None or
            type(policy.get("e8m0_shift")) is not int or
            not 1 <= policy["e8m0_shift"] <= 8 or
            policy.get("oracle") != "dim16_reduced_precision_product_and_accumulator" or
            policy.get("source_header_sha256") != manifest.get("source_header_sha256") or
            manifest.get("precision") != "FP8" or
            manifest.get("shape_mnk") != [64, 64, 64] or
            manifest.get("tile_mnk") != [64, 64, 64]):
        raise ValueError("attention QK candidate lacks its explicit source derivation")
    hashes = policy.get("source_arrays_sha256")
    if (not isinstance(hashes, dict) or
            set(hashes) != {"activation", "weight", "activation_scales", "weight_scales"} or
            any(not isinstance(value, str) or len(value) != 64 or
                any(c not in "0123456789abcdef" for c in value)
                for value in hashes.values())):
        raise ValueError("attention QK candidate lacks pinned source operand hashes")


def validate_attention_pv_proxy(manifest: dict) -> None:
    """Keep a Torch-exp Muon boundary probe distinct from executed Muon bytes."""
    if manifest.get("origin") != ATTENTION_PV_PROXY_ORIGIN:
        raise ValueError("attention PV proxy has the wrong payload origin")
    policy = manifest.get("source_derivation")
    if (not isinstance(policy, dict) or
            policy.get("schema") != "mx_gemmini.attention_pv_proxy.v1" or
            policy.get("stage") != "gqa_pv_head0_block0" or
            policy.get("exp_policy") != "torch_exp_bf16_proxy_for_mu_fexp" or
            policy.get("causal_mask") != "first_key_block_fully_visible" or
            policy.get("output_oracle") != "dim16_reduced_precision_product_and_accumulator" or
            policy.get("source_header_sha256") != manifest.get("source_header_sha256") or
            manifest.get("precision") != "FP8" or
            manifest.get("shape_mnk") != [64, 64, 64] or
            manifest.get("tile_mnk") != [64, 64, 64] or
            manifest.get("output_format") is not None):
        raise ValueError("attention PV proxy lacks its explicit source derivation")
    for field in ("qk_golden_bf16_sha256", "muon_requant_source_sha256"):
        value = policy.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"attention PV proxy lacks {field}")
    hashes = policy.get("source_arrays_sha256")
    if (not isinstance(hashes, dict) or
            set(hashes) != {"activation", "activation_scales", "weight", "weight_scales"} or
            any(hashes[name] != manifest.get("resources", {}).get(name, {}).get("sha256")
                for name in hashes)):
        raise ValueError("attention PV proxy operand hashes differ from its bundle")


def vpu_requant_shape_is_legal(shape: tuple[int, int, int],
                               tile: tuple[int, int, int]) -> bool:
    """Check the one-output-tile FP8 geometry accepted by SPAD_REQUANT."""
    m, n, k = shape
    blocks = m * n // 32
    return (m == tile[0] and n == tile[1] and m % 16 == 0 and
            n % 32 == 0 and k % 32 == 0 and
            32 <= blocks <= 2048 and blocks % 32 == 0)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_sha256(manifest: dict) -> str:
    """Hash the canonical manifest, excluding any filesystem location."""
    return _sha(manifest_json(manifest).encode())


def manifest_json(manifest: dict) -> str:
    """Serialize the exact external resources bound to one typed MX site."""
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Resource:
    data: bytes
    shape: tuple[int, ...]
    element_bits: int
    layout: str

    def descriptor(self, name: str) -> dict:
        if len(self.data) != self.element_bits // 8 * _product(self.shape):
            raise ValueError(f"{name}: storage shape does not match byte count")
        return {"file": f"{name}.bin", "sha256": _sha(self.data),
                "bytes": len(self.data), "shape": list(self.shape),
                "element_bits": self.element_bits, "layout": self.layout}


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        if dimension <= 0:
            raise ValueError("payload dimensions must be positive")
        result *= dimension
    return result


def read_source_payload(kernel: SourceGemm, *,
                        fp6_quantized_specialization: bool = False,
                        vpu_spad_requant_x2: bool = False) -> dict[str, Resource]:
    """Read checked FP8/FP4/FP6 packed operands, scales, LUTs, and goldens."""
    if not kernel.data_header_present:
        raise ValueError("source MX payload requires a data header")
    m, n, k = kernel.shape
    if k % 32 or n % 32:
        raise ValueError("source MX payload requires 32-element scale groups")
    source = kernel.data_header.read_text(encoding="ascii")
    packed = kernel.datatype != "FP8"
    if kernel.datatype not in {"FP8", "FP4", "FP6"}:
        raise ValueError("unknown source MX payload precision")
    if fp6_quantized_specialization and (kernel.datatype != "FP6" or kernel.quant_output):
        raise ValueError("FP6 quantized specialization requires the checked-in fullout driver")
    if vpu_spad_requant_x2 and (kernel.datatype != "FP8" or kernel.quant_output or
                                fp6_quantized_specialization or
                                not vpu_requant_shape_is_legal(kernel.shape, kernel.tile)):
        raise ValueError("VPU/SPAD x2 specialization requires a single-tile FP8 fullout driver")
    if kernel.datatype == "FP6":
        fp6 = read_source_fp6_payload(kernel)
        resources = {
            "activation": Resource(fp6.activation_bytes, (m // 2, k), 8, "packed_even_odd_m_nibbles"),
            "weight": Resource(fp6.weight_bytes, (k, n // 2), 8, "packed_even_odd_n_nibbles"),
            "activation_lut": Resource(fp6.activation_lut_bytes, (64, 3), 32, "row_pair_lut_6bit"),
            "weight_lut": Resource(fp6.weight_lut_bytes, (64, 3), 32, "column_pair_lut_6bit"),
            "output_lut": Resource(fp6.output_lut_bytes, (64, 3), 32, "output_pair_lut_6bit"),
            "activation_scales": Resource(fp6.activation_scale_bytes, (k // 32, m), 8, "k_group_row_e8m0"),
            "weight_scales": Resource(fp6.weight_scale_bytes, (k // 32, n), 8, "k_group_column_e8m0"),
            "golden_bf16": Resource(fp6.golden_bf16_bytes, (m, n), 16, "row_major_bf16"),
        }
    else:
        restream = kernel.driver.parent.name == "gemm_mxgemmini_ws_restream"
        a_name = "A_in_hw" if packed else "A_in_data" if restream else "A_in"
        a_shape = (m // 2, k) if packed else (m, k)
        a_decl = "[MATMUL_M / 2][MATMUL_K]" if packed else "[MATMUL_M][MATMUL_K]"
        b_shape = (k, n // 2) if packed else (k, n)
        b_decl = "[MATMUL_K][MATMUL_N / 2]" if packed else "[MATMUL_K][MATMUL_N]"
        if restream:
            row = _array(source, name="A_scales_row", ctype="uint8_t",
                         dimensions="[MATMUL_GK][MATMUL_M]", count=k // 32 * m,
                         maximum=255)
            tiled = _array(source, name="A_scales_tiled", ctype="uint8_t",
                           dimensions="[256][64]", count=k // 32 * m,
                           maximum=255)
            if any(tiled[block * 64 * 64 + group * 64 + lane] !=
                   row[group * m + block * 64 + lane]
                   for block in range(4) for group in range(k // 32)
                   for lane in range(64)):
                raise ValueError("source re-stream activation-scale tiling differs from canonical data")
        resources = {
            "activation": Resource(bytes(_array(source, name=a_name, ctype="uint8_t",
                                                dimensions=a_decl, count=_product(a_shape), maximum=255)),
                                   a_shape, 8, "packed_even_odd_m_nibbles" if packed else "row_major_codes"),
            "weight": Resource(bytes(_array(source, name="B_in", ctype="uint8_t",
                                            dimensions=b_decl, count=_product(b_shape), maximum=255)),
                               b_shape, 8, "packed_even_odd_n_nibbles" if packed else "row_major_codes"),
            "activation_scales": Resource(bytes(_array(source, name="A_scales_row", ctype="uint8_t",
                                                        dimensions="[MATMUL_GK][MATMUL_M]",
                                                        count=k // 32 * m, maximum=255)),
                                          (k // 32, m), 8, "k_group_row_e8m0"),
            "weight_scales": Resource(bytes(_array(source, name="B_scales_col", ctype="uint8_t",
                                                    dimensions="[MATMUL_GK][MATMUL_N]",
                                                    count=k // 32 * n, maximum=255)),
                                      (k // 32, n), 8, "k_group_column_e8m0"),
            "golden_bf16": Resource(_bytes(_array(source, name="C_out_bf16", ctype="uint16_t",
                                                  dimensions="[MATMUL_M][MATMUL_N]",
                                                  count=m * n, maximum=65535), 2),
                                    (m, n), 16, "row_major_bf16"),
        }
    resources["output_scales"] = Resource(bytes(_array(
        source, name="C_scales_row", ctype="uint8_t",
        dimensions="[MATMUL_GN][MATMUL_M]", count=n // 32 * m, maximum=255)),
        (n // 32, m), 8, "n_group_row_e8m0")
    if kernel.quant_output and kernel.datatype == "FP6":
        projected = bytes(_array(source, name="C_proj_hw", ctype="uint8_t",
                                 dimensions="[64][128]", count=m * n // 2,
                                 maximum=255))
        packed_n = bytes(_array(source, name="C_out", ctype="uint8_t",
                                dimensions="[MATMUL_M][MATMUL_N / 2]",
                                count=m * n // 2, maximum=255))
        for row in range(m):
            for col in range(n):
                from_n = (packed_n[row * (n // 2) + col // 2] >> (4 * (col & 1))) & 15
                from_m = (projected[(row // 2) * n + col] >> (4 * (row & 1))) & 15
                if from_n != from_m:
                    raise ValueError("source FP6 output projections differ")
        source_scales = resources["output_scales"].data
        row_major_scales = bytes(source_scales[group * m + row]
                                 for row in range(m) for group in range(n // 32))
        current_codes, current_scales = quantize_bf16_fp6_lut_output(
            resources["golden_bf16"].data, m, n, resources["output_lut"].data)
        resources["source_fp6_packed"] = Resource(
            projected, (m // 2, n), 8, "packed_even_odd_m_fp6_lut_index")
        resources["nicolas_fp6"] = Resource(
            current_codes, (m // 2, n), 8, "packed_even_odd_m_fp6_lut_index")
        resources["golden_output_scales"] = Resource(
            row_major_scales, (m, n // 32), 8, "row_major_n_group_e8m0")
        resources["nicolas_output_scales"] = Resource(
            current_scales, (m, n // 32), 8, "row_major_n_group_e8m0")
    if fp6_quantized_specialization:
        source_codes = bytes(_array(source, name="C_proj_hw", ctype="uint8_t",
                                    dimensions="[64][128]", count=m * n // 2,
                                    maximum=255))
        current_codes, current_scales = quantize_bf16_fp6_lut_output(
            resources["golden_bf16"].data, m, n, resources["output_lut"].data)
        resources["source_fp6_packed"] = Resource(
            source_codes, (m // 2, n), 8, "packed_even_odd_m_fp6_lut_index")
        resources["nicolas_fp6"] = Resource(
            current_codes, (m // 2, n), 8, "packed_even_odd_m_fp6_lut_index")
        resources["nicolas_output_scales"] = Resource(
            current_scales, (m, n // 32), 8, "row_major_n_group_e8m0")
        source_scales = resources["output_scales"].data
        resources["golden_output_scales"] = Resource(
            bytes(source_scales[group * m + row]
                  for row in range(m) for group in range(n // 32)),
            (m, n // 32), 8, "row_major_n_group_e8m0")
    if (kernel.quant_output and kernel.datatype in {"FP8", "FP4"}) or vpu_spad_requant_x2:
        if kernel.datatype not in {"FP8", "FP4"}:
            raise ValueError("source quantized-output golden is qualified only for FP8/FP4")
        codes = bytes(_array(source, name="C_out", ctype="uint8_t",
                             dimensions="[MATMUL_M][MATMUL_N]", count=m * n,
                             maximum=255))
        groups = n // 32
        scales = resources["output_scales"].data
        row_major_scales = bytes(scales[group * m + row]
                                 for row in range(m) for group in range(groups))
        resources["golden_fp8"] = Resource(codes, (m, n), 8, "row_major_fp8_e4m3")
        resources["golden_output_scales"] = Resource(
            row_major_scales, (m, groups), 8, "row_major_n_group_e8m0")
        if kernel.datatype == "FP8":
            current_codes, current_scales = quantize_bf16_fp8_output(
                exact_bf16_x2(resources["golden_bf16"].data)
                if vpu_spad_requant_x2 else resources["golden_bf16"].data, m, n)
            resources["nicolas_fp8"] = Resource(
                current_codes, (m, n), 8, "row_major_fp8_e4m3")
        else:
            current_codes, current_scales = quantize_bf16_fp4_output(
                resources["golden_bf16"].data, m, n)
            resources["nicolas_fp4"] = Resource(
                current_codes, (m // 2, n), 8, "packed_even_odd_m_fp4_e2m1")
        resources["nicolas_output_scales"] = Resource(
            current_scales, (m, groups), 8, "row_major_n_group_e8m0")
    for name, resource in resources.items():
        resource.descriptor(name)
    return resources


def make_manifest(kernel: SourceGemm, resources: dict[str, Resource], *,
                  site_id: str, profile_sha256: str,
                  fp6_quantized_specialization: bool = False,
                  vpu_spad_requant_x2: bool = False,
                  source_derivation: dict | None = None) -> dict:
    if not site_id or len(profile_sha256) != 64:
        raise ValueError("payload needs a site ID and target profile digest")
    manifest = {
        "schema": SCHEMA, "site_id": site_id, "precision": kernel.datatype,
        "shape_mnk": list(kernel.shape), "tile_mnk": list(kernel.tile),
        "source_driver_sha256": _sha(kernel.driver.read_bytes()),
        "source_header_sha256": _sha(kernel.data_header.read_bytes()),
        "profile_sha256": profile_sha256,
        "origin": "radiance_source_header_specialization",
        "resources": {name: resource.descriptor(name)
                      for name, resource in sorted(resources.items())},
    }
    if source_derivation is not None:
        if kernel.quant_output or fp6_quantized_specialization or vpu_spad_requant_x2:
            raise ValueError("derived FP4 GEMM fixture requires plain BF16 output")
        manifest["origin"] = DERIVED_GEMM_FIXTURE_ORIGIN
        manifest["source_derivation"] = source_derivation
        validate_derived_gemm_fixture(manifest)
    if kernel.quant_output:
        manifest["output_format"] = {"FP8": "fp8_e4m3", "FP4": "fp4_e2m1",
                                     "FP6": "fp6_e3m2"}[kernel.datatype]
        manifest["output_oracle"] = {
            "FP8": "nicolas_mxquant_po2_rne_v1",
            "FP4": "nicolas_fp4_e3m1_e2m1_v1",
            "FP6": "nicolas_fp6_e3m2_lut_po2_rne_v1",
        }[kernel.datatype]
        manifest["source_quant_golden_convention"] = "source_header"
        if kernel.datatype == "FP4":
            manifest["source_quant_header_format"] = "fp8_e4m3"
        if kernel.datatype == "FP6":
            manifest["source_quant_header_format"] = "packed_fp6_lut_index"
    if fp6_quantized_specialization:
        if kernel.datatype != "FP6" or kernel.quant_output:
            raise ValueError("FP6 output specialization requires a fullout source driver")
        manifest.update({
            "output_format": "fp6_e3m2",
            "output_oracle": "nicolas_fp6_e3m2_lut_po2_rne_v1",
            "source_quant_golden_convention": "source_header_projected",
            "source_quant_header_format": "packed_fp6_lut_index",
            "output_specialization": "bf16_fullout_to_fp6_lut_quantized",
        })
    if vpu_spad_requant_x2:
        if (kernel.datatype != "FP8" or kernel.quant_output or
                not vpu_requant_shape_is_legal(kernel.shape, kernel.tile)):
            raise ValueError("VPU/SPAD x2 output specialization requires single-tile FP8 fullout source")
        manifest.update({
            "output_format": "fp8_e4m3",
            "output_oracle": "nicolas_vpu_x2_spad_requant_fp8_v1",
            "source_quant_golden_convention": "source_header_unscaled",
            "output_specialization": "matrix_vpu_x2_spad_requant_fp8",
        })
    return manifest


def write_bundle(directory: Path, kernel: SourceGemm, *, site_id: str,
                 profile_sha256: str,
                 fp6_quantized_specialization: bool = False,
                 vpu_spad_requant_x2: bool = False,
                 source_derivation: dict | None = None) -> dict:
    resources = read_source_payload(
        kernel, fp6_quantized_specialization=fp6_quantized_specialization,
        vpu_spad_requant_x2=vpu_spad_requant_x2)
    manifest = make_manifest(kernel, resources, site_id=site_id,
                             profile_sha256=profile_sha256,
                             fp6_quantized_specialization=fp6_quantized_specialization,
                             vpu_spad_requant_x2=vpu_spad_requant_x2,
                             source_derivation=source_derivation)
    directory.mkdir(parents=True, exist_ok=False)
    for name, resource in resources.items():
        (directory / f"{name}.bin").write_bytes(resource.data)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_bundle(directory: Path) -> tuple[dict, dict[str, bytes]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("resources"), dict):
        raise ValueError("unknown MX payload bundle schema")
    if manifest.get("origin") == ATTENTION_QK_CANDIDATE_ORIGIN:
        validate_attention_qk_candidate(manifest)
    elif manifest.get("origin") == ATTENTION_PV_PROXY_ORIGIN:
        validate_attention_pv_proxy(manifest)
    elif manifest.get("origin") == DERIVED_GEMM_FIXTURE_ORIGIN:
        validate_derived_gemm_fixture(manifest)
    elif manifest.get("origin") == TARGET_MESH_REFERENCE_ORIGIN:
        validate_target_mesh_reference(manifest)
    elif "source_derivation" in manifest:
        raise ValueError("derived MX payload must declare its candidate origin")
    precision = manifest.get("precision")
    shape = manifest.get("shape_mnk")
    tile = manifest.get("tile_mnk")
    if (precision not in {"FP8", "FP4", "FP6"} or
            not isinstance(shape, list) or len(shape) != 3 or
            not isinstance(tile, list) or len(tile) != 3 or
            any(type(d) is not int or d <= 0 for d in shape + tile)):
        raise ValueError("MX payload precision, shape, or tile is malformed")
    m, n, k = shape
    if m % 2 or n % 32 or k % 32:
        raise ValueError("MX payload shape is not legal for source packing")
    packed = precision != "FP8"
    expected = {
        "activation": ((m // 2 if packed else m, k), 8,
                       "packed_even_odd_m_nibbles" if packed else "row_major_codes"),
        "weight": ((k, n // 2 if packed else n), 8,
                   "packed_even_odd_n_nibbles" if packed else "row_major_codes"),
        "activation_scales": ((k // 32, m), 8, "k_group_row_e8m0"),
        "weight_scales": ((k // 32, n), 8, "k_group_column_e8m0"),
        "output_scales": ((n // 32, m), 8, "n_group_row_e8m0"),
        "golden_bf16": ((m, n), 16, "row_major_bf16"),
    }
    if precision == "FP6":
        expected.update({
            "activation_lut": ((64, 3), 32, "row_pair_lut_6bit"),
            "weight_lut": ((64, 3), 32, "column_pair_lut_6bit"),
            "output_lut": ((64, 3), 32, "output_pair_lut_6bit"),
        })
    output_format = manifest.get("output_format")
    if output_format is not None:
        valid = ((precision == "FP8" and output_format == "fp8_e4m3" and
                  manifest.get("output_oracle") in {"nicolas_mxquant_po2_rne_v1",
                                                    "nicolas_vpu_x2_spad_requant_fp8_v1"}) or
                 (precision == "FP4" and output_format == "fp4_e2m1" and
                  manifest.get("output_oracle") == "nicolas_fp4_e3m1_e2m1_v1") or
                 (precision == "FP6" and output_format == "fp6_e3m2" and
                  manifest.get("output_oracle") == "nicolas_fp6_e3m2_lut_po2_rne_v1" and
                  manifest.get("output_specialization") in {
                      None, "bf16_fullout_to_fp6_lut_quantized"}))
        expected_convention = ("source_header_projected" if precision == "FP6" and
                               manifest.get("output_specialization") ==
                               "bf16_fullout_to_fp6_lut_quantized" else
                               "source_header_unscaled" if manifest.get("output_specialization") ==
                               "matrix_vpu_x2_spad_requant_fp8" else "source_header")
        if (not valid or manifest.get("source_quant_golden_convention") != expected_convention or
                (manifest.get("output_specialization") == "matrix_vpu_x2_spad_requant_fp8" and
                 manifest.get("output_oracle") != "nicolas_vpu_x2_spad_requant_fp8_v1") or
                (manifest.get("output_oracle") == "nicolas_vpu_x2_spad_requant_fp8_v1" and
                 (precision != "FP8" or manifest.get("output_specialization") !=
                  "matrix_vpu_x2_spad_requant_fp8" or
                  not vpu_requant_shape_is_legal(tuple(shape), tuple(tile)))) or
                (precision == "FP4" and
                 manifest.get("source_quant_header_format") != "fp8_e4m3") or
                (precision == "FP6" and
                 manifest.get("source_quant_header_format") != "packed_fp6_lut_index")):
            raise ValueError("source quantized-output golden has an unsupported format")
        expected.update({
            "golden_output_scales": ((m, n // 32), 8, "row_major_n_group_e8m0"),
            "nicolas_output_scales": ((m, n // 32), 8, "row_major_n_group_e8m0"),
        })
        if precision != "FP6":
            expected["golden_fp8"] = ((m, n), 8, "row_major_fp8_e4m3")
        if precision == "FP8":
            expected["nicolas_fp8"] = ((m, n), 8, "row_major_fp8_e4m3")
        else:
            if precision == "FP4":
                expected["nicolas_fp4"] = ((m // 2, n), 8, "packed_even_odd_m_fp4_e2m1")
            else:
                expected["source_fp6_packed"] = ((m // 2, n), 8, "packed_even_odd_m_fp6_lut_index")
                expected["nicolas_fp6"] = ((m // 2, n), 8, "packed_even_odd_m_fp6_lut_index")
    if set(manifest["resources"]) != set(expected):
        raise ValueError("MX payload resource set differs from precision requirements")
    resources = {}
    for name, descriptor in manifest["resources"].items():
        if (name not in {"activation", "weight", "activation_scales", "weight_scales",
                         "output_scales", "activation_lut", "weight_lut", "output_lut",
                         "golden_bf16", "golden_fp8", "golden_output_scales",
                         "nicolas_fp8", "nicolas_fp4", "nicolas_fp6",
                         "source_fp6_packed", "nicolas_output_scales"} or
                descriptor.get("file") != f"{name}.bin"):
            raise ValueError("MX payload bundle has an unknown resource or unsafe file")
        data = (directory / descriptor["file"]).read_bytes()
        if descriptor.get("sha256") != _sha(data) or descriptor.get("bytes") != len(data):
            raise ValueError(f"MX payload resource {name} digest or size differs")
        spec = expected[name]
        if (descriptor.get("shape") != list(spec[0]) or
                descriptor.get("element_bits") != spec[1] or
                descriptor.get("layout") != spec[2]):
            raise ValueError(f"MX payload resource {name} storage contract differs")
        Resource(data, *spec).descriptor(name)
        resources[name] = data
    if output_format in {"fp8_e4m3", "fp4_e2m1", "fp6_e3m2"}:
        bf16_basis = (exact_bf16_x2(resources["golden_bf16"])
                      if manifest.get("output_specialization") == "matrix_vpu_x2_spad_requant_fp8"
                      else resources["golden_bf16"])
        codes, scales = (quantize_bf16_fp8_output(bf16_basis, m, n)
                         if output_format == "fp8_e4m3" else
                         quantize_bf16_fp4_output(resources["golden_bf16"], m, n)
                         if output_format == "fp4_e2m1" else
                         quantize_bf16_fp6_lut_output(
                             resources["golden_bf16"], m, n, resources["output_lut"]))
        name = {"fp8_e4m3": "nicolas_fp8", "fp4_e2m1": "nicolas_fp4",
                "fp6_e3m2": "nicolas_fp6"}[output_format]
        if (resources[name] != codes or resources["nicolas_output_scales"] != scales):
            raise ValueError("Nicolas output oracle differs from bound BF16 source golden")
    return manifest, resources
