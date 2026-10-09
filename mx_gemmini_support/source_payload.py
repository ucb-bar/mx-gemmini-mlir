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

from .source_fp6 import _array, _bytes, read_source_fp6_payload
from .source_gemm import SourceGemm
from .quant_reference import quantize_bf16_fp8_output


SCHEMA = "mx_gemmini.source_payload.v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_sha256(manifest: dict) -> str:
    """Hash the canonical manifest, excluding any filesystem location."""
    return _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())


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


def read_source_payload(kernel: SourceGemm) -> dict[str, Resource]:
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
        a_name = "A_in_hw" if packed else "A_in"
        a_shape = (m // 2, k) if packed else (m, k)
        a_decl = "[MATMUL_M / 2][MATMUL_K]" if packed else "[MATMUL_M][MATMUL_K]"
        b_shape = (k, n // 2) if packed else (k, n)
        b_decl = "[MATMUL_K][MATMUL_N / 2]" if packed else "[MATMUL_K][MATMUL_N]"
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
    if kernel.quant_output:
        if kernel.datatype != "FP8":
            raise ValueError("source quantized-output golden is qualified only for FP8")
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
        current_codes, current_scales = quantize_bf16_fp8_output(
            resources["golden_bf16"].data, m, n)
        resources["nicolas_fp8"] = Resource(
            current_codes, (m, n), 8, "row_major_fp8_e4m3")
        resources["nicolas_output_scales"] = Resource(
            current_scales, (m, groups), 8, "row_major_n_group_e8m0")
    for name, resource in resources.items():
        resource.descriptor(name)
    return resources


def make_manifest(kernel: SourceGemm, resources: dict[str, Resource], *,
                  site_id: str, profile_sha256: str) -> dict:
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
    if kernel.quant_output:
        manifest["output_format"] = "fp8_e4m3"
        manifest["output_oracle"] = "nicolas_mxquant_po2_rne_v1"
        manifest["source_quant_golden_convention"] = "source_header"
    return manifest


def write_bundle(directory: Path, kernel: SourceGemm, *, site_id: str,
                 profile_sha256: str) -> dict:
    resources = read_source_payload(kernel)
    manifest = make_manifest(kernel, resources, site_id=site_id,
                             profile_sha256=profile_sha256)
    directory.mkdir(parents=True, exist_ok=False)
    for name, resource in resources.items():
        (directory / f"{name}.bin").write_bytes(resource.data)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_bundle(directory: Path) -> tuple[dict, dict[str, bytes]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or not isinstance(manifest.get("resources"), dict):
        raise ValueError("unknown MX payload bundle schema")
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
        if (precision != "FP8" or output_format != "fp8_e4m3" or
                manifest.get("output_oracle") != "nicolas_mxquant_po2_rne_v1" or
                manifest.get("source_quant_golden_convention") != "source_header"):
            raise ValueError("source quantized-output golden has an unsupported format")
        expected.update({
            "golden_fp8": ((m, n), 8, "row_major_fp8_e4m3"),
            "golden_output_scales": ((m, n // 32), 8, "row_major_n_group_e8m0"),
            "nicolas_fp8": ((m, n), 8, "row_major_fp8_e4m3"),
            "nicolas_output_scales": ((m, n // 32), 8, "row_major_n_group_e8m0"),
        })
    if set(manifest["resources"]) != set(expected):
        raise ValueError("MX payload resource set differs from precision requirements")
    resources = {}
    for name, descriptor in manifest["resources"].items():
        if (name not in {"activation", "weight", "activation_scales", "weight_scales",
                         "output_scales", "activation_lut", "weight_lut", "output_lut",
                         "golden_bf16", "golden_fp8", "golden_output_scales",
                         "nicolas_fp8", "nicolas_output_scales"} or
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
    if output_format == "fp8_e4m3":
        codes, scales = quantize_bf16_fp8_output(resources["golden_bf16"], m, n)
        if (resources["nicolas_fp8"] != codes or
                resources["nicolas_output_scales"] != scales):
            raise ValueError("Nicolas output oracle differs from bound BF16 source golden")
    return manifest, resources
