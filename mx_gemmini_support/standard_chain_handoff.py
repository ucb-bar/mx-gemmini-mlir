"""Bind a current portable model2MLIR two-matmul graph to MX source bytes.

The f32 graph establishes operation order and tensor shapes. A separate
checked Nicolas source/header pair supplies quantized operands and goldens.
No quantized values are inferred from PyTorch example inputs.
"""

from __future__ import annotations

import hashlib
import re

from .target_profile import profile_sha256, require_compute


_FORMATS = {"FP8": ("fp8_e4m3", "direct", "mxfp8", 8),
            "FP4": ("fp4_e2m1", "direct", "mxfp4", 0),
            "FP6": ("fp6_e3m2", "lut", "mxfp6", 4)}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def portable_chain_shapes(mlir_text: str) -> tuple[tuple[int, int, int],
                                                     tuple[int, int, int]]:
    """Accept only a two-site, directly connected f32 matmul function."""
    from xdsl.context import Context
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import Builtin, Float32Type, TensorType
    from xdsl.dialects.func import Func, FuncOp
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.tensor import Tensor
    from xdsl.parser import Parser

    context = Context(allow_unregistered=False)
    for dialect in (Builtin, Func, Tensor, Arith, Linalg):
        context.load_dialect(dialect)
    try:
        module = Parser(context, mlir_text).parse_module()
    except Exception as error:
        raise ValueError("portable MX chain cannot parse selected frontend MLIR") from error
    if getattr(module.attributes.get("prov.level"), "data", None) != "linalg-on-tensors":
        raise ValueError("portable MX chain needs linalg-on-tensors provenance")
    functions = list(module.ops)
    if (len(functions) != 1 or not isinstance(functions[0], FuncOp) or
            functions[0].is_declaration or len(functions[0].body.blocks) != 1):
        raise ValueError("portable MX chain needs one defined function")
    block = functions[0].body.blocks[0]
    ops = list(block.ops)
    if (len(block.args) != 3 or [op.name for op in ops] != [
            "tensor.empty", "arith.constant", "linalg.fill", "linalg.matmul",
            "tensor.empty", "arith.constant", "linalg.fill", "linalg.matmul",
            "func.return"]):
        raise ValueError("portable MX chain needs exactly two connected matmuls")
    shapes = []
    for index, base in enumerate((0, 4)):
        empty, zero, fill, matmul = ops[base:base + 4]
        left = block.args[0] if index == 0 else ops[3].results[0]
        right = block.args[index + 1]
        if (len(empty.results) != 1 or len(zero.results) != 1 or
                list(fill.operands) != [zero.results[0], empty.results[0]] or
                len(fill.results) != 1 or
                list(matmul.operands) != [left, right, fill.results[0]] or
                len(matmul.results) != 1 or
                getattr(getattr(zero.properties.get("value"), "value", None),
                        "data", None) != 0.0):
            raise ValueError("portable MX chain initializer or SSA edge differs")
        expected = {"prov.region_id": f"matmul_{index}",
                    "prov.op": "matmul", "prov.family": "contraction",
                    "prov.aten": "aten.mm.default", "prov.orig_dtype": "float32"}
        if any(getattr(matmul.attributes.get(name), "data", None) != value
               for name, value in expected.items()):
            raise ValueError("portable MX chain matmul provenance differs")
        types = [left.type, right.type, empty.results[0].type,
                 fill.results[0].type, matmul.results[0].type]
        if any(not isinstance(t, TensorType) or
               not isinstance(t.element_type, Float32Type) or
               len(t.get_shape()) != 2 for t in types):
            raise ValueError("portable MX chain needs rank-two f32 tensors")
        (m, k), (other_k, n), *output_shapes = [tuple(t.get_shape()) for t in types]
        if (any(type(dim) is not int or dim <= 0 for dim in (m, n, k, other_k)) or
                k != other_k or any(shape != (m, n) for shape in output_shapes)):
            raise ValueError("portable MX chain tensor shapes differ")
        shapes.append((m, n, k))
    if (list(ops[8].operands) != list(ops[7].results) or
            shapes[0][0] != shapes[1][0] or shapes[0][1] != shapes[1][2]):
        raise ValueError("portable MX chain return or inter-site shape differs")
    return shapes[0], shapes[1]


def portable_chain_manifest(mlir_text: str, profile: dict, *, precision: str,
                            source_driver_sha256: str,
                            source_header_sha256: str) -> dict:
    """Construct the explicit binding for a legal MX mode and pinned source."""
    if precision not in _FORMATS or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in (source_driver_sha256, source_header_sha256)):
        raise ValueError("portable MX chain needs a precision and pinned source hashes")
    fmt, projection, quant_format, pe_mode = _FORMATS[precision]
    require_compute(profile, fmt, fmt, pe_mode=pe_mode,
                    activation_projection=projection,
                    weight_projection=projection)
    shapes = portable_chain_shapes(mlir_text)
    return {
        "schema": "mx_gemmini.portable_chain_binding.v1",
        "precision": precision,
        "profile_sha256": profile_sha256(profile),
        "source_mlir_sha256": hashlib.sha256(mlir_text.encode()).hexdigest(),
        "source_driver_sha256": source_driver_sha256,
        "source_header_sha256": source_header_sha256,
        "sites": [{"site_id": name, "status": "quantized", "format": quant_format,
                   "shape": list(shape)}
                  for name, shape in zip(("functional:matmul",
                                          "functional:matmul_1"), shapes)],
    }


def validate_portable_chain(mlir_text: str, manifest: dict, profile: dict,
                            *, precision: str, source_driver_sha256: str,
                            source_header_sha256: str) -> tuple[tuple[int, int, int],
                                                                 tuple[int, int, int]]:
    """Fail closed if any source, shape, precision, or profile binding changed."""
    expected = portable_chain_manifest(
        mlir_text, profile, precision=precision,
        source_driver_sha256=source_driver_sha256,
        source_header_sha256=source_header_sha256)
    if manifest != expected:
        raise ValueError("portable MX chain manifest differs from current capture or source")
    return tuple(tuple(site["shape"]) for site in expected["sites"])
