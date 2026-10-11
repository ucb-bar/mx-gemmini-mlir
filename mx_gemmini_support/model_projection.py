"""Bind an actual model2MLIR rank-two site to an FP8 MX output stripe.

The caller supplies the site's floating operands. The frontend graph contains
their types and provenance, but not their runtime values. This module records
that distinction and uses the same packed operand and physical command paths
as source-qualified contractions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from .operands import pack_direct_operands
from .source_payload import (MODEL2MLIR_PROJECTION_GOLDEN_SHA256,
                             MODEL2MLIR_PROJECTION_ORIGIN, Resource, load_bundle,
                             validate_model2mlir_projection)
from .target_profile import profile_sha256


MODEL_MATH_SHA256 = "e91f2f83ff58c8d6a4c5c1161c9df4a63fc34e7052528a0b87f4dc27116e7b60"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def portable_projection_shape(mlir_text: str) -> tuple[int, int, int]:
    """Verify the current model2MLIR portable single-matmul SSA structure."""
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
        module.verify()
    except Exception as error:
        raise ValueError("projection capture is not valid portable MLIR") from error
    if getattr(module.attributes.get("prov.level"), "data", None) != "linalg-on-tensors":
        raise ValueError("projection capture lacks portable provenance")
    functions = list(module.ops)
    if len(functions) != 1 or not isinstance(functions[0], FuncOp):
        raise ValueError("projection capture needs one function")
    block = functions[0].body.block
    ops = list(block.ops)
    if (len(block.args) != 2 or [op.name for op in ops] !=
            ["tensor.empty", "arith.constant", "linalg.fill", "linalg.matmul",
             "func.return"]):
        raise ValueError("projection capture needs one direct matmul and return")
    empty, zero, fill, matmul, returned = ops
    if (list(fill.operands) != [zero.results[0], empty.results[0]] or
            list(matmul.operands) != [block.args[0], block.args[1], fill.results[0]] or
            list(returned.operands) != list(matmul.results) or
            getattr(getattr(zero.properties.get("value"), "value", None), "data", None) != 0.0 or
            any(getattr(matmul.attributes.get(key), "data", None) != value
                for key, value in {"prov.op": "matmul", "prov.family": "contraction",
                                   "prov.aten": "aten.mm.default"}.items())):
        raise ValueError("projection capture's matmul dataflow differs")
    types = [block.args[0].type, block.args[1].type,
             empty.results[0].type, fill.results[0].type, matmul.results[0].type]
    if any(not isinstance(t, TensorType) or
           not isinstance(t.element_type, Float32Type) or len(t.get_shape()) != 2
           for t in types):
        raise ValueError("projection capture needs rank-two f32 tensors")
    (m, k), (rhs_k, n), *outputs = [tuple(t.get_shape()) for t in types]
    if rhs_k != k or any(shape != (m, n) for shape in outputs):
        raise ValueError("projection capture's tensor shapes differ")
    return m, n, k


def write_model_projection_bundle(directory: Path, *, worklist: dict, site: dict,
                                  activation, weight, row_start: int,
                                  row_count: int, column_start: int,
                                  column_count: int, tile_k: int,
                                  profile: dict, reference_root: Path) -> dict:
    """Quantize one full-K model projection stripe and write checked resources.

    `activation` and `weight` are complete f32 matrices for the selected
    model2MLIR matmul. A short M axis is zero-padded to the target mesh; the
    exact row count remains in the provenance so callers can discard padding.
    """
    import numpy as np
    import torch
    from mxq.nn.operand_capture import functional_contraction_operands

    if (worklist.get("schema") != "mx_gemmini.model2mlir_contraction_worklist.v1" or
            site not in worklist.get("rank2_matmuls", []) or
            site.get("source_op") != "matmul" or
            site.get("element_type") != "f32" or
            worklist.get("profile_sha256") != profile_sha256(profile)):
        raise ValueError("projection site is absent from the selected model2MLIR worklist")
    m, n, k = site["shape_mnk"]
    if (not isinstance(activation, np.ndarray) or not isinstance(weight, np.ndarray) or
            activation.dtype != np.float32 or weight.dtype != np.float32 or
            activation.shape != (m, k) or weight.shape != (k, n) or
            not np.isfinite(activation).all() or not np.isfinite(weight).all()):
        raise ValueError("model2MLIR projection needs finite full-size f32 operands")
    if (any(type(value) is not int for value in (row_start, row_count,
                                                column_start, column_count, tile_k)) or
            row_start < 0 or row_count <= 0 or row_start + row_count > m or
            column_start < 0 or column_count < 32 or column_count % 32 or
            column_start + column_count > n or
            column_start % 32 or k % 32 or tile_k < 32 or tile_k % 32 or k % tile_k):
        raise ValueError("projection stripe or K tile is outside complete FP8 MX bounds")
    mesh = profile["geometry"]["mesh_columns"]
    if mesh != 16 or profile["geometry"]["mesh_rows"] != 16:
        raise ValueError("model projection slice currently needs the qualified DIM16 profile")
    padded_m = ((row_count + 15) // 16) * 16
    raw_a = np.ascontiguousarray(activation[row_start:row_start + row_count])
    raw_b = np.ascontiguousarray(weight[:, column_start:column_start + column_count])
    a = torch.zeros((padded_m, k), dtype=torch.bfloat16)
    a[:row_count] = torch.from_numpy(raw_a).to(torch.bfloat16)
    b = torch.from_numpy(raw_b).to(torch.bfloat16)
    encoded = functional_contraction_operands(a, b, "mxfp8")
    packed = pack_direct_operands(
        "mxfp8", encoded.activation_codes.tolist(), encoded.weight_codes.tolist())
    a_scales = encoded.activation_scales.T.contiguous().numpy().tobytes()
    b_scales = encoded.weight_scales.T.contiguous().numpy().tobytes()
    resources = {
        "activation": Resource(packed.activation_bytes, (padded_m, k), 8, "row_major_codes"),
        "weight": Resource(packed.weight_bytes, (k, column_count), 8, "row_major_codes"),
        "activation_scales": Resource(a_scales, (k // 32, padded_m), 8, "k_group_row_e8m0"),
        "weight_scales": Resource(b_scales, (k // 32, column_count), 8, "k_group_column_e8m0"),
        "output_scales": Resource(bytes(padded_m * column_count // 32),
                                  (column_count // 32, padded_m), 8, "n_group_row_e8m0"),
    }
    model_cpp = reference_root / "lib/golden/mx_golden.cpp"
    model_math = reference_root / "lib/golden/mx_fp_math.h"
    if _sha(model_cpp.read_bytes()) != MODEL2MLIR_PROJECTION_GOLDEN_SHA256 or \
            _sha(model_math.read_bytes()) != MODEL_MATH_SHA256:
        raise ValueError("independent MX hardware arithmetic model differs from pinned source")
    if directory.exists():
        raise ValueError(f"refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    for name, resource in resources.items():
        (directory / f"{name}.bin").write_bytes(resource.data)
    with TemporaryDirectory(prefix="model-projection-reference-", dir=directory.parent) as temp:
        executable = Path(temp) / "mx_golden"
        subprocess.run(["g++", "-O2", "-std=c++17", "-I", str(model_cpp.parent),
                        str(model_cpp), "-o", str(executable)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        golden_path = directory / "golden_bf16.bin"
        subprocess.run([str(executable), str(padded_m), str(column_count), str(k),
                        *(str(directory / f"{name}.bin") for name in
                          ("activation", "weight", "activation_scales", "weight_scales")),
                        str(golden_path), "0"], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    resources["golden_bf16"] = Resource(golden_path.read_bytes(),
                                         (padded_m, column_count), 16, "row_major_bf16")
    manifest = {
        "schema": "mx_gemmini.source_payload.v1",
        "site_id": "functional:matmul", "precision": "FP8",
        "shape_mnk": [padded_m, column_count, k],
        "tile_mnk": [padded_m, 32, tile_k],
        "profile_sha256": profile_sha256(profile),
        "origin": MODEL2MLIR_PROJECTION_ORIGIN,
        "model2mlir_projection": {
            "source_mlir_sha256": worklist["source_mlir_sha256"],
            "region_id": site["region_id"], "fqn": site["fqn"],
            "source_op": site["source_op"], "source_shape_mnk": site["shape_mnk"],
            "row_start": row_start, "row_count": row_count,
            "column_start": column_start, "column_count": column_count,
            "input_sha256": _sha(raw_a.tobytes()),
            "weight_sha256": _sha(raw_b.tobytes()),
            "golden_model_sha256": MODEL2MLIR_PROJECTION_GOLDEN_SHA256,
        },
        "resources": {name: resource.descriptor(name)
                      for name, resource in sorted(resources.items())},
    }
    validate_model2mlir_projection(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    load_bundle(directory)
    return manifest
