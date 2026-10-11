"""Bind a portable model2MLIR matmul to an explicit MX source payload.

Current model2MLIR emits standard linalg for this PyTorch graph. Its f32
matmul establishes the operation and shape; the checked source bundle owns
the MX codes, E8M0 scales, and LUTs. This binder never infers quantization
from floating-point example values.
"""

from __future__ import annotations

import hashlib
import json

from .source_payload import manifest_sha256
from .target_profile import profile_sha256
from .verify_profile_ir import verify_ir


_FORMATS = {"FP8": ("fp8_e4m3", "direct"),
            "FP4": ("fp4_e2m1", "direct"),
            "FP6": ("fp6_e3m2", "lut")}


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _portable_shape(mlir_text: str) -> tuple[int, int, int]:
    """Accept only the exact single-matmul data flow emitted by model2MLIR."""
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
        raise ValueError("portable MX binding cannot parse the selected frontend MLIR") from error
    if getattr(module.attributes.get("prov.level"), "data", None) != "linalg-on-tensors":
        raise ValueError("portable MX binding needs linalg-on-tensors provenance")
    functions = list(module.ops)
    if len(functions) != 1 or not isinstance(functions[0], FuncOp) or \
            functions[0].is_declaration or len(functions[0].body.blocks) != 1:
        raise ValueError("portable MX binding needs one defined matmul function")
    block = functions[0].body.blocks[0]
    operations = list(block.ops)
    if ([op.name for op in operations] !=
            ["tensor.empty", "arith.constant", "linalg.fill", "linalg.matmul",
             "func.return"] or len(block.args) != 2):
        raise ValueError("portable MX binding needs one fill, matmul, and return")
    empty, zero, fill, matmul, returned = operations
    if (len(empty.results) != 1 or len(zero.results) != 1 or
            list(fill.operands) != [zero.results[0], empty.results[0]] or
            len(fill.results) != 1 or
            list(matmul.operands) != [block.args[0], block.args[1], fill.results[0]] or
            len(matmul.results) != 1 or
            list(returned.operands) != list(matmul.results) or
            getattr(getattr(zero.properties.get("value"), "value", None), "data", None) != 0.0):
        raise ValueError("portable MX matmul has a different data flow or initializer")
    for name, value in {"prov.op": "matmul", "prov.family": "contraction",
                        "prov.aten": "aten.mm.default", "prov.orig_dtype": "float32"}.items():
        if getattr(matmul.attributes.get(name), "data", None) != value:
            raise ValueError(f"portable MX matmul lacks {name} provenance")
    types = [arg.type for arg in block.args] + [empty.results[0].type,
                                                fill.results[0].type,
                                                matmul.results[0].type]
    if any(not isinstance(t, TensorType) or not isinstance(t.element_type, Float32Type)
           or len(t.get_shape()) != 2 for t in types):
        raise ValueError("portable MX matmul needs rank-two f32 tensors")
    (m, k), (weight_k, n), empty_shape, fill_shape, output_shape = [
        tuple(t.get_shape()) for t in types]
    if (not all(type(dim) is int and dim > 0 for dim in (m, k, weight_k, n)) or
            k != weight_k or any(shape != (m, n) for shape in
                                 (empty_shape, fill_shape, output_shape))):
        raise ValueError("portable MX matmul shape or K dimension differs")
    return m, n, k


def bind_standard_matmul(mlir_text: str, profile: dict,
                         source_manifest: dict) -> tuple[str, dict]:
    """Return a verified MX plan and an auditable explicit binding receipt."""
    shape = _portable_shape(mlir_text)
    if (source_manifest.get("schema") != "mx_gemmini.source_payload.v1" or
            source_manifest.get("site_id") != "functional:matmul" or
            source_manifest.get("shape_mnk") != list(shape) or
            source_manifest.get("precision") not in _FORMATS or
            source_manifest.get("profile_sha256") != profile_sha256(profile)):
        raise ValueError("portable matmul differs from the explicit MX source payload")
    fmt, projection = _FORMATS[source_manifest["precision"]]
    cells = [cell for cell in profile["legal_compute"] if
             cell["activation_format"] == cell["weight_format"] == fmt and
             cell["activation_projection"] == cell["weight_projection"] == projection]
    if len(cells) != 1:
        raise ValueError("portable matmul has no unique legal MX precision mode")
    source_sha = hashlib.sha256(mlir_text.encode()).hexdigest()
    payload_sha = manifest_sha256(source_manifest)
    policy = {"schema": "mx_gemmini.explicit_standard_matmul_policy.v1",
              "site_id": "functional:matmul", "precision": source_manifest["precision"],
              "profile_sha256": profile_sha256(profile), "payload_manifest_sha256": payload_sha}
    contract = {"schema": "mx_gemmini.portable_matmul_binding.v1",
                "source_mlir_sha256": source_sha, "shape_mnk": list(shape),
                "site_id": "functional:matmul"}
    census = {"schema": "mx_gemmini.portable_matmul_census.v1", "contract": contract,
              "policy_sha256": _digest(policy)}
    contract_sha, policy_sha, census_sha = map(_digest, (contract, policy, census))
    common = (f'site_id = "functional:matmul", contract_sha256 = "{contract_sha}", '
              f'policy_sha256 = "{policy_sha}", manifest_sha256 = "{census_sha}", '
              f'profile_sha256 = "{profile_sha256(profile)}"')
    codes, scales, result = "tensor<?x?xi8>", "tensor<?x?xi8>", "tensor<?x?xbf16>"
    selected = (f'activation_format = "{fmt}", activation_projection = "{projection}", '
                f'weight_format = "{fmt}", weight_projection = "{projection}", '
                f'pe_mode = {cells[0]["pe_mode"]} : i32')
    rendered = (
        f'builtin.module attributes {{mx.contract_sha256 = "{contract_sha}", '
        f'mx.policy_sha256 = "{policy_sha}", '
        f'prov.quantization_manifest_sha256 = "{census_sha}", '
        f'prov.quantization = "explicit:mx_gemmini", '
        f'mx.source_mlir_sha256 = "{source_sha}", '
        f'mx.profile_sha256 = "{profile_sha256(profile)}", '
        f'mx.frontend_binding_schema = "mx_gemmini.portable_matmul_binding.v1"}} {{\n'
        f'  func.func @site_0(%a: {codes}, %as: {scales}, %b: {codes}, %bs: {scales}) -> {result} {{\n'
        f'    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) '
        f'{{{common}, {selected}}} : ({codes}, {scales}, {codes}, {scales}) -> {result}\n'
        f'    %out = "mx_gemmini.readout_bf16"(%acc) '
        f'{{{common}}} : ({result}) -> {result}\n'
        f'    func.return %out : {result}\n  }}\n}}\n')
    verify_ir(rendered, profile)
    return rendered, {"schema": "mx_gemmini.portable_matmul_binding.v1",
                      "source_mlir_sha256": source_sha, "shape_mnk": list(shape),
                      "profile_sha256": profile_sha256(profile),
                      "payload_manifest_sha256": payload_sha,
                      "contract_sha256": contract_sha, "policy_sha256": policy_sha,
                      "census_sha256": census_sha, "mode": cells[0]}
