"""Strict model2MLIR contraction import for a future whole-program MX lowerer.

The result describes work, not an executable placement. In particular, f32
model activations have no implied MX codes, scales, or output conversion.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib

from .target_profile import profile_sha256


def _text(op, name: str) -> str | None:
    value = op.attributes.get(name)
    data = getattr(value, "data", None)
    return data if isinstance(data, str) else None


def build_model2mlir_worklist(mlir_text: str, profile: dict) -> dict:
    """Import one verified frontend function and enumerate every contraction.

    The exact source digest and profile are part of the worklist. No operation
    is silently assigned to MX; callers must later provide explicit precision
    and data bindings and lower all remaining operations into an executable
    lane before claiming a complete program.
    """
    from xdsl.context import Context
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import BFloat16Type, Builtin, Float32Type, TensorType
    from xdsl.dialects.func import Func, FuncOp
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.math import Math
    from xdsl.dialects.scf import Scf
    from xdsl.dialects.tensor import Tensor
    from xdsl.parser import Parser

    context = Context(allow_unregistered=False)
    for dialect in (Builtin, Func, Tensor, Arith, Linalg, Math, Scf):
        context.load_dialect(dialect)
    try:
        module = Parser(context, mlir_text).parse_module()
        module.verify()
    except Exception as error:
        raise ValueError("model2MLIR module is not valid supported portable MLIR") from error
    if getattr(module.attributes.get("prov.level"), "data", None) != "linalg-on-tensors":
        raise ValueError("model2MLIR worklist needs linalg-on-tensors provenance")
    functions = list(module.ops)
    if (len(functions) != 1 or not isinstance(functions[0], FuncOp) or
            functions[0].is_declaration or len(functions[0].body.blocks) != 1):
        raise ValueError("model2MLIR worklist needs one defined function")

    groups: OrderedDict[tuple[str, str], dict] = OrderedDict()
    matmuls = []
    for operation in functions[0].walk():
        if _text(operation, "prov.family") == "contraction":
            fqn = _text(operation, "prov.fqn")
            region = _text(operation, "prov.region_id")
            kind = _text(operation, "prov.op")
            if not region or not kind:
                raise ValueError("model2MLIR contraction lacks stable source provenance")
            group = groups.setdefault((region, kind), {"fqns": set(), "operations": []})
            if fqn:
                group["fqns"].add(fqn)
            group["operations"].append(operation.name)
        if operation.name != "linalg.matmul":
            continue
        if (_text(operation, "prov.family") != "contraction" or
                _text(operation, "prov.region_id") is None or
                _text(operation, "prov.fqn") is None):
            raise ValueError("model2MLIR matmul lacks contraction provenance")
        if len(operation.operands) != 3 or len(operation.results) != 1:
            raise ValueError("model2MLIR matmul operand convention changed")
        types = [value.type for value in operation.operands] + [operation.results[0].type]
        if any(not isinstance(t, TensorType) or
               not isinstance(t.element_type, (Float32Type, BFloat16Type)) or
               len(t.get_shape()) != 2 for t in types):
            raise ValueError("model2MLIR matmul needs static rank-two floating tensors")
        dtypes = {str(t.element_type) for t in types}
        if len(dtypes) != 1:
            raise ValueError("model2MLIR matmul has mixed input or output element types")
        (m, k), (weight_k, n), output_shape, result_shape = [
            tuple(t.get_shape()) for t in types]
        if (not all(type(dim) is int and dim > 0 for dim in (m, k, weight_k, n)) or
                k != weight_k or output_shape != (m, n) or result_shape != (m, n)):
            raise ValueError("model2MLIR matmul has inconsistent static shapes")
        matmuls.append({
            "fqn": _text(operation, "prov.fqn"),
            "region_id": _text(operation, "prov.region_id"),
            "source_op": _text(operation, "prov.op"),
            "shape_mnk": [m, n, k],
            "element_type": next(iter(dtypes)),
            "k_requires_32_element_scale_padding": k % 32 != 0,
            "status": "requires_explicit_mx_quantization_and_program_lowering",
        })

    if any(len(group["fqns"]) > 1 for group in groups.values()):
        raise ValueError("model2MLIR contraction has conflicting module provenance")
    count_by_kind = Counter(kind for _, kind in groups)
    rank2_groups = {(site["region_id"], site["source_op"]) for site in matmuls}
    if len(rank2_groups) != len(matmuls):
        raise ValueError("model2MLIR contraction has ambiguous rank-two site identity")
    return {
        "schema": "mx_gemmini.model2mlir_contraction_worklist.v1",
        "status": "frontend_import_only_no_executable_placement",
        "source_mlir_sha256": hashlib.sha256(mlir_text.encode()).hexdigest(),
        "profile_sha256": profile_sha256(profile),
        "function": str(functions[0].sym_name.data),
        "legal_mx_compute_modes": profile["legal_compute"],
        "contraction_region_count": len(groups),
        "contraction_region_kinds": dict(sorted(count_by_kind.items())),
        "rank2_matmuls": matmuls,
        "other_contraction_regions": [
            {"fqn": next(iter(group["fqns"]), None), "region_id": region,
             "source_op": kind,
             "operations": dict(sorted(Counter(group["operations"]).items()))}
            for (region, kind), group in groups.items()
            if (region, kind) not in rank2_groups
        ],
    }
