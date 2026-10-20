"""Fuse closed immutable projection/sigmoid/multiply/packing source sequences."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .gated_projection_packing import emit_gated_projection_packing


def register_gated_projections(
    groups,
    kernels,
    profile,
    workspace_bytes,
    compile_source,
    abi_entries,
    *,
    activation_arithmetic="reciprocal",
):
    from merlin.llvmlower.device_shim import LogicalBuffer

    by_symbol = {k["symbol"]: k for k in kernels}
    records = []
    compiled = {}
    for group in groups:
        if len(group) != 2:
            continue
        gate, up = [by_symbol[b["symbol"]] for b in group]
        dims = gate["dimensions"]
        shape = tuple(dims[x] for x in ("m", "n", "k"))
        if any(
            k["batched"]
            or not k["prepacked_checkpoint_weight"]
            or k["precision"] != "mxfp8"
            or k["padded_shape_mnk"] != list(shape)
            or k["dimensions"] != dims
            or k["tile_mnk"] != gate["tile_mnk"]
            or k.get("activation_preparation", {}).get("consumer_arguments") != [0, 2]
            for k in (gate, up)
        ):
            continue
        m, n, _k = shape
        preparations = {
            row["activation_preparation"]["prepare_symbol"]
            for row in kernels
            if not row["batched"]
            and row["precision"] == "mxfp8"
            and row["prepacked_checkpoint_weight"]
            and row.get("activation_preparation")
            and (row["dimensions"]["m"], row["dimensions"]["k"]) == (m, n)
            and (row["padded_shape_mnk"][0], row["padded_shape_mnk"][2]) == (m, n)
        }
        if not preparations:
            continue
        issuer = Path(gate["schedule_file"]).stem
        key = (shape, tuple(gate["tile_mnk"]), issuer)
        generic = compiled.get(key)
        if generic is None:
            name = f"mx_gated_projection_shape_{len(compiled)}"
            try:
                generated = emit_gated_projection_packing(
                    profile,
                    shape,
                    symbol=name,
                    issuer_symbol=issuer,
                    tile_mnk=tuple(gate["tile_mnk"]),
                    activation_arithmetic=activation_arithmetic,
                )
            except ValueError:
                continue
            if generated.workspace_bytes > workspace_bytes:
                continue
            compile_source(name, generated.source)
            generic = (name, generated)
            compiled[key] = generic
        name, generated = generic
        entry = f"mx_gated_projection_packed_{len(records)}"
        weights = [
            (part["symbol"] + "_fixed_codes", part["symbol"] + "_fixed_scales")
            for part in (gate, up)
        ]
        declarations = "\n".join(
            f"extern const uint8_t {symbol}[];" for pair in weights for symbol in pair
        )
        source = f"""#include <stdint.h>
{declarations}
void {name}(const uint8_t *,const uint8_t *,const uint8_t *,const uint8_t *,const uint8_t *,uint8_t *);
void {entry}_kernel(const uint8_t *input,uint8_t *output) {{
  {name}(input,{weights[0][0]},{weights[0][1]},{weights[1][0]},{weights[1][1]},output);
}}
"""
        compile_source(entry + "_kernel", source)
        abi_entries[entry] = (
            entry + "_kernel",
            (
                LogicalBuffer(
                    "input", "input", "input_0", (generated.input_packed_bytes,), "i8"
                ),
                LogicalBuffer(
                    "output", "output", "out_0", (generated.output_packed_bytes,), "i8"
                ),
            ),
        )
        records.append(
            {
                "symbol": entry,
                "kernel_symbol": entry + "_kernel",
                "generic_symbol": name,
                "shape_mnk": list(shape),
                "gate_consumer": gate["activation_preparation"]["consumer_symbol"],
                "up_consumer": up["activation_preparation"]["consumer_symbol"],
                "source_kernels": [gate["symbol"], up["symbol"]],
                "source_regions": [b["region"] for b in group],
                "prepare_symbols": sorted(preparations),
                "input_packed_bytes": generated.input_packed_bytes,
                "output_packed_bytes": generated.output_packed_bytes,
                "workspace_bytes": generated.workspace_bytes,
                "schedule_digest": generated.schedule_digest,
                "activation_arithmetic": activation_arithmetic,
                "numeric_policy": (
                    "unchanged_fp8_matrices_bf16_outputs_f32_reciprocal_sigmoid_two_f32_multiplies_fp8"
                    if activation_arithmetic == "reciprocal"
                    else "unchanged_fp8_matrices_bf16_outputs_f32_gate_divide_denominator_multiply_up_fp8"
                ),
                "completion": "original_matrix_issuers_and_each_vector_tile_synchronous",
            }
        )
    return records


def source_matrix_callees(catalog, routing):
    """Resolve immutable source bindings without relying on generated names."""
    bindings = {row["region"]: row for row in catalog["bindings"]}
    resolved = {}
    for row in routing["routed"]:
        binding = bindings.get(row["source_region"])
        if binding is None:
            continue
        if any(
            row[key] != binding[key]
            for key in ("source_operation_ordinal", "tensor_types")
        ):
            raise ValueError("gated projection source binding differs from routing")
        previous = resolved.setdefault(row["symbol"], binding["symbol"])
        if previous != binding["symbol"]:
            raise ValueError("gated projection source callee has ambiguous bindings")
    return resolved


def prepared_matrix_consumers(catalog, endpoint, source_callees):
    m, width, _ = endpoint["shape_mnk"]
    return prepared_consumers_for_packed_input(
        catalog,
        rows=m,
        width=width,
        packed_bytes=endpoint["output_packed_bytes"],
        prepare_symbols=endpoint["prepare_symbols"],
        source_callees=source_callees,
    )


def prepared_consumers_for_packed_input(
    catalog, *, rows, width, packed_bytes, prepare_symbols, source_callees
):
    """Resolve existing immutable prepared ABIs for one canonical FP8 input."""
    kernels = {row["symbol"]: row for row in catalog["kernels"]}
    consumers = {}
    for callee, name in source_callees.items():
        row = kernels[name]
        dims = row["dimensions"]
        preparation = row.get("activation_preparation", {})
        if (
            not row["batched"]
            and row["precision"] == "mxfp8"
            and row["prepacked_checkpoint_weight"]
            and preparation.get("consumer_arguments") == [0, 2]
            and (dims["m"], dims["k"]) == (rows, width)
            and (row["padded_shape_mnk"][0], row["padded_shape_mnk"][2])
            == (rows, width)
            and preparation["packed_bytes"] == packed_bytes
            and preparation["prepare_symbol"] in prepare_symbols
        ):
            consumers[callee] = row
    return consumers


def rewrite_gated_projections(module, catalog, *, source_callees=None):
    from merlin.xdsl_dialects.lowering.shared_preparation import contiguous_tensor_root
    from merlin.xdsl_dialects.lowering.tensor_chains import (
        destination_only_tensor_call,
        pure_tensor_compute,
    )
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import (
        ArrayAttr,
        DictionaryAttr,
        StringAttr,
        TensorType,
        i8,
    )
    from xdsl.ir import Region
    from xdsl.ir.affine import AffineMap

    result = module.clone()
    result.verify()
    declarations = {
        o.sym_name.data: o for o in result.body.block.ops if isinstance(o, func.FuncOp)
    }
    sigmoid = {
        r["symbol"]: r
        for r in catalog["vpu_unary"]["kernels"]
        if r["operation"] == "sigmoid"
        and r["dtype"] == "f32"
        and r.get("implementation") == "f32_reciprocal"
    }
    records = []

    def declare(call):
        symbol = call.callee.root_reference.data
        signature = tuple(v.type for v in call.arguments)
        outputs = tuple(v.type for v in call.results)
        attrs = ArrayAttr(
            [
                DictionaryAttr({"bufferization.access": StringAttr(mode)})
                for mode in ["read"] * (len(signature) - 1) + ["write"]
            ]
        )
        previous = declarations.get(symbol)
        if previous is not None and (
            previous.body.blocks
            or tuple(previous.function_type.inputs) != signature
            or tuple(previous.function_type.outputs) != outputs
            or previous.arg_attrs != attrs
        ):
            raise ValueError(
                "gated projection endpoint collides with an incompatible declaration"
            )
        if previous is None:
            declaration = func.FuncOp(
                symbol,
                (signature, outputs),
                Region(),
                visibility="private",
                arg_attrs=attrs,
            )
            result.body.block.add_op(declaration)
            declarations[symbol] = declaration

    for endpoint in catalog.get("gated_projection_packing", []):
        m, n, _ = endpoint["shape_mnk"]
        consumers = prepared_matrix_consumers(catalog, endpoint, source_callees or {})
        for prepare in tuple(result.walk()):
            if not isinstance(prepare, func.CallOp) or not destination_only_tensor_call(
                prepare
            ):
                continue
            original = prepare.callee.root_reference.data
            matrix = consumers.get(original)
            if matrix is None:
                if (
                    original not in endpoint["prepare_symbols"]
                    or len(prepare.arguments) != 2
                    or str(prepare.arguments[0].type) != f"tensor<{m}x{n}xf32>"
                    or str(prepare.results[0].type)
                    != f"tensor<{endpoint['output_packed_bytes']}xi8>"
                ):
                    continue
            else:
                out_n = matrix["dimensions"]["n"]
                if len(prepare.arguments) != 3 or [
                    str(v.type) for v in prepare.arguments
                ] != [
                    f"tensor<{m}x{n}xf32>",
                    f"tensor<{n}x{out_n}xf32>",
                    f"tensor<{m}x{out_n}xf32>",
                ]:
                    continue
            operations = {prepare}

            def root(value, operations=operations):
                base = contiguous_tensor_root(value)
                while value is not base:
                    operations.add(value.owner)
                    value = value.owner.operands[0]
                return base

            def multiply(value, operations=operations, root=root):
                op = root(value).owner
                if (
                    getattr(op, "name", None) != "linalg.generic"
                    or not pure_tensor_compute(op)
                    or len(op.inputs) != 2
                    or len(op.outputs) != 1
                    or len(op.results) != 1
                    or not op.inputs[0].type == op.inputs[1].type == op.results[0].type
                    or op.outputs[0].type != op.results[0].type
                ):
                    raise ValueError("not an exact full-shape FP32 multiply")
                rank = len(op.results[0].type.get_shape())
                steps = tuple(op.body.block.ops)
                if (
                    tuple(a.data for a in op.indexing_maps)
                    != (AffineMap.identity(rank),) * 3
                    or len(op.iterator_types) != rank
                    or any(
                        str(it) != "#linalg.iterator_type<parallel>"
                        for it in op.iterator_types
                    )
                    or len(steps) != 2
                    or steps[0].name != "arith.mulf"
                    or steps[1].name != "linalg.yield"
                    or tuple(steps[0].operands) != tuple(op.body.block.args[:2])
                    or tuple(steps[1].operands) != (steps[0].results[0],)
                    or any(
                        a.data
                        for a in (steps[0].properties.get("fastmath"),)
                        if a is not None
                    )
                ):
                    raise ValueError("multiply arithmetic or indexing differs")
                operations.add(op)
                return op

            def call(value, symbols, operations=operations, root=root):
                op = root(value).owner
                if (
                    not isinstance(op, func.CallOp)
                    or op.callee.root_reference.data not in symbols
                    or len(op.arguments) != 2
                    or not destination_only_tensor_call(op)
                ):
                    raise ValueError("not a registered private projection or sigmoid")
                operations.add(op)
                return op

            try:
                second = multiply(prepare.arguments[0])
                first = multiply(second.inputs[0])
                up = call(second.inputs[1], {endpoint["up_consumer"]})
                gate = call(first.inputs[0], {endpoint["gate_consumer"]})
                sig = call(first.inputs[1], sigmoid)
                if (
                    root(sig.arguments[0]) is not gate.results[0]
                    or sig.arguments[0].type != first.results[0].type
                    or first.results[0].type != second.results[0].type
                    or str(gate.results[0].type) != f"tensor<{m}x{n}xf32>"
                    or up.results[0].type != gate.results[0].type
                    or gate.arguments[0] is not up.arguments[0]
                    or str(gate.arguments[0].type)
                    != f"tensor<{endpoint['input_packed_bytes']}xi8>"
                ):
                    continue
                if any(
                    use.operation not in operations
                    for op in operations
                    if op is not prepare
                    for value in op.results
                    for use in value.uses
                ):
                    continue
                block = prepare.parent
                if block is None or any(op.parent is not block for op in operations):
                    continue
                ordered = [op for op in block.ops if op in operations]
                if ordered[0] is not gate or ordered[-1] is not prepare:
                    continue
                cursor = gate
                while cursor is not prepare:
                    if cursor not in operations and not pure_tensor_compute(cursor):
                        break
                    cursor = cursor.next_op
                if cursor is not prepare:
                    continue
            except (ValueError, TypeError, AttributeError, KeyError):
                continue
            packed_type = TensorType(i8, [endpoint["output_packed_bytes"]])
            destination = tensor.EmptyOp((), packed_type)
            fused = func.CallOp(
                endpoint["symbol"],
                [gate.arguments[0], destination.results[0]],
                [packed_type],
            )
            declare(fused)
            replacements = [destination, fused]
            replacement = fused
            if matrix is not None:
                replacement = func.CallOp(
                    matrix["activation_preparation"]["consumer_symbol"],
                    [fused.results[0], prepare.arguments[-1]],
                    [prepare.results[0].type],
                )
                declare(replacement)
                replacements.append(replacement)
            block.insert_ops_before(replacements, prepare)
            prepare.results[0].replace_all_uses_with(replacement.results[0])
            records.append(
                {
                    "symbol": endpoint["symbol"],
                    "source_kernels": endpoint["source_kernels"],
                    "source_regions": endpoint["source_regions"],
                    "original_sigmoid": sig.callee.root_reference.data,
                    "original_prepare": original if matrix is None else None,
                    "original_matrix_consumer": original
                    if matrix is not None
                    else None,
                    "matrix_consumer_kernel": matrix["symbol"]
                    if matrix is not None
                    else None,
                    "shape_mnk": endpoint["shape_mnk"],
                }
            )
            for op in reversed(ordered):
                op.detach()
                op.erase()
    result.verify()
    return result, records


class MXGatedProjectionPostOffload:
    def __init__(self, catalog_path):
        self.catalog_path = Path(catalog_path)

    def __call__(self, prepared, sidecar, workdir):
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text

        catalog = json.loads(self.catalog_path.read_text())
        routing = json.loads(Path(sidecar).read_text())
        if (
            Path(routing["catalog_manifest"]).resolve() != self.catalog_path.resolve()
            or routing["model_sha256"] != catalog["source_sha256"]
            or not catalog.get("fuse_gated_projections")
        ):
            raise ValueError("gated projection catalog differs from source routing")
        result, records = rewrite_gated_projections(
            parse_mlir_file(prepared),
            catalog,
            source_callees=source_matrix_callees(catalog, routing),
        )
        output = Path(workdir) / "mx_gated_projections.mlir"
        output.write_text(text(result))
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        output.with_name("mx_gated_projections_rewrite.json").write_text(
            json.dumps(
                {
                    "schema": "mx_gemmini.gated_projection_packing.v1",
                    "source_sha256": digest(prepared),
                    "output_sha256": digest(output),
                    "catalog_sha256": digest(self.catalog_path),
                    "sidecar_sha256": digest(sidecar),
                    "fusions": records,
                },
                indent=2,
            )
            + "\n"
        )
        return output
