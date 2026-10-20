"""Fuse closed FP32 RMS graphs into their canonical FP8 preparation boundary."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .f32_rms_packing import emit_f32_rms_packing
from .merlin_gated_projection import (
    prepared_consumers_for_packed_input,
    source_matrix_callees,
)


def register_f32_rms_packing(
    prepared,
    kernels,
    profile,
    workspace_bytes,
    compile_source,
    abi_entries,
    *,
    reduction="ordered",
):
    from merlin.llvmlower.device_shim import LogicalBuffer

    from .merlin_vpu_reduce import selected_row_reduce_sites

    shapes = sorted(
        {
            tuple(row["shape"])
            for row in selected_row_reduce_sites(Path(prepared).read_text(), profile)
            if row["operation"] == "rsum"
        }
    )
    records = []
    for shape in shapes:
        rows, width = math.prod(shape[:-1]), shape[-1]
        preparations = sorted(
            {
                k["activation_preparation"]["prepare_symbol"]
                for k in kernels
                if not k["batched"]
                and k["prepacked_checkpoint_weight"]
                and k["precision"] == "mxfp8"
                and k.get("activation_preparation", {}).get("consumer_arguments")
                == [0, 2]
                and (k["dimensions"]["m"], k["dimensions"]["k"]) == (rows, width)
                and (k["padded_shape_mnk"][0], k["padded_shape_mnk"][2])
                == (rows, width)
            }
        )
        if not preparations:
            continue
        symbol = "mx_f32_rms_packed_" + "_".join(map(str, shape)) + "_" + reduction
        try:
            generated = emit_f32_rms_packing(
                profile,
                rows=rows,
                width=width,
                symbol=symbol + "_kernel",
                reduction=reduction,
            )
        except ValueError:
            continue
        if generated.workspace_bytes > workspace_bytes:
            continue
        compile_source(
            symbol + "_kernel", generated.source, extra_flags=("-ffp-contract=off", "-fno-math-errno")
        )
        buffers = (
            LogicalBuffer("input", "input", "input_0", shape, "f32"),
            LogicalBuffer("gamma", "input", "input_1", (width,), "f32"),
            LogicalBuffer("divisor", "input", "input_2", (), "f32"),
            LogicalBuffer("epsilon", "input", "input_3", (), "f32"),
            LogicalBuffer("packed", "output", "out_0", (generated.packed_bytes,), "i8"),
        )
        abi_entries[symbol] = (symbol + "_kernel", buffers)
        records.append(
            {
                "symbol": symbol,
                "kernel_symbol": symbol + "_kernel",
                "shape": list(shape),
                "rows": rows,
                "width": width,
                "packed_bytes": generated.packed_bytes,
                "workspace_bytes": generated.workspace_bytes,
                "prepare_symbols": preparations,
                "reduction": reduction,
                "numeric_policy": "fp32_square_sum_divide_add_rsqrt_two_multiplies_then_canonical_fp8",
            }
        )
    return records


def rewrite_f32_rms_packing(module, catalog, *, source_callees=None):
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
    from xdsl.ir.affine import AffineExpr, AffineMap

    result = module.clone()
    result.verify()
    # Shared preparation leaves unused views from the former matrix operands.
    # Erase only dead, pure shape views before checking intermediate privacy.
    for operation in reversed(tuple(result.walk())):
        if (
            operation.name in {"tensor.collapse_shape", "tensor.expand_shape"}
            and pure_tensor_compute(operation)
            and all(not value.uses for value in operation.results)
        ):
            operation.detach()
            operation.erase()
    declarations = {
        o.sym_name.data: o for o in result.body.block.ops if isinstance(o, func.FuncOp)
    }
    records = []

    def declare(call):
        name = call.callee.root_reference.data
        types = tuple(v.type for v in call.arguments)
        outputs = tuple(v.type for v in call.results)
        attrs = ArrayAttr(
            [
                DictionaryAttr({"bufferization.access": StringAttr(mode)})
                for mode in ["read"] * (len(types) - 1) + ["write"]
            ]
        )
        previous = declarations.get(name)
        if previous is not None:
            if (
                previous.body.blocks
                or tuple(previous.function_type.inputs) != types
                or tuple(previous.function_type.outputs) != outputs
                or previous.arg_attrs != attrs
            ):
                raise ValueError(
                    "FP32 RMS endpoint collides with an incompatible declaration"
                )
        else:
            previous = func.FuncOp(
                name, (types, outputs), Region(), visibility="private", arg_attrs=attrs
            )
            result.body.block.add_op(previous)
            declarations[name] = previous

    for endpoint in catalog.get("f32_rms_packing", []):
        shape = tuple(endpoint["shape"])
        rows, width = endpoint["rows"], endpoint["width"]
        consumers = prepared_consumers_for_packed_input(
            catalog,
            rows=rows,
            width=width,
            packed_bytes=endpoint["packed_bytes"],
            prepare_symbols=endpoint["prepare_symbols"],
            source_callees=source_callees or {},
        )
        for terminal in tuple(result.walk()):
            if not isinstance(
                terminal, func.CallOp
            ) or not destination_only_tensor_call(terminal):
                continue
            name = terminal.callee.root_reference.data
            matrix = consumers.get(name)
            if matrix is None:
                if (
                    name not in endpoint["prepare_symbols"]
                    or len(terminal.arguments) != 2
                    or str(terminal.results[0].type)
                    != f"tensor<{endpoint['packed_bytes']}xi8>"
                ):
                    continue
            else:
                out_n = matrix["dimensions"]["n"]
                if len(terminal.arguments) != 3 or [
                    str(v.type) for v in terminal.arguments
                ] != [
                    f"tensor<{rows}x{width}xf32>",
                    f"tensor<{width}x{out_n}xf32>",
                    f"tensor<{rows}x{out_n}xf32>",
                ]:
                    continue
            if str(terminal.arguments[0].type) != f"tensor<{rows}x{width}xf32>":
                continue
            operations = {terminal}

            def root(value, operations=operations):
                base = contiguous_tensor_root(value)
                while value is not base:
                    operations.add(value.owner)
                    value = value.owner.operands[0]
                return base

            def generic(value, operation, arity, operations=operations, root=root):
                op = root(value).owner
                if (
                    op.name != "linalg.generic"
                    or not pure_tensor_compute(op)
                    or len(op.inputs) != arity
                    or len(op.outputs) != 1
                    or len(op.results) != 1
                    or op.outputs[0].type != op.results[0].type
                    or any(
                        str(it) != "#linalg.iterator_type<parallel>"
                        for it in op.iterator_types
                    )
                    or any(
                        str(v.type.element_type) != "f32"
                        for v in (*op.inputs, *op.results)
                    )
                ):
                    raise ValueError("not an exact FP32 pointwise tensor operation")
                steps = tuple(op.body.block.ops)
                if (
                    len(steps) != 2
                    or steps[0].name != operation
                    or steps[1].name != "linalg.yield"
                    or tuple(steps[1].operands) != (steps[0].results[0],)
                ):
                    raise ValueError("different FP32 scalar arithmetic")
                if any(
                    a.data
                    for a in (steps[0].properties.get("fastmath"),)
                    if a is not None
                ):
                    raise ValueError("fast-math operation")
                operations.add(op)
                return op, steps[0]

            def maps(op, expected):
                if tuple(a.data for a in op.indexing_maps) != tuple(expected):
                    raise ValueError("different FP32 broadcast indexing")

            def constant(value):
                if (
                    str(value.type) != "tensor<f32>"
                    or value.owner.name != "tensor.splat"
                ):
                    raise ValueError("not a scalar constant tensor")
                op = value.owner.operands[0].owner
                if op.name != "arith.constant":
                    raise ValueError("not a scalar constant")
                attr = op.properties.get("value", op.attributes.get("value"))
                return float(attr.value.data)

            try:
                final, body = generic(terminal.arguments[0], "arith.mulf", 2)
                rank = len(shape)
                identity = AffineMap.identity(rank)
                gamma_map = AffineMap(rank, 0, (AffineExpr.dimension(rank - 1),))
                if tuple(final.results[0].type.get_shape()) != shape or tuple(
                    body.operands
                ) != tuple(final.body.block.args[:2]):
                    continue
                mapping = tuple(a.data for a in final.indexing_maps)
                if mapping == (gamma_map, identity, identity):
                    gamma, scaled = final.inputs
                elif mapping == (identity, gamma_map, identity):
                    scaled, gamma = final.inputs
                else:
                    continue
                if str(gamma.type) != f"tensor<{width}xf32>":
                    continue
                first, body = generic(scaled, "arith.mulf", 2)
                if first.results[0].type != final.results[0].type or tuple(
                    body.operands
                ) != tuple(first.body.block.args[:2]):
                    continue
                inv_map = AffineMap(
                    rank,
                    0,
                    tuple(AffineExpr.dimension(i) for i in range(rank - 1))
                    + (AffineExpr.constant(0),),
                )
                maps(first, (identity, inv_map, identity))
                x = first.inputs[0]
                inverse, body = generic(first.inputs[1], "math.rsqrt", 1)
                inv_shape = (*shape[:-1], 1)
                if tuple(inverse.results[0].type.get_shape()) != inv_shape or tuple(
                    body.operands
                ) != (inverse.body.block.args[0],):
                    continue
                maps(inverse, (identity, identity))
                add, body = generic(inverse.inputs[0], "arith.addf", 2)
                maps(add, (identity, AffineMap(rank, 0, ()), identity))
                if tuple(body.operands) != tuple(add.body.block.args[:2]):
                    continue
                epsilon = add.inputs[1]
                if not math.isfinite(constant(epsilon)) or constant(epsilon) <= 0:
                    continue
                division, body = generic(add.inputs[0], "arith.divf", 2)
                if (
                    tuple(body.operands) != tuple(division.body.block.args[:2])
                    or tuple(division.results[0].type.get_shape()) != shape[:-1]
                ):
                    continue
                maps(
                    division,
                    (
                        AffineMap.identity(rank - 1),
                        AffineMap(rank - 1, 0, ()),
                        AffineMap.identity(rank - 1),
                    ),
                )
                divisor = division.inputs[1]
                if constant(divisor) != width:
                    continue
                reduce = root(division.inputs[0]).owner
                if (
                    reduce.name != "linalg.reduce"
                    or len(reduce.results) != 1
                    or tuple(reduce.dimensions.get_values()) != (rank - 1,)
                    or reduce.results[0].type != division.inputs[0].type
                    or reduce.init.type != reduce.results[0].type
                ):
                    continue
                init = reduce.init.owner
                if (
                    init.name != "tensor.splat"
                    or init.operands[0].owner.name != "arith.constant"
                ):
                    continue
                attr = init.operands[0].owner.properties["value"]
                if float(attr.value.data) != 0:
                    continue
                steps = tuple(reduce.region.block.ops)
                if (
                    len(steps) != 2
                    or steps[0].name != "arith.addf"
                    or steps[1].name != "linalg.yield"
                    or tuple(steps[0].operands) != tuple(reduce.region.block.args)
                    or tuple(steps[1].operands) != (steps[0].results[0],)
                ):
                    continue
                if any(
                    a.data
                    for a in (steps[0].properties.get("fastmath"),)
                    if a is not None
                ):
                    continue
                operations.add(reduce)
                square, body = generic(reduce.input, "arith.mulf", 1)
                if (
                    square.inputs[0] is not x
                    or square.results[0].type != x.type
                    or tuple(body.operands) != (square.body.block.args[0],) * 2
                ):
                    continue
                maps(square, (identity, identity))
                if x.type != final.results[0].type:
                    continue
                if any(
                    use.operation not in operations
                    for op in operations
                    if op is not terminal
                    for value in op.results
                    for use in value.uses
                ):
                    continue
                block = terminal.parent
                if block is None or any(op.parent is not block for op in operations):
                    continue
                ordered = [op for op in block.ops if op in operations]
                if ordered[0] is not square or ordered[-1] is not terminal:
                    continue
                cursor = square
                while cursor is not terminal:
                    if cursor not in operations and not pure_tensor_compute(cursor):
                        break
                    cursor = cursor.next_op
                if cursor is not terminal:
                    continue
            except (ValueError, TypeError, AttributeError, KeyError):
                continue
            packed_type = TensorType(i8, [endpoint["packed_bytes"]])
            destination = tensor.EmptyOp((), packed_type)
            fused = func.CallOp(
                endpoint["symbol"],
                [x, gamma, divisor, epsilon, destination.results[0]],
                [packed_type],
            )
            declare(fused)
            replacements = [destination, fused]
            replacement = fused
            if matrix is not None:
                replacement = func.CallOp(
                    matrix["activation_preparation"]["consumer_symbol"],
                    [fused.results[0], terminal.arguments[-1]],
                    [terminal.results[0].type],
                )
                declare(replacement)
                replacements.append(replacement)
            block.insert_ops_before(replacements, terminal)
            terminal.results[0].replace_all_uses_with(replacement.results[0])
            regions = sorted(
                {
                    getattr(op.attributes.get("prov.region_id"), "data", "")
                    for op in operations
                }
                - {""}
            )
            records.append(
                {
                    "symbol": endpoint["symbol"],
                    "shape": list(shape),
                    "reduction": endpoint["reduction"],
                    "source_regions": regions,
                    "original_prepare": name if matrix is None else None,
                    "original_matrix_consumer": name if matrix is not None else None,
                    "matrix_consumer_kernel": matrix["symbol"]
                    if matrix is not None
                    else None,
                }
            )
            for op in reversed(ordered):
                op.detach()
                op.erase()
    result.verify()
    return result, records


class MXF32RMSPackingPostOffload:
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
            or not catalog.get("fuse_f32_rms_packing")
        ):
            raise ValueError("FP32 RMS packing catalog differs from source routing")
        result, records = rewrite_f32_rms_packing(
            parse_mlir_file(prepared),
            catalog,
            source_callees=source_matrix_callees(catalog, routing),
        )
        output = Path(workdir) / "mx_f32_rms_packing.mlir"
        output.write_text(text(result))
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        output.with_name("mx_f32_rms_packing_rewrite.json").write_text(
            json.dumps(
                {
                    "schema": "mx_gemmini.f32_rms_packing.v1",
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
