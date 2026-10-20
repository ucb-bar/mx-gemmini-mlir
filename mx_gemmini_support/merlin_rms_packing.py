"""Bind private catalog RMS sequences to a streamed VPU/FP8 endpoint."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .vpu_rms_packing import emit_vpu_rms_packing


def register_rms_packing(pairs, kernels, profile, compile_source, abi_entries):
    from merlin.llvmlower.device_shim import LogicalBuffer

    records = {}
    for pair in pairs:
        shape = tuple(pair["shape"])
        if len(shape) < 2 or "private_bf16_output" not in pair:
            continue
        width, rows = shape[-1], math.prod(shape[:-1])
        if (
            pair["input_shapes"] != [list(shape), [*shape[:-1], 1], [width]]
            or pair["input_axes"]
            != [list(range(len(shape))), list(range(len(shape))), [len(shape) - 1]]
            or rows % 16
            or not 32 <= width <= 4096
            or width & (width - 1)
        ):
            continue
        preparations = set()
        for kernel in kernels:
            d = kernel["dimensions"]
            packing = kernel.get("activation_preparation", {})
            endpoint = packing.get("private_bf16_input")
            if (
                endpoint
                and not kernel["batched"]
                and kernel["precision"] == "mxfp8"
                and (d["m"], d["k"]) == (rows, width)
                and kernel["padded_shape_mnk"][0] == rows
                and kernel["padded_shape_mnk"][2] == width
            ):
                preparations.add(endpoint["symbol"])
        if not preparations:
            continue
        name = "mx_vpu_rms_packed_" + "_".join(map(str, shape))
        if name not in records:
            generated = emit_vpu_rms_packing(
                profile, rows=rows, width=width, symbol=name + "_kernel"
            )
            compile_source(name + "_kernel", generated.source)
            buffers = (
                LogicalBuffer("input", "input", "input_0", shape, "f32"),
                LogicalBuffer("gamma", "input", "input_1", (width,), "f32"),
                LogicalBuffer("divisor", "input", "input_2", (), "f32"),
                LogicalBuffer("epsilon", "input", "input_3", (), "f32"),
                LogicalBuffer(
                    "packed", "output", "out_0", (generated.packed_bytes,), "i8"
                ),
            )
            abi_entries[name] = (name + "_kernel", buffers)
            records[name] = {
                "symbol": name,
                "kernel_symbol": name + "_kernel",
                "shape": list(shape),
                "packed_bytes": generated.packed_bytes,
                "rows_per_chunk": generated.rows_per_chunk,
                "producer_symbols": [],
                "prepare_symbols": sorted(preparations),
                "numeric_policy": "unchanged_bf16_vpu_rms_then_fp8_block32_packing",
            }
        producer = pair["private_bf16_output"]["symbol"]
        if producer not in records[name]["producer_symbols"]:
            records[name]["producer_symbols"].append(producer)
    return list(records.values())


def rewrite_rms_packing(module, catalog):
    """Replace only closed, synchronous destination-only catalog call sequences.

    All crossed operations must be pure tensor computation. Registered endpoint
    metadata establishes the target arithmetic; complete SSA use closure keeps
    every intermediate private. Unknown calls or live escapes keep the original.
    """
    from merlin.xdsl_dialects.lowering.shared_preparation import contiguous_tensor_root
    from merlin.xdsl_dialects.lowering.tensor_chains import (
        destination_only_tensor_call,
        pure_tensor_compute,
    )
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
    from xdsl.ir import Region

    result = module.clone()
    result.verify()
    unary = {x["symbol"]: x for x in catalog["vpu_unary"]["kernels"]}
    binary = {x["symbol"]: x for x in catalog["vpu_elementwise"]["kernels"]}
    reduce = {x["symbol"]: x for x in catalog["vpu_reduce"]["kernels"]}
    declarations = {
        o.sym_name.data: o for o in result.body.block.ops if isinstance(o, func.FuncOp)
    }
    records = []
    for endpoint in catalog.get("rms_packing", []):
        shape = tuple(endpoint["shape"])
        width = shape[-1]
        for prepare in tuple(result.walk()):
            if (
                not isinstance(prepare, func.CallOp)
                or prepare.callee.root_reference.data not in endpoint["prepare_symbols"]
            ):
                continue
            operations = set()

            def root(value):
                base = contiguous_tensor_root(value)
                while value is not base:
                    operations.add(value.owner)
                    value = value.owner.operands[0]
                return base

            def call(value, collection, operation):
                owner = root(value).owner
                if not isinstance(
                    owner, func.CallOp
                ) or not destination_only_tensor_call(owner):
                    raise ValueError("not a private destination-only endpoint")
                spec = collection.get(owner.callee.root_reference.data)
                if (
                    spec is None
                    or spec.get("operation") != operation
                    or spec.get("dtype", "f32") != "f32"
                ):
                    raise ValueError("different catalog arithmetic")
                operations.add(owner)
                return owner, spec

            def constant(value):
                owner = value.owner
                if owner.name != "tensor.splat":
                    raise ValueError("RMS constants must be explicit scalar splats")
                scalar = owner.operands[0].owner
                if scalar.name != "arith.constant":
                    raise ValueError("RMS scalar is not a constant")
                attr = scalar.properties.get("value", scalar.attributes.get("value"))
                return float(attr.value.data)

            try:
                multiply = root(prepare.arguments[0]).owner
                if (
                    not isinstance(multiply, func.CallOp)
                    or multiply.callee.root_reference.data
                    not in endpoint["producer_symbols"]
                    or not destination_only_tensor_call(multiply)
                    or len(multiply.arguments) != 4
                    or len(prepare.arguments) != 2
                ):
                    continue
                x, inverse, gamma = multiply.arguments[:3]
                if (
                    str(x.type) != "tensor<" + "x".join(map(str, shape)) + "xf32>"
                    or str(gamma.type) != f"tensor<{width}xf32>"
                    or str(prepare.results[0].type)
                    != f"tensor<{endpoint['packed_bytes']}xi8>"
                ):
                    continue
                operations.update((multiply, prepare))
                rsqrt, rsqrt_spec = call(inverse, unary, "rsqrt")
                add, add_spec = call(rsqrt.arguments[0], binary, "add")
                divide, divide_spec = call(add.arguments[0], binary, "div")
                summed, sum_spec = call(divide.arguments[0], reduce, "rsum")
                square, square_spec = call(summed.arguments[0], unary, "square")
                divisor, epsilon = divide.arguments[1], add.arguments[1]
                if (
                    add_spec["rhs_shape"]
                    or divide_spec["rhs_shape"]
                    or tuple(square_spec["shape"]) != shape
                    or tuple(sum_spec["shape"]) != shape
                    or tuple(rsqrt_spec["shape"]) != (*shape[:-1], 1)
                    or tuple(add_spec["shape"]) != (*shape[:-1], 1)
                    or tuple(divide_spec["shape"]) != shape[:-1]
                    or tuple(sum_spec["output_shape"]) != shape[:-1]
                    or str(divisor.type) != "tensor<f32>"
                    or str(epsilon.type) != "tensor<f32>"
                    or root(square.arguments[0]) is not root(x)
                    or constant(divisor) != width
                    or not math.isfinite(constant(epsilon))
                    or constant(epsilon) <= 0
                ):
                    continue
                # No removed result may escape to another arithmetic or call.
                if any(
                    use.operation not in operations
                    for op in operations
                    if op is not prepare
                    for value in op.results
                    for use in value.uses
                ):
                    continue
                block = prepare.parent
                if any(op.parent is not block for op in operations):
                    continue
                ordered = [op for op in block.ops if op in operations]
                first, last = ordered[0], ordered[-1]
                if first is not square or last is not prepare:
                    continue
                cursor = first
                while cursor is not last:
                    if cursor not in operations and not pure_tensor_compute(cursor):
                        break
                    cursor = cursor.next_op
                if cursor is not last:
                    continue
                # Each removed device call keeps its explicit readonly/writeonly contract.
                if any(
                    isinstance(op, func.CallOp) and not destination_only_tensor_call(op)
                    for op in operations
                ):
                    continue
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            arguments = [x, gamma, divisor, epsilon]
            destination = tensor.EmptyOp((), prepare.results[0].type)
            fused = func.CallOp(
                endpoint["symbol"],
                [*arguments, destination.results[0]],
                [prepare.results[0].type],
            )
            signature = tuple(v.type for v in fused.arguments)
            attrs = ArrayAttr(
                [DictionaryAttr({"bufferization.access": StringAttr("read")})] * 4
                + [DictionaryAttr({"bufferization.access": StringAttr("write")})]
            )
            previous = declarations.get(endpoint["symbol"])
            if previous is not None and (
                tuple(previous.function_type.inputs) != signature
                or tuple(previous.function_type.outputs) != (fused.results[0].type,)
                or previous.body.blocks
                or previous.arg_attrs != attrs
            ):
                raise ValueError("RMS endpoint collides with an existing declaration")
            block.insert_ops_before([destination, fused], prepare)
            prepare.results[0].replace_all_uses_with(fused.results[0])
            records.append(
                {
                    "symbol": endpoint["symbol"],
                    "shape": list(shape),
                    "epsilon": constant(epsilon),
                    "removed_calls": [
                        op.callee.root_reference.data
                        for op in ordered
                        if isinstance(op, func.CallOp)
                    ],
                }
            )
            for op in reversed(ordered):
                op.detach()
                op.erase()
            if previous is None:
                declaration = func.FuncOp(
                    endpoint["symbol"],
                    (signature, [fused.results[0].type]),
                    Region(),
                    visibility="private",
                    arg_attrs=attrs,
                )
                result.body.block.add_op(declaration)
                declarations[endpoint["symbol"]] = declaration
    result.verify()
    return result, records


class MXRmsPackingPostOffload:
    def __init__(self, catalog_path):
        self.catalog_path = Path(catalog_path)

    def __call__(self, prepared, sidecar, workdir):
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text

        catalog = json.loads(self.catalog_path.read_text())
        route = json.loads(Path(sidecar).read_text())
        if (
            Path(route["catalog_manifest"]).resolve() != self.catalog_path.resolve()
            or route["model_sha256"] != catalog["source_sha256"]
            or not catalog.get("fuse_rms_packing")
        ):
            raise ValueError("RMS packing catalog differs from source routing")
        result, records = rewrite_rms_packing(parse_mlir_file(prepared), catalog)
        output = Path(workdir) / "mx_rms_packing.mlir"
        output.write_text(text(result))
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        output.with_name("mx_rms_packing_rewrite.json").write_text(
            json.dumps(
                {
                    "schema": "mx_gemmini.rms_packing.v1",
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
