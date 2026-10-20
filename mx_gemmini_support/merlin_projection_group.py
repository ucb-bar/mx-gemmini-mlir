"""Group independent immutable projections of one explicitly packed SSA input."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .projection_group import emit_projection_group


def projection_candidates(prepared, plan):
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.xdsl_dialects.lowering.shared_preparation import contiguous_tensor_root

    module = parse_mlir_file(prepared)
    operations = list(module.walk())
    kernels = {k["symbol"]: k for k in plan["kernels"]}
    groups = defaultdict(list)
    for binding in plan["bindings"]:
        kernel = kernels[binding["symbol"]]
        if (
            kernel["rank"] != 2
            or not kernel["prepacked_checkpoint_weight"]
            or kernel["precision"] != "mxfp8"
        ):
            continue
        op = operations[binding["source_operation_ordinal"]]
        m, _, k = kernel["shape"]
        groups[contiguous_tensor_root(op.operands[0]), m, k].append(binding)
    return [
        sorted(group, key=lambda b: b["source_operation_ordinal"])
        for group in groups.values()
        if 2 <= len(group) <= 8
    ]


def register_projection_groups(
    groups, kernels, profile, workspace_bytes, compile_source, abi_entries
):
    from merlin.llvmlower.device_shim import LogicalBuffer

    by_symbol = {k["symbol"]: k for k in kernels}
    records = []
    for group in groups:
        parts, consumers, selected = [], [], []
        for binding in group:
            kernel = by_symbol[binding["symbol"]]
            packing = kernel.get("activation_preparation")
            d = kernel["dimensions"]
            shape = (d["m"], d["n"], d["k"])
            if (
                packing is None
                or kernel["batched"]
                or not kernel["prepacked_checkpoint_weight"]
                or kernel["precision"] != "mxfp8"
                or packing["consumer_arguments"] != [0, 2]
                or list(shape) != kernel["padded_shape_mnk"]
            ):
                break
            parts.append(
                {
                    "shape": shape,
                    "tile_mnk": kernel["tile_mnk"],
                    "weight_symbols": (
                        kernel["symbol"] + "_fixed_codes",
                        kernel["symbol"] + "_fixed_scales",
                    ),
                }
            )
            consumers.append(packing["consumer_symbol"])
            selected.append(kernel)
        if len(parts) != len(group):
            continue
        name = f"mx_projection_group_{len(records)}"
        try:
            generated = emit_projection_group(
                profile,
                parts,
                symbol=name + "_kernel",
                workspace_symbol="mx_catalog_workspace",
            )
        except ValueError:
            continue
        # Do not increase the existing private serial workspace reservation.
        if generated.workspace_bytes > workspace_bytes:
            continue
        packed = selected[0]["activation_preparation"]["packed_bytes"]
        if any(k["activation_preparation"]["packed_bytes"] != packed for k in selected):
            continue
        compile_source(name + "_kernel", generated.source)
        buffers = (
            LogicalBuffer("packed", "input", "input_0", (packed,), "i8"),
        ) + tuple(
            LogicalBuffer(f"output{i}", "output", f"out_{i}", shape, "f32")
            for i, shape in enumerate(generated.output_shapes)
        )
        abi_entries[name] = (name + "_kernel", buffers)
        records.append(
            {
                "symbol": name,
                "kernel_symbol": name + "_kernel",
                "consumer_symbols": consumers,
                "kernel_symbols": [k["symbol"] for k in selected],
                "source_regions": [b["region"] for b in group],
                "packed_bytes": packed,
                "output_shapes": [list(s) for s in generated.output_shapes],
                "workspace_bytes": generated.workspace_bytes,
                "schedule_digests": list(generated.schedule_digests),
                "completion": "one_fence_before_all_f32_readouts",
                "numerics": "unchanged_original_tiles_weights_and_bf16_matrix_outputs",
            }
        )
    return records


def rewrite_projection_groups(module, catalog):
    from merlin.xdsl_dialects.lowering.tensor_chains import (
        destination_only_tensor_call,
        pure_tensor_compute,
    )
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
    from xdsl.ir import Region

    result = module.clone()
    result.verify()
    declarations = {
        op.sym_name.data: op
        for op in result.body.block.ops
        if isinstance(op, func.FuncOp)
    }
    records = []
    for group in catalog.get("projection_groups", []):
        found = defaultdict(list)
        for op in result.walk():
            if (
                isinstance(op, func.CallOp)
                and op.callee.root_reference.data in group["consumer_symbols"]
            ):
                found[op.callee.root_reference.data].append(op)
        if any(len(found[name]) != 1 for name in group["consumer_symbols"]):
            continue
        calls = [found[name][0] for name in group["consumer_symbols"]]
        block = calls[0].parent
        packed = calls[0].arguments[0]
        if (
            block is None
            or any(
                c.parent is not block
                or len(c.arguments) != 2
                or not destination_only_tensor_call(c)
                or c.arguments[0] is not packed
                for c in calls
            )
            or str(packed.type) != f"tensor<{group['packed_bytes']}xi8>"
        ):
            continue
        expected = [
            "tensor<" + "x".join(map(str, shape)) + "xf32>"
            for shape in group["output_shapes"]
        ]
        if [str(c.results[0].type) for c in calls] != expected:
            continue
        ordered = [op for op in block.ops if op in calls]
        if ordered != calls:
            continue
        cursor = calls[0]
        while cursor is not calls[-1]:
            if cursor not in calls and not pure_tensor_compute(cursor):
                break
            cursor = cursor.next_op
        if cursor is not calls[-1]:
            continue
        destinations = [tensor.EmptyOp((), c.results[0].type) for c in calls]
        fused = func.CallOp(
            group["symbol"],
            [packed, *(d.results[0] for d in destinations)],
            [c.results[0].type for c in calls],
        )
        signature = tuple(v.type for v in fused.arguments)
        attrs = ArrayAttr(
            [DictionaryAttr({"bufferization.access": StringAttr("read")})]
            + [DictionaryAttr({"bufferization.access": StringAttr("write")})]
            * len(calls)
        )
        previous = declarations.get(group["symbol"])
        if previous is not None and (
            previous.body.blocks
            or tuple(previous.function_type.inputs) != signature
            or tuple(previous.function_type.outputs)
            != tuple(v.type for v in fused.results)
            or previous.arg_attrs != attrs
        ):
            raise ValueError("grouped projection collides with an existing declaration")
        block.insert_ops_before([*destinations, fused], calls[0])
        for old, new in zip(calls, fused.results, strict=True):
            old.results[0].replace_all_uses_with(new)
            old.detach()
            old.erase()
        if previous is None:
            declaration = func.FuncOp(
                group["symbol"],
                (signature, [v.type for v in fused.results]),
                Region(),
                visibility="private",
                arg_attrs=attrs,
            )
            result.body.block.add_op(declaration)
            declarations[group["symbol"]] = declaration
        records.append(
            {
                "symbol": group["symbol"],
                "consumer_symbols": group["consumer_symbols"],
                "kernel_symbols": group["kernel_symbols"],
                "source_regions": group["source_regions"],
            }
        )
    result.verify()
    return result, records


class MXProjectionGroupPostOffload:
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
            or not catalog.get("group_projections")
        ):
            raise ValueError("projection group catalog differs from source routing")
        result, records = rewrite_projection_groups(parse_mlir_file(prepared), catalog)
        output = Path(workdir) / "mx_projection_groups.mlir"
        output.write_text(text(result))
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        output.with_name("mx_projection_groups_rewrite.json").write_text(
            json.dumps(
                {
                    "schema": "mx_gemmini.projection_groups.v1",
                    "source_sha256": digest(prepared),
                    "output_sha256": digest(output),
                    "catalog_sha256": digest(self.catalog_path),
                    "sidecar_sha256": digest(sidecar),
                    "groups": records,
                },
                indent=2,
            )
            + "\n"
        )
        return output
