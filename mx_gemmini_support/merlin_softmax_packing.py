"""Stream private hybrid softmax probabilities into batch-local FP8 matrices."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dense_kernel import emit_dense_f32_kernel
from .scheduled_native_issuer import emit_scheduled_native_issuer
from .vpu_row_softmax import emit_vpu_row_softmax


def register_softmax_packing(
    softmax, kernels, bindings, profile, compile_source, abi_entries
):
    from merlin.llvmlower.device_shim import LogicalBuffer

    by_kernel = {k["symbol"]: k for k in kernels}
    records = {}
    for binding in bindings:
        compact = binding.get("batch_rhs_compact")
        kernel = by_kernel[binding["symbol"]]
        if not compact or not kernel["batched"] or kernel["precision"] != "mxfp8":
            continue
        d = kernel["dimensions"]
        batch, m, n, k = (d[x] for x in ("batch", "m", "n", "k"))
        if kernel["padded_shape_mnk"] != [m, n, k]:
            continue
        for producer in softmax:
            if (
                producer["implementation"] not in {"f32_reduce", "f32_striped16"}
                or producer["logical_rows"] != batch * m
                or producer["row_width"] != k
                or k % 32
            ):
                continue
            if k > 8192:
                continue
            chunk_rows = min(producer["logical_rows"], 8192 // k)
            original_launches = (batch * m + chunk_rows - 1) // chunk_rows
            serial_launches = batch * ((m + chunk_rows - 1) // chunk_rows)
            # Serial matrices must not add EXP launches to the original batch.
            # A smaller temporary buffer alone did not pay for that increase.
            if serial_launches > original_launches:
                continue
            key = (producer["symbol"], kernel["symbol"], compact["repeat"])
            if key not in records:
                entry = f"mx_softmax_packed_{len(records)}"
                generated = emit_vpu_row_softmax(
                    profile,
                    logical_rows=m,
                    row_width=k,
                    symbol=entry + "_softmax",
                    implementation=producer["implementation"],
                    packed_matrix_rows=m,
                )
                packed_bytes = m * k * 33 // 32
                issuer_symbol = Path(kernel["schedule_file"]).stem
                issuer = emit_scheduled_native_issuer(
                    profile,
                    (m, n, k),
                    precision="FP8",
                    symbol=issuer_symbol,
                    tile_mnk=tuple(kernel["tile_mnk"]),
                ).issuer
                matrix = emit_dense_f32_kernel(
                    issuer,
                    (m, n, k),
                    symbol=entry + "_matrix_slice",
                    issuer_symbol=issuer_symbol,
                    shared_workspace_symbol="mx_catalog_workspace",
                    retain_runtime_weight=True,
                    prepared_activation=True,
                )
                if matrix.workspace_bytes > compact["workspace_bytes"]:
                    raise ValueError(
                        "prepared softmax matrix increased its private serial workspace"
                    )
                source = (
                    generated.source
                    + matrix.source
                    + f"""
static uint8_t {entry}_packed[{packed_bytes}] __attribute__((aligned(64)));
void {entry}_kernel(const float *scores,const float *rhs,float *output) {{
  if (!scores || !rhs || !output) __builtin_trap();
  for (uint32_t batch=0; batch<{batch}u; ++batch) {{
    {entry}_softmax(scores+batch*{m * k}u,{entry}_packed);
    {entry}_matrix_slice({entry}_packed,rhs+(batch/{compact["repeat"]}u)*{k * n}u,
                         output+batch*{m * n}u,batch%{compact["repeat"]}u==0u);
  }}
}}
"""
                )
                compile_source(entry + "_kernel", source)
                abi_entries[entry] = (
                    entry + "_kernel",
                    (
                        LogicalBuffer(
                            "scores",
                            "input",
                            "input_0",
                            tuple(producer["shape"]),
                            "f32",
                        ),
                        LogicalBuffer(
                            "rhs",
                            "input",
                            "input_1",
                            (batch // compact["repeat"], k, n),
                            "f32",
                        ),
                        LogicalBuffer("out", "output", "out_0", (batch, m, n), "f32"),
                    ),
                )
                records[key] = {
                    "symbol": entry,
                    "kernel_symbol": entry + "_kernel",
                    "original_softmax": producer["symbol"],
                    "original_consumers": [],
                    "source_kernel": kernel["symbol"],
                    "shape": producer["shape"],
                    "matrix_shape_bmnk": [batch, m, n, k],
                    "rhs_repeat": compact["repeat"],
                    "packed_bytes": packed_bytes,
                    "packed_lifetime": "one_serial_matrix",
                    "selection": "nonincreasing_exp_launches",
                    "unfused_exp_launches": original_launches,
                    "fused_exp_launches": serial_launches,
                    "matrix_batch": batch,
                    "workspace_bytes": matrix.workspace_bytes,
                    "numeric_policy": "unchanged_hybrid_softmax_then_fp8_per_matrix_block32",
                    "completion": "synchronous_serial_no_persistent_cross_call_state",
                }
            consumers = records[key]["original_consumers"]
            if compact["entry_symbol"] not in consumers:
                consumers.append(compact["entry_symbol"])
    return list(records.values())


def rewrite_softmax_packing(module, catalog):
    from merlin.xdsl_dialects.lowering.shared_preparation import contiguous_tensor_root
    from merlin.xdsl_dialects.lowering.tensor_chains import (
        destination_only_tensor_call,
        pure_tensor_compute,
    )
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr
    from xdsl.ir import Region
    from xdsl.ir.affine import AffineMap

    result = module.clone()
    declarations = {
        o.sym_name.data: o for o in result.body.block.ops if isinstance(o, func.FuncOp)
    }
    records = []

    def declare(symbol, call):
        types = tuple(v.type for v in call.arguments)
        outputs = tuple(v.type for v in call.results)
        attrs = ArrayAttr(
            [DictionaryAttr({"bufferization.access": StringAttr("read")})]
            * (len(types) - 1)
            + [DictionaryAttr({"bufferization.access": StringAttr("write")})]
        )
        old = declarations.get(symbol)
        if old is not None:
            if (
                old.body.blocks
                or tuple(old.function_type.inputs) != types
                or tuple(old.function_type.outputs) != outputs
                or old.arg_attrs != attrs
            ):
                raise ValueError(
                    "softmax packing collides with an incompatible declaration"
                )
        else:
            old = func.FuncOp(
                symbol,
                (types, outputs),
                Region(),
                visibility="private",
                arg_attrs=attrs,
            )
            result.body.block.add_op(old)
            declarations[symbol] = old

    for endpoint in catalog.get("softmax_packing", []):
        batch, m, n, k = endpoint["matrix_shape_bmnk"]
        expected_a = f"tensor<{batch}x{m}x{k}xf32>"
        expected_b = f"tensor<{batch // endpoint['rhs_repeat']}x{k}x{n}xf32>"
        expected_out = f"tensor<{batch}x{m}x{n}xf32>"
        expected_scores = "tensor<" + "x".join(map(str, endpoint["shape"])) + "xf32>"
        for consumer in tuple(result.walk()):
            if (
                not isinstance(consumer, func.CallOp)
                or consumer.callee.root_reference.data
                not in endpoint["original_consumers"]
                or len(consumer.arguments) != 3
                or not destination_only_tensor_call(consumer)
                or [str(v.type) for v in consumer.arguments]
                != [expected_a, expected_b, expected_out]
            ):
                continue
            operations = set()

            def flat_root(value, operations=operations):
                root = contiguous_tensor_root(value)
                while value is not root:
                    operations.add(value.owner)
                    value = value.owner.operands[0]
                op = root.owner
                if (
                    getattr(op, "name", None) == "linalg.generic"
                    and pure_tensor_compute(op)
                    and len(op.inputs) == len(op.outputs) == len(op.results) == 1
                    and op.inputs[0].type == root.type == op.outputs[0].type
                    and all(
                        str(it) == "#linalg.iterator_type<parallel>"
                        for it in op.iterator_types
                    )
                    and tuple(m.data for m in op.indexing_maps)
                    == (AffineMap.identity(len(root.type.get_shape())),) * 2
                    and len(tuple(op.body.block.ops)) == 1
                    and op.body.block.last_op.name == "linalg.yield"
                    and tuple(op.body.block.last_op.operands)
                    == (op.body.block.args[0],)
                ):
                    operations.add(op)
                    return flat_root(op.inputs[0])
                return root

            root = flat_root(consumer.arguments[0])
            producer = root.owner
            if (
                not isinstance(producer, func.CallOp)
                or not destination_only_tensor_call(producer)
                or len(producer.arguments) != 2
                or producer.callee.root_reference.data != endpoint["original_softmax"]
                or str(producer.arguments[0].type) != expected_scores
                or str(producer.results[0].type) != expected_scores
                or producer.parent is not consumer.parent
            ):
                continue
            operations.add(producer)
            if any(
                use.operation not in operations | {consumer}
                for op in operations
                for v in op.results
                for use in v.uses
            ):
                continue
            cursor = producer
            while cursor is not consumer and cursor is not None:
                if cursor not in operations and not pure_tensor_compute(cursor):
                    break
                cursor = cursor.next_op
            if cursor is not consumer:
                continue
            matrix = func.CallOp(
                endpoint["symbol"],
                [producer.arguments[0], consumer.arguments[1], consumer.arguments[2]],
                [consumer.results[0].type],
            )
            declare(endpoint["symbol"], matrix)
            block = consumer.parent
            block.insert_op_before(matrix, consumer)
            consumer.results[0].replace_all_uses_with(matrix.results[0])
            original = consumer.callee.root_reference.data
            consumer.detach()
            consumer.erase()
            for op in reversed([o for o in block.ops if o in operations]):
                op.detach()
                op.erase()
            records.append(
                {
                    "symbol": endpoint["symbol"],
                    "original_softmax": endpoint["original_softmax"],
                    "original_consumer": original,
                    "source_kernel": endpoint["source_kernel"],
                    "packed_bytes": endpoint["packed_bytes"],
                }
            )
    result.verify()
    return result, records


class MXSoftmaxPackingPostOffload:
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
            or not catalog.get("fuse_softmax_packing")
        ):
            raise ValueError("softmax packing differs from source routing")
        result, records = rewrite_softmax_packing(parse_mlir_file(prepared), catalog)
        output = Path(workdir) / "mx_softmax_packing.mlir"
        output.write_text(text(result))
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        output.with_name("mx_softmax_packing_rewrite.json").write_text(
            json.dumps(
                {
                    "schema": "mx_gemmini.softmax_packing.v1",
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
