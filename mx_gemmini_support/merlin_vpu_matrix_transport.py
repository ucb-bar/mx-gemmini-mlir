"""Bind qualified compact VPU outputs to exact MX activation preparations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _tensor(shape, dtype):
    return "tensor<" + "".join(f"{d}x" for d in shape) + dtype + ">"


def register_producer(pair, profile, selection, compile_source, abi_entries):
    from merlin.llvmlower.device_shim import LogicalBuffer

    from .merlin_connected_multiply import emit_chain_kernel

    entry = pair["symbol"] + "_bf16_output"
    kernel = entry + "_kernel"
    source, _ = emit_chain_kernel(pair | {"symbol": entry, "kernel_symbol": kernel}, profile,
                                 output_dtype="bf16", **selection)
    compile_source(kernel, source)
    inputs = tuple(LogicalBuffer(name, "input", f"input_{i}", tuple(shape), "f32")
                   for i, (name, shape) in enumerate(zip("abc", pair["input_shapes"], strict=True)))
    abi_entries[entry] = (kernel, (*inputs, LogicalBuffer("out", "output", "out_0",
                                                       tuple(pair["shape"]), "bf16")))
    return {"symbol": entry, "kernel_symbol": kernel,
            "original_argument_types": [_tensor(s, "f32") for s in [*pair["input_shapes"], pair["shape"]]],
            "rounding": "unchanged_bf16_vpu_readback", "completion": "synchronous_native_issuer"}


def register_preparation(generated, shape, prepare_entry, compile_source, abi_entries, sources):
    from merlin.llvmlower.device_shim import LogicalBuffer

    from .dense_kernel import emit_activation_preparation

    entry, kernel = prepare_entry + "_bf16_input", prepare_entry + "_bf16_input_kernel"
    source, packed_bytes, padding_bytes = emit_activation_preparation(
        generated.issuer, shape, symbol=kernel, input_dtype="bf16")
    previous = sources.get(entry)
    if previous is None:
        compile_source(kernel, source)
        sources[entry] = source
    elif previous != source:
        raise ValueError("compact preparation differs across compatible consumers")
    m, _, k = shape
    abi_entries[entry] = (kernel, (
        LogicalBuffer("activation", "input", "input_0", (m, k), "bf16"),
        LogicalBuffer("packed", "output", "out_0", (packed_bytes,), "i8")))
    return {"symbol": entry, "kernel_symbol": kernel,
            "original_argument_types": [_tensor((m, k), "f32"), _tensor((packed_bytes,), "i8")],
            "packed_bytes": packed_bytes, "padding_workspace_bytes": padding_bytes,
            "completion": "synchronous_native_issuer"}


class MXVPUMatrixTransportPostOffload:
    """Keep only private checked VPU-to-preparation edges in BF16 storage."""

    def __init__(self, catalog_path):
        self.catalog_path = Path(catalog_path)

    def __call__(self, prepared, sidecar, workdir):
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text
        from merlin.xdsl_dialects.lowering.private_transport import (
            PrivateTensorTransport,
            prune_unobserved_transport_views,
            rewrite_private_transports,
        )
        from xdsl.dialects.builtin import bf16

        prepared, sidecar, workdir = map(Path, (prepared, sidecar, workdir))
        catalog = json.loads(self.catalog_path.read_text())
        routing = json.loads(sidecar.read_text())
        if (Path(routing["catalog_manifest"]).resolve() != self.catalog_path.resolve()
                or routing["model_sha256"] != catalog["source_sha256"]
                or catalog.get("vpu_matrix_transport") != "experimental_private_bf16"):
            raise ValueError("VPU matrix transport differs from selected catalog/source routing")
        bindings = {r["region"]: r for r in catalog["bindings"]}
        if len(bindings) != len(catalog["bindings"]) or len(routing["routed"]) != len(bindings):
            raise ValueError("VPU matrix transport source binding count changed")
        for row in routing["routed"]:
            bound = bindings.get(row["source_region"])
            if (bound is None or bound["source_operation_ordinal"] != row["source_operation_ordinal"]
                    or bound["tensor_types"] != row["tensor_types"]):
                raise ValueError("VPU matrix transport source binding changed")
        producers, consumers = {}, {}
        for row in catalog["vpu_elementwise"]["connected_multiplies"]:
            if row.get("private_bf16_output"):
                endpoint = row["private_bf16_output"]
                if producers.setdefault(row["symbol"], endpoint) != endpoint:
                    raise ValueError("compact producer binding is inconsistent")
        for row in catalog["kernels"]:
            packing = row.get("activation_preparation")
            if packing and packing.get("private_bf16_input"):
                if row["batched"] or not row["prepacked_checkpoint_weight"]:
                    raise ValueError("compact preparation needs immutable regular matrix operands")
                endpoint = packing["private_bf16_input"]
                if consumers.setdefault(packing["prepare_symbol"], endpoint) != endpoint:
                    raise ValueError("compact preparation binding is inconsistent")

        def select(producer, consumer, index):
            first = producers.get(producer.callee.root_reference.data)
            second = consumers.get(consumer.callee.root_reference.data)
            if first is None or second is None:
                return None
            if (index != 0 or [str(v.type) for v in producer.arguments] != first["original_argument_types"]
                    or [str(v.type) for v in consumer.arguments] != second["original_argument_types"]
                    or first["completion"] != "synchronous_native_issuer"
                    or second["completion"] != "synchronous_native_issuer"):
                raise ValueError("compact endpoint type, position or completion changed")
            return PrivateTensorTransport(first["symbol"], second["symbol"], bf16)

        cleaned, removed = prune_unobserved_transport_views(parse_mlir_file(prepared))
        result, edges = rewrite_private_transports(cleaned, select)
        workdir.mkdir(parents=True, exist_ok=True)
        output = workdir / "mx_vpu_matrix_transport.mlir"
        output.write_text(text(result))
        digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        (workdir / "mx_vpu_matrix_transport_rewrite.json").write_text(json.dumps({
            "schema": "mx_gemmini.private_vpu_matrix_transport.v1",
            "source_sha256": digest(prepared), "output_sha256": digest(output),
            "catalog_sha256": digest(self.catalog_path), "sidecar_sha256": digest(sidecar),
            "selection": "experimental_private_edge_legality_only",
            "removed_unused_static_views": removed, "edges": edges,
        }, indent=2, sort_keys=True) + "\n")
        return output
