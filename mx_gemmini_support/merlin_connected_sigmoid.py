"""Bind resident sigmoid chains to selected source arithmetic and catalog entries."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .connected_sigmoid_multiply import (
    check_sigmoid_multiply,
    emit_connected_sigmoid_multiply,
)
from .merlin_connected_multiply import multiply_pairs
from .target_profile import profile_sha256


def source_program(chain):
    """Outline the actual three generics without changing their scalar bodies."""
    from merlin.xdsl_dialects.lowering.dispatch_program import (
        Buffer,
        DispatchProgram,
        Node,
    )
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    functions = []
    for name, op in zip(("sigmoid", "multiply0", "multiply1"), chain["operations"], strict=True):
        block = Block(arg_types=[value.type for value in op.inputs])
        empty = tensor.EmptyOp((), op.results[0].type)
        mapping = dict(zip(op.inputs, block.args, strict=True))
        mapping[op.outputs[0]] = empty.results[0]
        copied = op.clone(value_mapper=mapping)
        block.add_ops([empty, copied, func.ReturnOp(*copied.results)])
        functions.append(func.FuncOp(name, ([v.type for v in op.inputs], [op.results[0].type]), Region(block)))
    first_inputs = ["x", "s"] if chain["gate_first"] else ["s", "x"]
    graph = DispatchProgram("forward", [0, 1], {
        name: Buffer(name, chain["shape"], "f32", "arg" if name in ("x", "u") else "intermediate",
                     ("x", "u").index(name) if name in ("x", "u") else None)
        for name in ("x", "u", "s", "m", "out")}, [
            Node("dispatch", "sigmoid", ["x"], ["s"]),
            Node("dispatch", "multiply0", first_inputs, ["m"]),
            Node("dispatch", "multiply1", ["m", "u"], ["out"])], ["out"])
    return ModuleOp(functions), graph


def sigmoid_chains(module, elementwise_sites, unary_sites):
    """Match selected typed sigmoid/multiply semantics; IDs bind sites only."""
    from merlin.xdsl_dialects.lowering.tensor_chains import (
        destination_only_tensor_call,
        private_tensor_chain,
        pure_tensor_compute,
    )

    by_source = defaultdict(list)
    for site in unary_sites:
        if site["operation"] == "sigmoid" and site["dtype"] == "f32":
            by_source[site["source_region"]].append(site)
    result, used = [], set()
    for pair in multiply_pairs(module, elementwise_sites):
        first, last = pair["first"], pair["second"]
        for index in (0, 1):
            producer = first.inputs[index].owner
            if (getattr(producer, "name", None) != "linalg.generic" or producer in used
                    or len(producer.inputs) != 1 or producer.inputs[0] != first.inputs[1 - index]):
                continue
            source = getattr(producer.attributes.get("prov.region_id"), "data", None)
            sites = by_source.get(source, [])
            if len(sites) != 1 or list(sites[0]["shape"]) != pair["shape"]:
                continue
            operations = (producer, first, last)
            if not private_tensor_chain(operations, can_cross=lambda op:
                    pure_tensor_compute(op) or destination_only_tensor_call(op)):
                continue
            order = "gate_first" if index == 1 else "sigmoid_first"
            symbol = "mx_vpu_sigmoid_mul_chain_f32_" + order + "_" + "_".join(map(str, pair["shape"]))
            chain = {"operations": operations, "shape": pair["shape"], "regions": pair["regions"],
                     "unary_region": sites[0]["region"], "gate_first": index == 1,
                     "symbol": symbol, "kernel_symbol": symbol + "_kernel",
                     "inputs": (producer.inputs[0], last.inputs[1])}
            outlined, program = source_program(chain)
            try:
                check_sigmoid_multiply(outlined, program.to_dict())
            except ValueError:
                continue
            result.append(chain)
            used.update(operations)
            break
    return result


def chain_record(chain):
    return {key: chain[key] for key in ("regions", "unary_region", "shape", "symbol", "kernel_symbol")}


def register_chain(chain, profile, compile_source, abi_entries):
    from merlin.llvmlower.device_shim import LogicalBuffer

    module, program = source_program(chain)
    code = emit_connected_sigmoid_multiply(module, program.to_dict(), profile,
                                           symbol=chain["kernel_symbol"], boundary_policy="bf16_rne_each_vpu_call")
    compile_source(chain["kernel_symbol"], code.source)
    abi_entries[chain["symbol"]] = (chain["kernel_symbol"], (
        LogicalBuffer("gate", "input", "input_0", tuple(chain["shape"]), "f32"),
        LogicalBuffer("up", "input", "input_1", tuple(chain["shape"]), "f32"),
        LogicalBuffer("out", "output", "out_0", tuple(chain["shape"]), "f32")))
    return {"removed_transfer_bytes": code.removed_transfer_bytes, "resident_bytes": code.resident_bytes}


class MXConnectedSigmoidPostOffload:
    """Fuse pure source operations before ordinary unary/binary call lowering."""

    def __init__(self, catalog_path, profile):
        self.catalog_path = Path(catalog_path)
        self._profile = profile

    def __call__(self, prepared, sidecar, workdir):
        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text
        from merlin.xdsl_dialects.lowering.tensor_chains import (
            destination_only_tensor_call,
            pure_tensor_compute,
            replace_private_tensor_chain,
        )

        from .merlin_vpu_elementwise import selected_elementwise_sites
        from .merlin_vpu_unary import selected_unary_sites

        prepared, sidecar, workdir = map(Path, (prepared, sidecar, workdir))
        catalog = json.loads(self.catalog_path.read_text())
        routing = json.loads(sidecar.read_text())
        if (Path(routing["catalog_manifest"]).resolve() != self.catalog_path.resolve()
                or routing["model_sha256"] != catalog["source_sha256"]
                or profile_sha256(self._profile) != catalog["profile_sha256"]
                or not catalog.get("connect_sigmoid_multiplies")):
            raise ValueError("sigmoid chain catalog differs from source routing")
        resolved = catalog["resolved_vpu_regions"]
        source = prepared.read_text()
        elements = [s for s in selected_elementwise_sites(source, self._profile)
                    if s["region"] in resolved["elementwise"]]
        unary = [s for s in selected_unary_sites(source, self._profile) if s["region"] in resolved["unary"]]
        module = parse_mlir_file(prepared)
        chains = sigmoid_chains(module, elements, unary)
        records = [chain_record(chain) for chain in chains]
        planned = catalog["vpu_elementwise"]["connected_sigmoid_multiplies"]
        expected = [{key: row[key] for key in
                     ("regions", "unary_region", "shape", "symbol", "kernel_symbol")} for row in planned]
        if records != expected:
            raise ValueError("sigmoid chain rewrite differs from the exact compiled source obligations")
        for chain in chains:
            replace_private_tensor_chain(module, chain["operations"], symbol=chain["symbol"],
                inputs=chain["inputs"], can_cross=lambda op:
                    pure_tensor_compute(op) or destination_only_tensor_call(op))
        workdir.mkdir(parents=True, exist_ok=True)
        output = workdir / "mx_connected_sigmoid.mlir"
        output.write_text(text(module))
        (workdir / "mx_connected_sigmoid_rewrite.json").write_text(json.dumps({
            "schema": "mx_gemmini.connected_sigmoid_source_rewrite.v1", "chains": records,
            "source_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "catalog_sha256": hashlib.sha256(self.catalog_path.read_bytes()).hexdigest(),
            "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()}, indent=2) + "\n")
        return output, records
