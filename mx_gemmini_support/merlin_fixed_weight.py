"""Remove host preparation of RHS operands embedded in an MX catalog."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .weight_provenance import source_parameter_arg


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _erase_dead_operand_tree(value) -> int:
    """Erase only dead pure operations on a proven fixed-weight path."""
    pending = [value]
    erased = 0
    while pending:
        current = pending.pop()
        owner = current.owner
        if (getattr(owner, "name", None) not in
                {"linalg.generic", "linalg.transpose", "tensor.empty"}
                or any(result.uses for result in owner.results)):
            continue
        pending.extend(owner.operands)
        owner.detach()
        owner.erase()
        erased += 1
    return erased


class MXFixedWeightPostOffload:
    """Keep the MX pointer ABI while removing unused fixed-RHS computation."""

    def __init__(self, catalog_path: Path):
        self.catalog_path = Path(catalog_path)

    def __call__(self, prepared: Path, sidecar: Path, workdir: Path) -> Path:
        from xdsl.dialects import func, tensor

        from merlin.frontends.linalg_mlir import parse_mlir_file
        from merlin.xdsl_dialects._common import text

        prepared, sidecar, workdir = map(Path, (prepared, sidecar, workdir))
        source_bytes = prepared.read_bytes()
        routing = json.loads(sidecar.read_text())
        if Path(routing["catalog_manifest"]).resolve() != self.catalog_path.resolve():
            raise ValueError("MX fixed RHS catalog differs from routing sidecar")
        catalog = json.loads(self.catalog_path.read_text())
        if catalog["source_sha256"] != routing["model_sha256"]:
            raise ValueError("MX fixed RHS catalog differs from offload binding")
        sources = {row["region"]: row for row in catalog["bindings"]}
        routed = routing["routed"]
        if len(sources) != len(catalog["bindings"]) or len(routed) != len(sources):
            raise ValueError("MX fixed RHS source binding count changed")
        entry_to_kernel = {}
        for row in routed:
            source_binding = sources.get(row["source_region"])
            if (source_binding is None or
                    source_binding["source_operation_ordinal"] != row["source_operation_ordinal"] or
                    source_binding["tensor_types"] != row["tensor_types"] or
                    entry_to_kernel.get(row["symbol"], source_binding["symbol"])
                    != source_binding["symbol"]):
                raise ValueError("MX fixed RHS source binding changed")
            entry_to_kernel[row["symbol"]] = source_binding["symbol"]
        kernels = {entry["symbol"]: entry for entry in catalog["kernels"]
                   if entry.get("prepacked_checkpoint_weight")}
        selected = {entry: kernels[kernel] for entry, kernel in
                    entry_to_kernel.items() if kernel in kernels}
        expected = Counter(row["symbol"] for row in catalog["bindings"]
                           if row["symbol"] in kernels)
        if sum(expected.values()) != len(selected):
            raise ValueError("MX fixed RHS entry count differs from source bindings")
        module = parse_mlir_file(prepared)
        forwards = [op for op in module.walk() if isinstance(op, func.FuncOp)
                    and op.body.blocks and op.sym_name.data == "forward"]
        if len(forwards) != 1:
            raise ValueError("MX fixed RHS rewrite needs one forward function")
        block = forwards[0].body.block
        calls = [op for op in block.ops if isinstance(op, func.CallOp)
                 and op.callee.root_reference.data in selected]
        found = Counter(selected[op.callee.root_reference.data]["symbol"] for op in calls)
        if found != expected:
            raise ValueError("MX fixed RHS calls differ from source bindings")
        records = []
        for call in calls:
            entry = call.callee.root_reference.data
            kernel = selected[entry]
            dimensions = kernel["dimensions"]
            rhs_type = f"tensor<{dimensions['k']}x{dimensions['n']}xf32>"
            if (kernel["batched"] or len(call.arguments) != 3
                    or str(call.arguments[1].type) != rhs_type):
                raise ValueError(f"MX fixed RHS call changed type: {entry}")
            original = call.arguments[1]
            argument, transforms, source_shape = source_parameter_arg(original)
            if (argument != kernel["checkpoint_parameter_arg"] or
                    list(source_shape or ()) != kernel["checkpoint_parameter_shape"] or
                    list(transforms) != kernel["checkpoint_parameter_transforms"]):
                raise ValueError(f"MX fixed RHS provenance changed: {entry}")
            placeholder = tensor.EmptyOp((), original.type)
            block.insert_op_before(placeholder, call)
            call.operands[1] = placeholder.results[0]
            erased = _erase_dead_operand_tree(original)
            records.append({"entry": entry, "kernel": kernel["symbol"],
                            "checkpoint_parameter": kernel["checkpoint_parameter_name"],
                            "erased_producers": erased})
        workdir.mkdir(parents=True, exist_ok=True)
        output = workdir / "mx_fixed_rhs.mlir"
        output.write_text(text(module), encoding="utf-8")
        (workdir / "mx_fixed_rhs_rewrite.json").write_text(json.dumps({
            "schema": "mx_gemmini.fixed_rhs_rewrite.v1",
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "output_sha256": _sha(output),
            "sidecar_sha256": _sha(sidecar),
            "catalog_sha256": _sha(self.catalog_path),
            "rewritten_calls": len(records),
            "erased_producers": sum(row["erased_producers"] for row in records),
            "calls": records,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output
