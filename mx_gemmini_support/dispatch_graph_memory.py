"""Map a verified Merlin dispatch DAG to MX host-buffer liveness events.

This is the graph half of physical planning. Every dispatch boundary is an
explicit host-buffer read/write; the selected kernel's on-chip command effects
must be certified separately before this can prove scratchpad residency.
"""

from __future__ import annotations

import hashlib
import json
from math import prod

from .physical_memory_plan import (
    Access, MemoryEvent, MemoryValue, PhysicalMemoryError, plan_physical_memory,
)
from .target_profile import profile_sha256


_ELEMENT_BYTES = {"f32": 4, "bf16": 2}
_ALIAS_VIEWS = {"tensor.collapse_shape", "tensor.expand_shape"}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def dispatch_graph_memory_plan(program, profile: dict, *, source_mlir_sha256: str,
                               recipe_sha256: str, driver_sha256: str,
                               dispatch_source_sha256: dict[str, str]) -> dict:
    """Prove static graph use/def, view aliases, captures, and host live ranges.

    ``program`` is Merlin's ``DispatchProgram``. Source/recipe/driver hashes
    and a hash of each emitted dispatch source are required for the receipt.
    The planner does not infer accelerator effects from a dispatch symbol.
    """
    from merlin.xdsl_dialects.lowering.dispatch_program import verify_program

    if problems := verify_program(program):
        raise PhysicalMemoryError("invalid Merlin dispatch DAG: " + "; ".join(problems))
    if not program.nodes or not program.results or not program.buffers:
        raise PhysicalMemoryError("dispatch graph needs nodes, buffers and results")
    if any(not isinstance(value, str) or len(value) != 64 or
           any(ch not in "0123456789abcdef" for ch in value)
           for value in (source_mlir_sha256, recipe_sha256, driver_sha256)):
        raise PhysicalMemoryError("dispatch graph needs exact source/recipe/driver SHA-256")
    if len(set(program.args)) != len(program.args):
        raise PhysicalMemoryError("dispatch graph argument indices repeat")
    arg_indices = sorted(b.arg_index for b in program.buffers.values() if b.kind == "arg")
    if arg_indices != sorted(program.args):
        raise PhysicalMemoryError("dispatch graph arguments differ from buffer table")
    written: set[str] = set()
    values: list[MemoryValue] = []
    sizes: dict[str, int] = {}
    for bid, buffer in program.buffers.items():
        if (buffer.id != bid or buffer.dtype not in _ELEMENT_BYTES or
                not buffer.shape or any(type(d) is not int or d <= 0
                                        for d in buffer.shape) or
                buffer.kind not in {"arg", "intermediate"}):
            raise PhysicalMemoryError("dispatch graph buffer needs static supported tensor type")
        sizes[bid] = prod(buffer.shape) * _ELEMENT_BYTES[buffer.dtype]
    aliases: dict[str, str] = {}
    events: list[MemoryEvent] = []
    expected_dispatches = {node.op for node in program.nodes if node.kind == "dispatch"}
    if set(dispatch_source_sha256) != expected_dispatches or any(
            type(digest) is not str or len(digest) != 64 or
            any(ch not in "0123456789abcdef" for ch in digest)
            for digest in dispatch_source_sha256.values()):
        raise PhysicalMemoryError("every dispatch needs one emitted-source SHA-256")
    for bid, buffer in program.buffers.items():
        if buffer.kind == "arg":
            values.append(MemoryValue(bid, "host", sizes[bid],
                                      f"Merlin argument {buffer.arg_index}", external=True))
    for index, node in enumerate(program.nodes):
        if (node.kind not in {"dispatch", "view"} or not node.op or
                (node.regions and (node.captures is None or
                                   not set(node.captures).issubset(node.inputs))) or
                any(bid in written for bid in node.outputs) or
                len(set(node.outputs)) != len(node.outputs)):
            raise PhysicalMemoryError("dispatch node has unknown captures or repeated definition")
        for bid in node.outputs:
            if bid not in program.buffers or program.buffers[bid].kind != "intermediate":
                raise PhysicalMemoryError("dispatch output has no intermediate buffer")
            written.add(bid)
        if node.kind == "view":
            if (node.op not in _ALIAS_VIEWS or node.regions or node.captures or
                    len(node.inputs) != 1 or len(node.outputs) != 1 or
                    sizes[node.inputs[0]] != sizes[node.outputs[0]] or
                    program.buffers[node.inputs[0]].dtype !=
                    program.buffers[node.outputs[0]].dtype):
                raise PhysicalMemoryError("view has no proved equal-byte alias semantics")
            out, src = node.outputs[0], node.inputs[0]
            aliases[out] = src
            values.append(MemoryValue(out, "host", sizes[out],
                                      f"Merlin node {index}: {node.op}", alias_of=src))
            events.append(MemoryEvent(f"node_{index}", "view", node.op,
                                      writes=(Access(out, 0, sizes[out]),)))
        else:
            for bid in node.outputs:
                values.append(MemoryValue(bid, "host", sizes[bid],
                                          f"Merlin node {index}: {node.op}"))
            events.append(MemoryEvent(
                f"node_{index}", "host", f"completed dispatch {node.op}",
                reads=tuple(Access(bid, 0, sizes[bid]) for bid in node.inputs),
                writes=tuple(Access(bid, 0, sizes[bid]) for bid in node.outputs)))
    if written != {bid for bid, b in program.buffers.items() if b.kind == "intermediate"}:
        raise PhysicalMemoryError("dispatch graph contains an unproduced intermediate")
    events.append(MemoryEvent(
        "return", "host", "Merlin program result retention",
        reads=tuple(Access(bid, 0, sizes[bid]) for bid in program.results)))
    bindings = {
        "profile_sha256": profile_sha256(profile),
        "source_mlir_sha256": source_mlir_sha256,
        "recipe_sha256": recipe_sha256,
        "driver_sha256": driver_sha256,
        "dispatch_program_sha256": _digest(program.to_dict()),
        "dispatch_sources_sha256": _digest(dispatch_source_sha256),
    }
    physical = plan_physical_memory(profile, tuple(values), tuple(events), bindings=bindings)
    physical.update(
        schema="mx_gemmini.dispatch_graph_memory.v1",
        status="verified_merlin_dag_host_buffer_liveness_only",
        scope="source-bound dispatch boundary host buffers; accelerator command effects are separate",
        program_sha256=bindings["dispatch_program_sha256"],
        program=program.to_dict(),
        alias_edges=aliases,
        dispatch_source_sha256=dict(dispatch_source_sha256),
    )
    return physical
