"""Realize private BF16 dispatch handoffs using checked MX VPU fragments.

This adapter accepts Merlin's serialized dispatch DAG without importing
Merlin. Physical rows, operand footprints, and commands come from the typed
VPU lowerer; the graph decides whether an intermediate is observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import math
import struct

from .command_ir import Command, Fence
from .vpu_elementwise_program import lower_vpu_elementwise_program
from .target_profile import profile_sha256


def _scalar_expression(function):
    """Read a pointwise scalar expression from the actual outlined function."""
    from xdsl.dialects.builtin import FloatAttr
    from xdsl.ir.affine import AffineMap

    block = function.body.block
    if len(block.args) != 1:
        raise ValueError("scalar VPU region requires one tensor input")
    values = {block.args[0]: ("input", "bf16")}

    def evaluate(ops, env):
        for op in ops:
            name = op.name
            if name == "arith.constant":
                attr = op.properties.get("value", op.attributes.get("value"))
                if not isinstance(attr, FloatAttr):
                    raise ValueError("scalar VPU constant must be floating point")
                env[op.results[0]] = ("constant", str(op.results[0].type), float(attr.value.data))
            elif name == "tensor.empty":
                env[op.results[0]] = ("empty",)
            elif name == "tensor.splat":
                env[op.results[0]] = env[op.operands[0]]
            elif name in {"arith.extf", "arith.truncf"}:
                env[op.results[0]] = ("cast", str(op.results[0].type), env[op.operands[0]])
            elif name in {"arith.mulf", "arith.addf"}:
                fastmath = op.properties.get("fastmath")
                if fastmath is not None and fastmath.data:
                    raise ValueError("scalar VPU region requires strict floating arithmetic")
                env[op.results[0]] = (name, str(op.results[0].type),
                                      env[op.operands[0]], env[op.operands[1]])
            elif name == "linalg.generic":
                if (len(op.outputs) != 1 or len(op.results) != 1
                        or any(mapping.data != AffineMap.identity(2) for mapping in op.indexing_maps)
                        or any(str(iterator) != "#linalg.iterator_type<parallel>" for iterator in op.iterator_types)):
                    raise ValueError("scalar VPU region requires identity parallel maps")
                inner = dict(zip(op.body.block.args, [env[value] for value in op.operands]))
                env[op.results[0]] = evaluate(op.body.block.ops, inner)
            elif name in {"linalg.yield", "func.return"} and len(op.operands) == 1:
                return env[op.operands[0]]
            else:
                raise ValueError(f"unsupported operation in scalar VPU region: {name}")
        raise ValueError("scalar VPU region has no result")

    expr = evaluate(block.ops, values)
    if expr[:2] == ("cast", "bf16"):
        expr = expr[2]
        compute_type = "f32"
        input_expr = ("cast", "f32", ("input", "bf16"))
    else:
        compute_type = "bf16"
        input_expr = ("input", "bf16")
    if (len(expr) != 4 or expr[0] not in {"arith.mulf", "arith.addf"}
            or expr[1] != compute_type or expr[2] != input_expr
            or expr[3][:2] != ("constant", compute_type)):
        raise ValueError("scalar VPU region is not one BF16 scalar multiply or add")
    scalar = expr[3][2]
    if not math.isfinite(scalar):
        raise ValueError("scalar VPU immediate must be finite BF16")
    try:
        bits = struct.unpack("<I", struct.pack("<f", scalar))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("scalar VPU immediate exceeds finite BF16 range") from error
    if compute_type == "f32" and bits & 0xffff:
        raise ValueError("scalar VPU f32 immediate is not exactly BF16")
    immediate = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16
    if immediate & 0x7f80 == 0x7f80:
        raise ValueError("scalar VPU immediate exceeds finite BF16 range")
    return ("muls" if expr[0] == "arith.mulf" else "adds"), immediate


def scalar_vpu_fragments(module, program: dict, profile: dict, *, source_sha256: str) -> list[dict]:
    """Compile strict BF16 scalar regions from typed MLIR, including explicit casts."""
    module.verify()
    functions = {op.sym_name.data: op for op in module.walk() if op.name == "func.func"}
    digest = profile_sha256(profile)
    policy = hashlib.sha256(b"bf16_stepwise_exact_dma_handoff_v1").hexdigest()
    fragments = []
    for index, node in enumerate(program["nodes"]):
        if len(node["inputs"]) != 1 or len(node["outputs"]) != 1:
            raise ValueError("scalar VPU dispatch requires one input and output")
        source = program["buffers"][node["inputs"][0]]
        result = program["buffers"][node["outputs"][0]]
        if (source["shape"] != result["shape"] or source["dtype"] != "bf16"
                or result["dtype"] != "bf16" or len(result["shape"]) != 2 or result["shape"][1] != 8):
            raise ValueError("scalar VPU dispatch requires BF16 rows of eight lanes")
        rows = result["shape"][0]
        function = functions[node["op"]]
        signature = function.function_type
        expected_type = (f"tensor<{rows}x8xbf16>",)
        if (tuple(str(t) for t in signature.inputs) != expected_type
                or tuple(str(t) for t in signature.outputs) != expected_type):
            raise ValueError("scalar VPU function signature differs from dispatch buffers")
        stride = (rows + 15) // 16 * 16
        kind, immediate = _scalar_expression(function)
        src, dst = (0, stride) if index % 2 == 0 else (stride, 0)
        text = f'''builtin.module attributes {{mx.profile_sha256 = "{digest}",
          mx.contract_sha256 = "{source_sha256}", mx.policy_sha256 = "{policy}",
          prov.quantization_manifest_sha256 = "{source_sha256}"}} {{
          func.func @kernel() {{
            "mx_gemmini.vpu_execute"() {{site_id = "{node['op']}", kind = "{kind}",
              src1_row = {src} : i32, src2_row = 0 : i32, dst_row = {dst} : i32,
              rows = {rows} : i32, reduction_length = 1 : i32, broadcast = false,
              immediate_bf16 = {immediate} : i32, profile_sha256 = "{digest}",
              contract_sha256 = "{source_sha256}", policy_sha256 = "{policy}",
              manifest_sha256 = "{source_sha256}"}} : () -> ()
            func.return
          }}
        }}\n'''
        fragments.append(dict(mlir=text, buffers=dict(
            src1=node["inputs"][0], src2=f"unused{index}", output=node["outputs"][0])))
    return fragments


@dataclass(frozen=True)
class ConnectedVpuProgram:
    commands: tuple[Command | Fence, ...]
    realized: frozenset[tuple[int, int, str]]
    removed_transfer_bytes: int


class VpuDispatchAdapter:
    """A conservative direct path for flat, contiguous BF16 VPU dispatches."""

    def __init__(self, program: dict, profile: dict, fragments: list[dict], *, measured_savings: dict | None = None):
        self.program = deepcopy(program)
        self.profile = deepcopy(profile)
        self.fragments = deepcopy(fragments)
        self.measured_savings = dict(measured_savings or {})
        nodes = program["nodes"]
        if len(nodes) != len(fragments):
            raise ValueError("VPU fragments must cover every dispatch in order")
        self.plans = []
        for node, fragment in zip(nodes, fragments):
            plan = lower_vpu_elementwise_program(fragment["mlir"], profile, fragment["buffers"])
            buffers = fragment["buffers"]
            if plan.second_dst_row is not None:
                raise ValueError("connected VPU adapter requires single-result operations")
            inputs = [buffers["src1"]] + ([buffers["src2"]] if plan.src2_rows else [])
            if (node["kind"] != "dispatch" or node.get("regions", 0)
                    or node["inputs"] != inputs or node["outputs"] != [buffers["output"]]):
                raise ValueError("typed VPU fragment differs from dispatch graph")
            spans = [(buffers["src1"], plan.rows), (buffers["output"], plan.output_rows)]
            if plan.src2_rows:
                spans.append((buffers["src2"], plan.src2_rows))
            for name, rows in spans:
                logical = program["buffers"][name]
                if logical["dtype"] != "bf16" or logical["shape"] != [rows, 8]:
                    raise ValueError("connected VPU buffers require contiguous BF16 rows")
            self.plans.append(plan)

    def handoff_facts(self, producer: int, consumer: int, buffer: str) -> dict | None:
        """Derive legality from graph uses, physical row identity, and profile."""
        nodes = self.program["nodes"]
        if not (0 <= producer < consumer < len(nodes)) or consumer != producer + 1:
            return None
        before, after = self.plans[producer], self.plans[consumer]
        uses = [i for i, node in enumerate(nodes) for value in node["inputs"] if value == buffer]
        if (nodes[producer]["outputs"] != [buffer] or uses != [consumer]
                or buffer in self.program["results"]
                or self.program["buffers"][buffer]["kind"] != "intermediate"
                or self.fragments[consumer]["buffers"]["src1"] != buffer
                or before.dst_row != after.src1_row or before.output_rows != after.rows):
            return None
        # A surviving second-operand DMA must not overwrite the resident input.
        if after.src2_rows and not (
            after.src2_row + after.src2_rows <= after.src1_row
            or after.src1_row + after.rows <= after.src2_row
        ):
            return None
        high_water = max(
            before.src1_row + before.rows, before.dst_row + before.output_rows,
            before.src2_row + before.src2_rows,
            after.src1_row + after.rows, after.dst_row + after.output_rows,
            after.src2_row + after.src2_rows,
        ) * 16
        return dict(
            interface="mx_bf16_resident_vpu", resident_bytes=high_water,
            capacity_bytes=self.profile["resources"]["scratchpad_bytes"],
            equivalent_to_materialized=True,
            saved_cycles=self.measured_savings.get((producer, consumer, buffer)),
        )

    def lower(self, selected: set[tuple[int, int, str]]) -> ConnectedVpuProgram:
        """Remove only the selected intermediate's DMA store and reload."""
        for edge in selected:
            if self.handoff_facts(*edge) is None:
                raise ValueError("selected VPU handoff has no legal direct path")
        commands = []
        removed = 0
        for index, plan in enumerate(self.plans):
            for command in plan.commands:
                omit = any(
                    command.rs1.buffer == buffer and (
                        (index == producer and command.funct == 3)
                        or (index == consumer and command.funct == 2)
                    ) for producer, consumer, buffer in selected
                )
                if omit:
                    removed += 1
                else:
                    commands.append(command)
            commands.append(Fence())
        expected = sum(2 * ((self.plans[p].output_rows + 15) // 16) for p, _, _ in selected)
        if removed != expected:
            raise ValueError("selected VPU handoff did not remove its exact DMA pair")
        original_vpu = [c for plan in self.plans for c in plan.commands if c.funct == 33]
        if [c for c in commands if isinstance(c, Command) and c.funct == 33] != original_vpu:
            raise ValueError("connected VPU lowering changed arithmetic commands")
        return ConnectedVpuProgram(tuple(commands), frozenset(selected), sum(
            self.plans[p].output_rows * 16 * 2 for p, _, _ in selected))
