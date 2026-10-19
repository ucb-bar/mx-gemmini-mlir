"""Retain the existing BF16 sigmoid and two multiply boundaries in scratchpad.

The source is checked as sigmoid(x), x*sigmoid(x), then multiplication by a
second tensor. This changes transport only under the existing stepwise VPU
policy; it does not replace that policy with an FP32 sigmoid approximation.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from math import floor, log2, prod, sqrt

from .command_ir import emit_c
from .connected_elementwise import check_multiply_regions
from .connected_vpu import VpuDispatchAdapter
from .target_profile import profile_sha256
from .vpu_add_kernel import emit_vpu_elementwise
from .vpu_unary_kernel import emit_vpu_unary, sigmoid_steps


@dataclass(frozen=True)
class ConnectedSigmoidMultiply:
    source: str
    shape: tuple[int, ...]
    removed_transfer_bytes: int
    resident_bytes: int


def check_sigmoid_multiply(module, program):
    """Require the actual strict full-shape scalar bodies and private graph edges."""
    from xdsl.dialects.builtin import FloatAttr
    from xdsl.ir.affine import AffineMap

    module.verify()
    nodes = program["nodes"]
    if len(nodes) != 3 or any(node.get("regions", 0) or node.get("captures") for node in nodes):
        raise ValueError("sigmoid/multiply transport requires exactly three dispatches")
    shape = check_multiply_regions(module, {**program, "nodes": nodes[1:]})
    unary, first, second = nodes
    if (unary["kind"] != "dispatch" or len(unary["inputs"]) != 1 or len(unary["outputs"]) != 1
            or set(first["inputs"]) != {unary["inputs"][0], unary["outputs"][0]}
            or second["inputs"][0] != first["outputs"][0]
            or second["inputs"][1] == unary["inputs"][0]
            or program["results"] != second["outputs"]):
        raise ValueError("sigmoid/multiply graph has unsupported operands or observable intermediates")
    if (len(program["args"]) != 2 or any(program["buffers"][name]["kind"] != "arg"
            for name in (unary["inputs"][0], second["inputs"][1]))):
        raise ValueError("sigmoid/multiply requires two explicit external input tensors")
    for index in (0, 1):
        value = nodes[index]["outputs"][0]
        uses = [i for i, node in enumerate(nodes) for name in node["inputs"] if name == value]
        if uses != [index + 1] or program["buffers"][value]["kind"] != "intermediate":
            raise ValueError("sigmoid/multiply intermediates must be private single-use tensors")
    functions = {op.sym_name.data: op for op in module.walk() if op.name == "func.func"}
    function = functions[unary["op"]]
    tensor = "tensor<" + "x".join(map(str, shape)) + "xf32>"
    if (tuple(str(t) for t in function.function_type.inputs) != (tensor,)
            or tuple(str(t) for t in function.function_type.outputs) != (tensor,)
            or not function.body.blocks):
        raise ValueError("sigmoid function signature differs from the dispatch graph")
    block = function.body.block
    generics = [op for op in block.ops if op.name == "linalg.generic"]
    if len(generics) != 1 or any(op.name not in {"tensor.empty", "linalg.generic", "func.return"}
                                 for op in block.ops):
        raise ValueError("sigmoid requires one pure generic")
    op = generics[0]
    body = list(op.body.block.ops)
    if (list(op.inputs) != list(block.args) or len(op.outputs) != 1
            or len(op.indexing_maps) != 2
            or any(m.data != AffineMap.identity(len(shape)) for m in op.indexing_maps)
            or any(str(i) != "#linalg.iterator_type<parallel>" for i in op.iterator_types)
            or [step.name for step in body] != ["arith.constant", "arith.negf", "math.exp",
                                                "arith.addf", "arith.divf", "linalg.yield"]
            or list(block.last_op.operands) != list(op.results)):
        raise ValueError("sigmoid scalar body or indexing is unsupported")
    constant = body[0].properties.get("value")
    if (not isinstance(constant, FloatAttr) or float(constant.value.data) != 1.0
            or [list(step.operands) for step in body] != [[], [op.body.block.args[0]],
                [body[1].results[0]], [body[0].results[0], body[2].results[0]],
                [body[0].results[0], body[3].results[0]], [body[4].results[0]]]
            or any(step.properties.get("fastmath") is not None and step.properties["fastmath"].data
                   for step in body)):
        raise ValueError("sigmoid requires strict one/(one+exp(-x)) arithmetic")
    return shape, first["inputs"][0] == unary["inputs"][0]


def _resident(profile, count, source_digest):
    from merlin.xdsl_dialects.lowering.connected_dispatch import (
        lower_connected_dispatches,
    )
    from merlin.xdsl_dialects.lowering.dispatch_program import (
        Buffer,
        DispatchProgram,
        Node,
    )

    rows = (count + 7) // 8
    names = ("x", "u", "neg", "exp", "denominator", "sigmoid", "middle", "output")
    buffers = {name: {"shape": [rows, 8], "dtype": "bf16",
                      "kind": "arg" if name in ("x", "u") else "intermediate",
                      **({"arg_index": ("x", "u").index(name)} if name in ("x", "u") else {})}
               for name in names}
    inputs = (("x",), ("neg",), ("exp",), ("denominator",), ("sigmoid", "x"), ("middle", "u"))
    outputs = names[2:]
    nodes = [{"kind": "dispatch", "op": "step" + str(i), "inputs": list(ins),
              "outputs": [out]} for i, (ins, out) in enumerate(zip(inputs, outputs, strict=True))]
    dispatch = DispatchProgram("forward", [0, 1],
        {name: Buffer(name, value["shape"], value["dtype"], value["kind"], value.get("arg_index"))
         for name, value in buffers.items()}, [Node(**node) for node in nodes], ["output"])
    graph = dispatch.to_dict()
    steps = (*sigmoid_steps(),
             {"kind": "mul", "src1_row": 0x2000, "src2_row": 0, "dst_row": 0x1000},
             {"kind": "mul", "src1_row": 0x1000, "src2_row": 0x3000, "dst_row": 0x2000})
    digest = profile_sha256(profile)
    fragments = []
    for i, (node, step) in enumerate(zip(nodes, steps, strict=True)):
        attributes = {**step, "rows": rows, "reduction_length": 1,
                      "immediate_bf16": step.get("immediate_bf16", 0)}
        physical = ", ".join(f'{key} = "{value}"' if isinstance(value, str) else f"{key} = {value} : i32"
                             for key, value in attributes.items())
        text = f'''builtin.module attributes {{mx.profile_sha256="{digest}",
          mx.contract_sha256="{source_digest}", mx.policy_sha256="{source_digest}",
          prov.quantization_manifest_sha256="{source_digest}"}} {{
          func.func @step() {{
            "mx_gemmini.vpu_execute"() {{site_id="step{i}", {physical}, broadcast=false,
              profile_sha256="{digest}", contract_sha256="{source_digest}",
              policy_sha256="{source_digest}", manifest_sha256="{source_digest}"}} : () -> ()
            func.return
          }}
        }}'''
        fragments.append({"mlir": text, "buffers": {"src1": node["inputs"][0],
            "src2": node["inputs"][1] if len(node["inputs"]) == 2 else "unused" + str(i),
            "output": node["outputs"][0]}})
    adapter = VpuDispatchAdapter(graph, profile, fragments)
    selected = {(i, i + 1, outputs[i]) for i in range(5)}
    # The first three handoffs already reside within the original sigmoid.
    original = adapter.lower({edge for edge in selected if edge[0] < 3})
    plan, result = lower_connected_dispatches(dispatch, adapter, require_profitable=False)
    if len(plan.handoffs) != 5 or result.realized != frozenset(selected):
        raise ValueError("sigmoid transport lost a required private physical handoff")
    high_water = max(adapter.handoff_facts(*edge)["resident_bytes"] for edge in selected)
    return result, result.removed_transfer_bytes - original.removed_transfer_bytes, high_water


def emit_connected_sigmoid_multiply(module, program, profile, *, symbol, connect=True,
                                    boundary_policy):
    """Emit an experimental complete serial kernel; profitability is measured separately.

    Caller-owned input/output spans must be disjoint. Large or nonfinite inputs
    execute all original calls, preserving their numeric and refusal behavior.
    The finite fast path preserves each original BF16 step, including sigmoid's
    internal rounding; its first multiply commutes only within this finite guard.
    """
    if boundary_policy != "bf16_rne_each_vpu_call" or not symbol.isidentifier() or not symbol.isascii():
        raise ValueError("sigmoid transport needs an explicit BF16 policy and valid entry symbol")
    shape, gate_first = check_sigmoid_multiply(module, program)
    elements, chunk = prod(shape), min(prod(shape), 8192)
    baseline = emit_vpu_unary(profile, shape=shape, dtype="f32", operation="sigmoid",
                              symbol=symbol + "_sigmoid").source
    baseline += "\n".join(emit_vpu_elementwise(
        profile, result_shape=shape, rhs_shape=shape, dtype="f32", operation="mul",
        symbol=symbol + "_mul" + str(i)).source for i in (0, 1))
    first = "x,s" if gate_first else "s,x"
    baseline += f'''\nstatic float {symbol}_s[{elements}] __attribute__((aligned(64)));
static float {symbol}_m[{elements}] __attribute__((aligned(64)));
static void {symbol}_baseline(const float *x,const float *u,float *o) {{
  float *s={symbol}_s,*m={symbol}_m;
  {symbol}_sigmoid(x,s); {symbol}_mul0({first},m); {symbol}_mul1(m,u,o);
}}\n'''
    if not connect:
        return ConnectedSigmoidMultiply(baseline + f'''
void {symbol}(const float *x,const float *u,float *o) {{ {symbol}_baseline(x,u,o); }}\n''', shape, 0, 0)
    source_digest = hashlib.sha256(str(module).encode()).hexdigest()
    issuers, dispatch, removed, resident = [], [], 0, 0
    for count in dict.fromkeys((chunk, elements % chunk)):
        if not count:
            continue
        code, saved, high_water = _resident(profile, count, source_digest)
        issuers.append(emit_c(list(code.commands), transport="rocket_rocc", buffers=("x", "u", "output"),
                              symbol=symbol + "_issue_" + str(count), internal=True))
        dispatch.append(f"case {count}: {symbol}_issue_{count}(xx,uu,rr); break;")
        removed += saved * (elements // chunk if count == chunk else 1)
        resident = max(resident, high_water)
    maximum = struct.unpack("<f", struct.pack("<I", 0x7f7f0000))[0]
    limit = 2.0 ** floor(log2(sqrt(maximum) / 4))
    limit_bits = struct.unpack("<I", struct.pack("<f", limit))[0]
    size = (chunk + 7) // 8 * 8
    source = baseline + "\n".join(issuers) + f'''
static uint16_t {symbol}_x[{size}] __attribute__((aligned(64)));
static uint16_t {symbol}_u[{size}] __attribute__((aligned(64)));
static uint16_t {symbol}_r[{size}] __attribute__((aligned(64)));
static uint16_t {symbol}_pack(float x,unsigned *safe,uint32_t limit) {{
  union {{float f;uint32_t u;}} w; w.f=x;
  *safe &= (w.u & 0x7fffffffu)<=limit;
  w.u+=0x7fffu+((w.u>>16)&1u); return w.u>>16;
}}
void {symbol}(const float *x,const float *u,float *o) {{
  if(!x || !u || !o) __builtin_trap();
  uint16_t *xx={symbol}_x,*uu={symbol}_u,*rr={symbol}_r;
  for(uint32_t base=0;base<{elements}u;base+={chunk}u) {{
    uint32_t count={elements}u-base; if(count>{chunk}u) count={chunk}u;
    unsigned safe=1;
    for(uint32_t i=0;i<(count+7u)/8u*8u;++i) {{
      xx[i]=i<count ? {symbol}_pack(x[base+i],&safe,0x41800000u) : 0;
      uu[i]=i<count ? {symbol}_pack(u[base+i],&safe,{limit_bits}u) : 0;
    }}
    if(!safe) {{ {symbol}_baseline(x,u,o); return; }}
    switch(count) {{ {''.join(dispatch)} default: __builtin_trap(); }}
    for(uint32_t i=0;i<count;++i) {{
      union {{float f;uint32_t u;}} w; w.u=(uint32_t)rr[i]<<16; o[base+i]=w.f;
    }}
  }}
}}\n'''
    return ConnectedSigmoidMultiply(source, shape, removed, resident)


def emit_sigmoid_with_connected_multiplies(module, program, profile, *, symbol, boundary_policy):
    """Current multiply-only optimization, including the same original sigmoid."""
    from .merlin_connected_multiply import chain_adapter, emit_chain_kernel

    if boundary_policy != "bf16_rne_each_vpu_call" or not symbol.isidentifier() or not symbol.isascii():
        raise ValueError("sigmoid transport needs an explicit BF16 policy and valid entry symbol")
    shape, gate_first = check_sigmoid_multiply(module, program)
    functions = {op.sym_name.data: op for op in module.walk() if op.name == "func.func"}
    ops = [next(op for op in functions[node["op"]].body.block.ops if op.name == "linalg.generic")
           for node in program["nodes"][1:]]
    pair = {"first": ops[0], "second": ops[1], "shape": list(shape),
            "kernel_symbol": symbol + "_multiplies"}
    multiply_source, removed = emit_chain_kernel(pair, profile)
    _, adapter = chain_adapter(pair, profile)
    resident = adapter.handoff_facts(0, 1, "m")["resident_bytes"]
    source = emit_vpu_unary(profile, shape=shape, dtype="f32", operation="sigmoid",
                            symbol=symbol + "_sigmoid").source + multiply_source
    source += f'''\nstatic float {symbol}_s[{prod(shape)}] __attribute__((aligned(64)));
void {symbol}(const float *x,const float *u,float *o) {{
  {symbol}_sigmoid(x,{symbol}_s);
  {symbol}_multiplies({"x," + symbol + "_s" if gate_first else symbol + "_s,x"},u,o);
}}\n'''
    return ConnectedSigmoidMultiply(source, shape, removed, resident)
