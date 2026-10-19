"""Integrate qualified BF16 multiply handoffs with whole-stage tensor lowering."""

from __future__ import annotations

from collections import defaultdict
from math import prod

from .connected_elementwise import F32MultiplyAdapter, multiply_projection


def multiply_pairs(module, sites, *, include_broadcasts: bool = False):
    """Bind supported pairs to actual SSA arithmetic, never provenance alone."""
    from merlin.xdsl_dialects.lowering.tensor_chains import (
        destination_only_tensor_call,
        private_tensor_pairs,
        pure_tensor_compute,
    )
    selected = {site["region"]: site for site in sites}
    operations, projections, ordinals = {}, {}, defaultdict(int)
    for function in module.body.block.ops:
        if function.name != "func.func" or not function.body.blocks or function.sym_name.data != "forward":
            continue
        for op in function.body.block.ops:
            if op.name != "linalg.generic":
                continue
            steps = list(op.body.block.ops)
            arithmetic = [step for step in steps if step.name != "linalg.yield"]
            if not arithmetic or arithmetic[-1].name != "arith.mulf" or not (
                    len(arithmetic) == 1 or
                    (len(arithmetic) == 2 and arithmetic[0].name == "arith.extf")):
                continue
            region = getattr(op.attributes.get("prov.region_id"), "data", None)
            key = f"{region}#mul{ordinals[region]}"
            ordinals[region] += 1
            if len(arithmetic) != 1:
                continue
            site = selected.get(key)
            if site is None or site["dtype"] != "f32" or site["operation"] != "mul":
                continue
            try:
                pattern, axes = multiply_projection(op)
            except ValueError:
                continue
            order = (1, 0) if site["swap"] else (0, 1)
            if (pattern.shape != tuple(site["shape"])
                    or tuple(pattern.input_shapes[i] for i in order) !=
                    (tuple(site["lhs_shape"]), tuple(site["rhs_shape"]))
                    or tuple(axes[i] for i in order) !=
                    (tuple(site["lhs_axes"]), tuple(site["rhs_axes"]))):
                continue
            operations[op] = site
            projections[op] = pattern, axes
    result = []
    for first, second in private_tensor_pairs(
            operations, can_cross=lambda op: pure_tensor_compute(op) or destination_only_tensor_call(op)):
        a, b = operations[first], operations[second]
        middle_indices = [i for i, value in enumerate(second.inputs) if value == first.results[0]]
        if (tuple(a["shape"]) != tuple(b["shape"])
                or len(middle_indices) != 1):
            continue
        middle_index = middle_indices[0]
        external_index = 1 - middle_index
        if len({*first.inputs, second.inputs[external_index]}) != 3:
            continue
        first_pattern, first_axes = projections[first]
        second_pattern, second_axes = projections[second]
        if second_axes[middle_index] != tuple(range(len(a["shape"]))):
            continue
        input_shapes = (*first_pattern.input_shapes, second_pattern.input_shapes[external_index])
        input_axes = (*first_axes, second_axes[external_index])
        if not include_broadcasts and (middle_index != 0 or any(shape != tuple(a["shape"]) for shape in input_shapes)):
            continue
        symbol = "mx_vpu_mul_chain_f32_" + "_".join(map(str, a["shape"]))
        if middle_index != 0 or any(shape != tuple(a["shape"]) for shape in input_shapes):
            symbol += "_inputs_" + "__".join(
                ("_".join(map(str, shape)) or "scalar") + "_axes_" +
                ("_".join(map(str, axes)) or "none")
                for shape, axes in zip(input_shapes, input_axes, strict=True)) + f"_middle{middle_index}"
        result.append({"first": first, "second": second, "regions": [a["region"], b["region"]],
                       "shape": list(a["shape"]), "input_shapes": [list(shape) for shape in input_shapes],
                       "input_axes": [list(axes) for axes in input_axes], "middle_index": middle_index,
                       "external_index": external_index, "symbol": symbol, "kernel_symbol": symbol + "_kernel"})
    return result


def chain_adapter(pair, profile):
    """Outline the verified source operations into the existing handoff adapter."""
    from merlin.xdsl_dialects.lowering.dispatch_program import (
        Buffer,
        DispatchProgram,
        Node,
    )
    from xdsl.dialects import func, tensor
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    functions = []
    for index, op in enumerate((pair["first"], pair["second"])):
        block = Block(arg_types=[v.type for v in op.inputs])
        empty = tensor.EmptyOp((), op.results[0].type)
        mapping = dict(zip(op.inputs, block.args, strict=True))
        mapping[op.outputs[0]] = empty.results[0]
        clone = op.clone(value_mapper=mapping)
        block.add_ops([empty, clone, func.ReturnOp(*clone.results)])
        functions.append(func.FuncOp(f"multiply{index}",
                                    ([v.type for v in op.inputs], [op.results[0].type]), Region(block)))
    shapes = dict(zip("abc", pair["input_shapes"], strict=True)) | {"m": pair["shape"], "o": pair["shape"]}
    program = DispatchProgram("forward", [0, 1, 2], {
        name: Buffer(name, shapes[name], "f32", "arg" if name in "abc" else "intermediate",
                     "abc".index(name) if name in "abc" else None)
        for name in ("a", "b", "c", "m", "o")}, [
            Node("dispatch", "multiply0", ["a", "b"], ["m"]),
            Node("dispatch", "multiply1", ["m", "c"] if pair["middle_index"] == 0 else ["c", "m"],
                 ["o"])], ["o"])
    return program, F32MultiplyAdapter(ModuleOp(functions), program.to_dict(), profile,
                                       boundary_policy="bf16_rne_each_vpu_call")


def emit_chain_kernel(pair, profile):
    """Emit a legal experimental handoff; do not assert unmeasured profitability."""
    from merlin.xdsl_dialects.lowering.connected_dispatch import (
        lower_connected_dispatches,
    )

    program, adapter = chain_adapter(pair, profile)
    plan, lowered = lower_connected_dispatches(program, adapter, require_profitable=False)
    if len(plan.handoffs) != 1:
        raise ValueError("multiply pair lost its private handoff")
    prefix = pair["kernel_symbol"]
    source = lowered.source
    for old, suffix in (("run_chain", "run"), ("materialized_chain", "materialized"),
                        ("chain_mul_0", "mul0"), ("chain_mul_1", "mul1")):
        source = source.replace(old, prefix + "_" + suffix)
    source += f'''\nvoid {prefix}(const float *a, const float *b, const float *c, float *o) {{
      static float middle[{prod(pair['shape'])}] __attribute__((aligned(64)));
      {prefix}_run(a,b,c,middle,o);
    }}\n'''
    return source, lowered.removed_transfer_bytes


def emit_chain_shim(pair):
    """Four ranked descriptors: three contiguous inputs and one fresh output."""
    shape, symbol, kernel = pair["shape"], pair["symbol"], pair["kernel_symbol"]
    if any(input_shape != shape for input_shape in pair["input_shapes"]):
        from merlin.llvmlower.device_shim import (
            LogicalBuffer,
            emit_logical_translation_unit,
        )

        buffers = tuple(LogicalBuffer(name, "input", f"input_{i}", tuple(input_shape), "f32")
                        for i, (name, input_shape) in enumerate(zip("abc", pair["input_shapes"], strict=True)))
        buffers += (LogicalBuffer("o", "output", "out_0", tuple(shape), "f32"),)
        unit = emit_logical_translation_unit("mx_gemmini", {symbol: (
            buffers, [{"role": b.role, "shape": b.shape, "dtype": b.dtype} for b in buffers])},
            kernel_symbol_for=lambda _: kernel)
        if unit.skipped or unit.symbols != (symbol,):
            raise ValueError("broadcast multiply ABI could not be emitted")
        return unit.text
    rank = len(shape)
    strides = [prod(shape[i + 1:]) for i in range(rank)]
    params = []
    for name in "abcd":
        params.extend([f"void *{name}_alloc", f"void *{name}_aligned", f"intptr_t {name}_off"])
        params.extend(f"intptr_t {name}_s{i}" for i in range(rank))
        params.extend(f"intptr_t {name}_st{i}" for i in range(rank))
    checks = []
    for name in "abcd":
        checks.extend([f"!{name}_aligned", f"{name}_off < 0"])
        checks.extend(f"{name}_s{i} != {dim}" for i, dim in enumerate(shape))
        checks.extend(f"{name}_st{i} != {step}" for i, step in enumerate(strides))
    fields = ["allocated", "aligned", "offset", *(f"sizes[{i}]" for i in range(rank)),
              *(f"strides[{i}]" for i in range(rank))]
    arguments = ",".join(f"{name}->{field}" for name in "abcd" for field in fields)
    stores = "\n".join(f"result->{field}=r.{field};" for field in fields)
    return f'''#include <stdint.h>
    typedef struct {{void *allocated,*aligned; intptr_t offset;
      intptr_t sizes[{rank}],strides[{rank}];}} mx_chain_memref;
    extern void {kernel}(const float *,const float *,const float *,float *);
    mx_chain_memref {symbol}({','.join(params)}) {{
      if({' || '.join(checks)}) __builtin_trap();
      {kernel}((const float *)a_aligned+a_off,(const float *)b_aligned+b_off,
               (const float *)c_aligned+c_off,(float *)d_aligned+d_off);
      return (mx_chain_memref){{d_alloc,d_aligned,d_off,
        {{{','.join(map(str,shape))}}},{{{','.join(map(str,strides))}}}}};
    }}
    void _mlir_ciface_{symbol}(mx_chain_memref *result,const mx_chain_memref *a,
      const mx_chain_memref *b,const mx_chain_memref *c,const mx_chain_memref *d) {{
      if(!result || !a || !b || !c || !d) __builtin_trap();
      mx_chain_memref r={symbol}({arguments}); {stores}
    }}\n'''
