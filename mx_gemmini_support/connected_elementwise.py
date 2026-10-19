"""Connected f32 multiply regions under the existing BF16 VPU boundary policy."""

from __future__ import annotations

import hashlib
import struct
from copy import deepcopy
from dataclasses import dataclass
from math import floor, log2, prod, sqrt

from .command_ir import emit_c
from .connected_vpu import VpuDispatchAdapter
from .target_profile import profile_sha256
from .vpu_add_kernel import emit_vpu_elementwise


@dataclass(frozen=True)
class ConnectedElementwise:
    source: str
    realized: frozenset[tuple[int, int, str]]
    removed_transfer_bytes: int


def multiply_projection(op):
    """Reuse the core's closed scalar proof and bind actual broadcast maps."""
    from merlin.frontends.linalg_patterns import recognize_static_projected_pointwise
    from merlin.llvmlower.tensor_broadcast import singleton_projection_axes

    try:
        pattern = recognize_static_projected_pointwise(op)
        if (pattern.operation != "arith.mulf" or pattern.ordered_types != ("f32",) * 4
                or not 1 <= len(pattern.shape) <= 4):
            raise ValueError("unsupported multiply type or rank")
        axes = tuple(singleton_projection_axes(mapping.data, pattern.shape, shape)
                     for mapping, shape in zip(tuple(op.indexing_maps)[:2], pattern.input_shapes, strict=True))
    except ValueError as exc:
        raise ValueError("multiply chain scalar body or strict floating indexing is unsupported") from exc
    return pattern, axes


def check_multiply_regions(module, program: dict) -> tuple[int, ...]:
    """Require two strict projected multiplies in their actual typed functions."""

    module.verify()
    if len(program["nodes"]) != 2:
        raise ValueError("multiply chain requires two dispatches")
    functions = {op.sym_name.data: op for op in module.walk() if op.name == "func.func"}
    shapes = []
    for node in program["nodes"]:
        if node["kind"] != "dispatch" or len(node["inputs"]) != 2 or len(node["outputs"]) != 1:
            raise ValueError("multiply chain requires binary single-result dispatches")
        metadata = [program["buffers"][name] for name in (*node["inputs"], *node["outputs"])]
        shape = tuple(metadata[-1]["shape"])
        if (not shape or any(type(d) is not int or d <= 0 for d in shape)
                or any(value["dtype"] != "f32" for value in metadata)):
            raise ValueError("multiply chain requires equal static f32 tensors")
        function = functions[node["op"]]
        tensors = tuple("tensor<" + "".join(f"{d}x" for d in value["shape"]) + "f32>" for value in metadata)
        if (tuple(str(t) for t in function.function_type.inputs) != tensors[:2]
                or tuple(str(t) for t in function.function_type.outputs) != tensors[2:]):
            raise ValueError("multiply chain signature differs from dispatch graph")
        block = function.body.block
        generics = [op for op in block.ops if op.name == "linalg.generic"]
        if (len(generics) != 1 or any(op.name not in {"tensor.empty", "linalg.generic", "func.return"}
                                     for op in block.ops)):
            raise ValueError("multiply chain requires one pure generic per dispatch")
        op = generics[0]
        pattern, _ = multiply_projection(op)
        if (list(op.inputs) != list(block.args) or len(op.outputs) != 1
                or pattern.shape != shape
                or pattern.input_shapes != tuple(tuple(value["shape"]) for value in metadata[:2])
                or list(block.last_op.operands) != list(op.results)):
            raise ValueError("multiply chain scalar body or indexing is unsupported")
        shapes.append(shape)
    if shapes[0] != shapes[1]:
        raise ValueError("multiply chain shapes differ")
    return shapes[0]


class F32MultiplyAdapter:
    """Retain the BF16 intermediate between two already selected VPU calls.

    The source f32 multiply policy still rounds through BF16 at each stage.
    A runtime range guard falls back to the original kernels for nonfinite or
    extreme inputs, preserving their handling of NaNs and intermediate overflow.
    """

    def __init__(self, module, program: dict, profile: dict, *, boundary_policy: str,
                 measured_savings: dict | None = None):
        if boundary_policy != "bf16_rne_each_vpu_call":
            raise ValueError("multiply chain requires the explicitly selected BF16 boundary policy")
        self.shape = check_multiply_regions(module, program)
        self.program = deepcopy(program)
        functions = {op.sym_name.data: op for op in module.walk() if op.name == "func.func"}
        self.node_axes = tuple(multiply_projection(next(op for op in functions[node["op"]].body.block.ops
                                                        if op.name == "linalg.generic"))[1]
                               for node in program["nodes"])
        middle = program["nodes"][0]["outputs"][0]
        consumer_inputs = program["nodes"][1]["inputs"]
        if middle not in consumer_inputs:
            raise ValueError("multiply chain buffer map differs from its private handoff")
        self.middle_index = consumer_inputs.index(middle)
        if (tuple(program["buffers"][middle]["shape"]) != self.shape
                or self.node_axes[1][self.middle_index] != tuple(range(len(self.shape)))):
            raise ValueError("multiply chain intermediate must have an identity consumer map")
        self.profile = deepcopy(profile)
        self.measured_savings = dict(measured_savings or {})
        self.elements = prod(self.shape)
        self.chunk = min(self.elements, 8192)
        self.sizes = tuple(dict.fromkeys((self.chunk, self.elements % self.chunk)))
        self.sizes = tuple(count for count in self.sizes if count)
        self.adapters = {count: self._wire_adapter(count) for count in self.sizes}

    def _wire_adapter(self, count):
        graph = deepcopy(self.program)
        if self.middle_index == 1:
            # Finite guarded multiplication is commutative. Put the private
            # intermediate on the physical SRC1 port supported by residency.
            # The source graph and exceptional-value fallback keep their order.
            graph["nodes"][1]["inputs"].reverse()
        rows = (count + 7) // 8
        for value in graph["buffers"].values():
            value.update(shape=[rows, 8], dtype="bf16")
        digest = profile_sha256(self.profile)
        source_sha = hashlib.sha256(str(self.program).encode()).hexdigest()
        fragments = []
        for index, node in enumerate(graph["nodes"]):
            src, dst = (0, 8192) if index == 0 else (8192, 0)
            text = f'''builtin.module attributes {{mx.profile_sha256 = "{digest}",
              mx.contract_sha256 = "{source_sha}", mx.policy_sha256 = "{source_sha}",
              prov.quantization_manifest_sha256 = "{source_sha}"}} {{
              func.func @multiply() {{
                "mx_gemmini.vpu_execute"() {{site_id = "{node['op']}", kind = "mul",
                  src1_row = {src} : i32, src2_row = 4096 : i32, dst_row = {dst} : i32,
                  rows = {rows} : i32, reduction_length = 1 : i32, broadcast = false,
                  immediate_bf16 = 0 : i32, profile_sha256 = "{digest}",
                  contract_sha256 = "{source_sha}", policy_sha256 = "{source_sha}",
                  manifest_sha256 = "{source_sha}"}} : () -> ()
                func.return
              }}
            }}'''
            fragments.append(dict(mlir=text, buffers=dict(
                src1=node["inputs"][0], src2=node["inputs"][1], output=node["outputs"][0])))
        return VpuDispatchAdapter(graph, self.profile, fragments)

    def handoff_facts(self, producer, consumer, buffer):
        facts = self.adapters[self.chunk].handoff_facts(producer, consumer, buffer)
        if facts is not None:
            facts.update(interface="mx_guarded_f32_bf16_multiply_chain",
                         saved_cycles=self.measured_savings.get((producer, consumer, buffer)))
        return facts

    def lower(self, selected):
        for edge in selected:
            if self.handoff_facts(*edge) is None:
                raise ValueError("f32 multiply chain has no legal selected handoff")
        nodes = self.program["nodes"]
        pointers = [*nodes[0]["inputs"], nodes[1]["inputs"][1 - self.middle_index],
                    nodes[0]["outputs"][0], nodes[1]["outputs"][0]]
        names = dict(zip(pointers, ("a", "b", "c", "middle", "output"), strict=True))
        baseline, calls = [], []
        for index, node in enumerate(nodes):
            inputs, axes = list(node["inputs"]), list(self.node_axes[index])
            shapes = [tuple(self.program["buffers"][name]["shape"]) for name in inputs]
            if shapes[0] != self.shape and shapes[1] == self.shape:
                # Match the existing standalone VPU provider's full-input-first
                # convention, including NaN refusal and infinity handling.
                inputs.reverse()
                shapes.reverse()
                axes.reverse()
            baseline.append(emit_vpu_elementwise(
                self.profile, result_shape=self.shape, lhs_shape=shapes[0], rhs_shape=shapes[1],
                lhs_axes=axes[0], rhs_axes=axes[1], dtype="f32", operation="mul",
                symbol=f"chain_mul_{index}").source)
            calls.append(f"chain_mul_{index}({','.join(names[name] for name in inputs)},"
                         f"{names[node['outputs'][0]]});")
        baseline = "\n".join(baseline) + '''\nvoid materialized_chain(const float *a, const float *b, const float *c,
            float *middle, float *output) {
          ''' + " ".join(calls) + '''
        }\n'''
        if not selected:
            return ConnectedElementwise(baseline + '''
              void run_chain(const float *a,const float *b,const float *c,float *m,float *o) {
                materialized_chain(a,b,c,m,o);
              }\n''', frozenset(), 0)
        if len(set(pointers)) != 5:
            raise ValueError("connected multiply chain requires independent external operands")
        issuers, dispatch = [], []
        for count, adapter in self.adapters.items():
            code = adapter.lower(selected)
            issuers.append(emit_c(list(code.commands), transport="rocket_rocc", buffers=tuple(pointers),
                                  symbol=f"chain_issue_{count}", internal=True))
            dispatch.append(f"case {count}: chain_issue_{count}(aa,bb,cc,rr,rr); break;")
        # Inputs below this derived BF16 bound cannot overflow the first
        # multiply. The final multiply may produce infinity, as the baseline can.
        maximum = struct.unpack("<f", struct.pack("<I", 0x7f7f0000))[0]
        limit = 2.0 ** floor(log2(sqrt(maximum) / 4))
        limit_bits = struct.unpack("<I", struct.pack("<f", limit))[0]
        size = (self.chunk + 7) // 8 * 8
        from merlin.llvmlower.tensor_broadcast import dense_broadcast_index

        external_axes = (*self.node_axes[0], self.node_axes[1][1 - self.middle_index])
        indices = [dense_broadcast_index(self.shape, tuple(self.program["buffers"][name]["shape"]),
                                         axes, index="i")
                   for name, axes in zip(pointers[:3], external_axes, strict=True)]
        indices = ["base+i" if expr == "(base + i)" else expr for expr in indices]
        source = baseline + "\n".join(issuers) + f'''
        static uint16_t aa[{size}] __attribute__((aligned(64)));
        static uint16_t bb[{size}] __attribute__((aligned(64)));
        static uint16_t cc[{size}] __attribute__((aligned(64)));
        static uint16_t rr[{size}] __attribute__((aligned(64)));
        static uint16_t chain_pack(float x, unsigned *safe) {{
          union {{float f; uint32_t u;}} word; word.f=x;
          *safe &= (word.u & 0x7fffffffu) <= {limit_bits}u;
          word.u+=0x7fffu+((word.u>>16)&1u); return word.u>>16;
        }}
        void run_chain(const float *a,const float *b,const float *c,float *middle,float *output) {{
          if(!a || !b || !c || !middle || !output) __builtin_trap();
          for(uint32_t base=0;base<{self.elements};base+={self.chunk}) {{
            uint32_t count={self.elements}-base;
            if(count>{self.chunk}) count={self.chunk};
            unsigned safe=1;
            for(uint32_t i=0;i<(count+7u)/8u*8u;++i) {{
              aa[i]=i<count ? chain_pack(a[{indices[0]}],&safe) : 0;
              bb[i]=i<count ? chain_pack(b[{indices[1]}],&safe) : 0;
              cc[i]=i<count ? chain_pack(c[{indices[2]}],&safe) : 0;
            }}
            if(!safe) {{ materialized_chain(a,b,c,middle,output); return; }}
            switch(count) {{ {''.join(dispatch)} default: __builtin_trap(); }}
            for(uint32_t i=0;i<count;++i) {{
              union {{float f; uint32_t u;}} word; word.u=(uint32_t)rr[i]<<16;
              output[base+i]=word.f;
            }}
          }}
        }}\n'''
        full, tail = divmod(self.elements, self.chunk)
        removed = full * self.adapters[self.chunk].lower(selected).removed_transfer_bytes
        if tail:
            removed += self.adapters[tail].lower(selected).removed_transfer_bytes
        return ConnectedElementwise(source, frozenset(selected), removed)
