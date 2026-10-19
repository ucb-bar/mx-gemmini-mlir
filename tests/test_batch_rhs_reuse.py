"""Private batch RHS preparation matches every original packed byte and output."""

import ctypes
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mx_gemmini_support.dense_kernel import emit_dense_f32_batch_kernel
from mx_gemmini_support.planned_runtime_issuer import emit_planned_runtime_issuer
from mx_gemmini_support.runtime_group_preparation import emit_runtime_group_preparation
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


@pytest.mark.parametrize("precision", ["FP4", "FP8"])
@pytest.mark.parametrize("m,n,k,repeat", [(1, 17, 37, 3), (19, 23, 64, 2)])
def test_repeated_packing_refreshes_groups_and_invocations(tmp_path_factory, precision, m, n, k, repeat):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C compiler unavailable")
    work = tmp_path_factory.mktemp("batch_rhs_" + precision)
    profile = load_profile(PROFILE)
    issuer = emit_planned_runtime_issuer(profile, (32, 32, 64), (32, 32, 64), precision=precision)
    batch = 3 * repeat
    kernels = [emit_dense_f32_batch_kernel(issuer, (batch, m, n, k), symbol=name, rhs_batch_repeat=r)
               for name, r in (("baseline", 1), ("reused", repeat))]
    assert kernels[0].workspace_bytes == kernels[1].workspace_bytes
    for name, kernel in zip(("baseline", "reused"), kernels, strict=True):
        (work / (name + ".c")).write_text(kernel.source)
    sizes = issuer.buffer_bytes
    snapshots = {"a": sizes["activation"], "asc": sizes["activation_scales"],
                 "b": sizes["weight"], "bsc": sizes["weight_scales"]}
    fields = "\n".join(f"unsigned char observed_{name}[{batch * size}];" for name, size in snapshots.items())
    copies = "\n".join(f"for (unsigned j=0;j<{size};++j) {{ observed_{name}[calls*{size}+j]=((const unsigned char*){name})[j]; h=(h^((const unsigned char*){name})[j])*16777619u; }}"
                        for name, size in snapshots.items())
    (work / "stub.c").write_text(f'''#include <stdint.h>
unsigned calls, weight_calls;
{fields}
void reset(void) {{ calls=weight_calls=0; }}
extern int __real_mx_pack_weight_f32(const float*,uint32_t,uint32_t,uint32_t,uint8_t*,uint8_t*);
int __wrap_mx_pack_weight_f32(const float*p,uint32_t r,uint32_t c,uint32_t f,uint8_t*b,uint8_t*s) {{
  ++weight_calls;return __real_mx_pack_weight_f32(p,r,c,f,b,s);
}}
void mx_issue_encoded(const void*a,const void*asc,const void*b,const void*bsc,const void*out,const void*scratch) {{
  (void)scratch;uint32_t h=2166136261u;
  {copies}
  for(unsigned j=0;j<32u*32u;++j)((uint16_t*)out)[j]=(uint16_t)(0x3f00u+((h+j)%127u));
  ++calls;
}}
''')
    (work / "runtime_group_prepare.h").write_text(emit_runtime_group_preparation())
    library = work / "batch.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-DMX_QUANT_GROUP_WORDS", "-DMX_QUANT_INLINE_GROUP",
                    "-I", str(work), "-I", str(ROOT / "mx_gemmini_support"),
                    str(work / "baseline.c"), str(work / "reused.c"), str(work / "stub.c"),
                    str(ROOT / "mx_gemmini_support/runtime_quant.c"),
                    "-Wl,--wrap=mx_pack_weight_f32", "-o", str(library)], check=True)
    loaded = ctypes.CDLL(str(library))
    rng = np.random.default_rng(219)
    lhs = rng.normal(size=(batch, m, k)).astype(np.float32)
    rhs = np.repeat(rng.normal(size=(3, k, n)).astype(np.float32), repeat, axis=0)
    for invocation in range(3):
        # Every call refreshes its first group; an interior group also changes.
        rhs[:repeat] *= np.float32(0.75)
        rhs[repeat:2 * repeat, invocation, invocation] += np.float32(2)
        inputs = lhs.tobytes(), rhs.tobytes()
        observations = []
        for name, weight_calls in (("baseline", batch), ("reused", 3)):
            loaded.reset()
            entry = getattr(loaded, name)
            entry.argtypes = [ctypes.c_void_p] * 3
            storage = np.full(batch * m * n + 32, 0xDEADBEEF, dtype=np.uint32)
            output = storage[16:-16].view(np.float32).reshape(batch, m, n)
            entry(lhs.ctypes.data, rhs.ctypes.data, output.ctypes.data)
            assert ctypes.c_uint.in_dll(loaded, "calls").value == batch
            assert ctypes.c_uint.in_dll(loaded, "weight_calls").value == weight_calls
            assert np.all(storage[:16] == 0xDEADBEEF) and np.all(storage[-16:] == 0xDEADBEEF)
            observed = {field: bytes((ctypes.c_ubyte * (batch * size)).in_dll(loaded, "observed_" + field))
                        for field, size in snapshots.items()}
            observations.append((output.tobytes(), observed))
        assert observations[0] == observations[1]
        assert inputs == (lhs.tobytes(), rhs.tobytes())


@pytest.mark.parametrize("repeat", [0, -1, 5, True, 1.0])
def test_incomplete_batch_groups_refuse(repeat):
    issuer = emit_planned_runtime_issuer(load_profile(PROFILE), (32, 32, 32), (32, 32, 32), precision="FP8")
    with pytest.raises(ValueError, match="complete positive contiguous groups"):
        emit_dense_f32_batch_kernel(issuer, (6, 1, 32, 32), symbol="bad", rhs_batch_repeat=repeat)


def post_fixture(tmp_path):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr

    module = parse_mlir_text('''module {
      func.func @forward(%a:tensor<6x1x4xf32>,%b:tensor<2x1x4x5xf32>)->tensor<6x1x5xf32> {
        %e=tensor.empty():tensor<2x3x4x5xf32>
        %p=linalg.generic {indexing_maps=[affine_map<(a,b,c,d)->(a,0,c,d)>,affine_map<(a,b,c,d)->(a,b,c,d)>],iterator_types=["parallel","parallel","parallel","parallel"]} ins(%b:tensor<2x1x4x5xf32>) outs(%e:tensor<2x3x4x5xf32>) {
          ^bb0(%x:f32,%y:f32):linalg.yield %x:f32
        }->tensor<2x3x4x5xf32>
        %f=tensor.collapse_shape %p [[0,1,2,3]]:tensor<2x3x4x5xf32> into tensor<120xf32>
        %r=tensor.expand_shape %f [[0,1,2]] output_shape [6,4,5]:tensor<120xf32> into tensor<6x4x5xf32>
        %o=tensor.empty():tensor<6x1x5xf32>
        %v=func.call @original(%a,%r,%o):(tensor<6x1x4xf32>,tensor<6x4x5xf32>,tensor<6x1x5xf32>)->tensor<6x1x5xf32>
        func.return %v:tensor<6x1x5xf32>
      }
      func.func private @original(tensor<6x1x4xf32>,tensor<6x4x5xf32>,tensor<6x1x5xf32>)->tensor<6x1x5xf32>
    }''')
    for op in module.body.block.ops:
        if op.sym_name.data == "original":
            op.properties["arg_attrs"] = ArrayAttr([DictionaryAttr({"bufferization.access": StringAttr(mode)})
                                                   for mode in ("read", "read", "write")])
    source = tmp_path / "source.mlir"
    source.write_text(text(module))
    obj = tmp_path / "catalog.o"
    obj.write_bytes(b"observed-catalog-object")
    compiled = tmp_path / "compiled.c"
    compiled.write_text("/* separately qualified provider source */")
    metadata = {"entry_symbol": "retained", "kernel_symbol": "compiled", "repeat": 3,
                "completion": "synchronous_native_issuer", "lifetime": "private_serial_batch_call",
                "provider_obligation": "original_results_rounding_effects_and_partial_refusal", "workspace_bytes": 512,
                "source_proof": {"contiguous_batch_repeat": 3, "shape": [6, 4, 5]}}
    types = ["tensor<6x1x4xf32>", "tensor<6x4x5xf32>", "tensor<6x1x5xf32>"]
    catalog = {"source_sha256": "source", "batch_rhs_reuse": "experimental_exact_contiguous_repetition",
               "integer_group_packing": True,
               "bindings": [{"region": "binding", "symbol": "native", "source_operation_ordinal": 15,
                             "tensor_types": types, "batch_rhs_reuse": metadata}],
               "compilation": {"object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
                               "source_sha256": {"compiled": hashlib.sha256(compiled.read_bytes()).hexdigest()}}}
    route = {"model_sha256": "source", "catalog_manifest": str(tmp_path / "catalog.json"),
             "catalog_object": str(obj), "routed": [{"symbol": "original", "source_region": "binding",
                                                       "source_operation_ordinal": 15, "tensor_types": types}]}
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    (tmp_path / "route.json").write_text(json.dumps(route))
    return source, catalog, route


def test_provider_retargets_only_source_proved_serial_rhs(tmp_path):
    from mx_gemmini_support.merlin_batch_rhs import MXBatchRHSReusePostOffload

    source, _, _ = post_fixture(tmp_path)
    original = source.read_bytes()
    out = MXBatchRHSReusePostOffload(tmp_path / "catalog.json")(source, tmp_path / "route.json", tmp_path / "rewrite")
    assert source.read_bytes() == original and "func.call @retained" in out.read_text()
    record = json.loads((out.parent / "mx_batch_rhs_reuse_rewrite.json").read_text())
    assert record["consumers"][0]["input_proofs"][0]["contiguous_batch_repeat"] == 3


@pytest.mark.parametrize("unproved", [False, True])
def test_shared_declaration_selects_each_current_call_independently(tmp_path, unproved):
    import copy

    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.xdsl_dialects._common import text
    from xdsl.dialects import func, tensor

    from mx_gemmini_support.merlin_batch_rhs import MXBatchRHSReusePostOffload

    source, catalog, route = post_fixture(tmp_path)
    module = parse_mlir_file(source)
    call = next(op for op in module.walk() if isinstance(op, func.CallOp))
    arguments = list(call.arguments)
    if unproved:
        unknown = tensor.EmptyOp([], arguments[1].type)
        call.parent.insert_op_before(unknown, call)
        arguments[1] = unknown.tensor
    other = func.CallOp("original", arguments, [value.type for value in call.results])
    call.parent.insert_op_before(other, call)
    source.write_text(text(module))
    binding = copy.deepcopy(catalog["bindings"][0])
    binding.update(region="second", source_operation_ordinal=29)
    binding["batch_rhs_reuse"]["entry_symbol"] = "second_retained"
    if unproved:
        binding.pop("batch_rhs_reuse")
    catalog["bindings"].append(binding)
    row = copy.deepcopy(route["routed"][0])
    row.update(source_region="second", source_operation_ordinal=29)
    route["routed"].append(row)
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    (tmp_path / "route.json").write_text(json.dumps(route))
    output = MXBatchRHSReusePostOffload(tmp_path / "catalog.json")(source, tmp_path / "route.json", tmp_path / "rewrite")
    rewritten = parse_mlir_file(output)
    callees = [op.callee.root_reference.data for op in rewritten.walk() if isinstance(op, func.CallOp)]
    assert callees == (["original", "retained"] if unproved else ["retained", "retained"])
    report = json.loads((output.parent / "mx_batch_rhs_reuse_rewrite.json").read_text())
    assert report["selected_calls"] == (1 if unproved else 2)


@pytest.mark.parametrize("change", ["family", "coverage"])
def test_shared_declaration_binding_changes_refuse_before_output(tmp_path, change):
    import copy

    from mx_gemmini_support.merlin_batch_rhs import MXBatchRHSReusePostOffload

    source, catalog, route = post_fixture(tmp_path)
    binding = copy.deepcopy(catalog["bindings"][0])
    binding.update(region="second", source_operation_ordinal=29)
    if change == "family":
        binding["symbol"] = "different_native_family"
    catalog["bindings"].append(binding)
    row = copy.deepcopy(route["routed"][0])
    row.update(source_region="second", source_operation_ordinal=29)
    route["routed"].append(row)
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    (tmp_path / "route.json").write_text(json.dumps(route))
    with pytest.raises(ValueError, match="incompatible native kernel families|call coverage changed"):
        MXBatchRHSReusePostOffload(tmp_path / "catalog.json")(source, tmp_path / "route.json", tmp_path / "rewrite")
    assert not (tmp_path / "rewrite").exists()


@pytest.mark.parametrize("change", ["object", "kernel", "repeat", "completion", "lifetime", "ordinal", "source", "type"])
def test_changed_provider_bindings_refuse(tmp_path, change):
    from mx_gemmini_support.merlin_batch_rhs import MXBatchRHSReusePostOffload

    source, catalog, route = post_fixture(tmp_path)
    alternative = catalog["bindings"][0]["batch_rhs_reuse"]
    if change in {"object", "kernel"}:
        (tmp_path / ("catalog.o" if change == "object" else "compiled.c")).write_bytes(b"changed")
    elif change in {"repeat", "completion", "lifetime"}:
        alternative[change] = 2 if change == "repeat" else "unknown"
    elif change == "ordinal":
        route["routed"][0]["source_operation_ordinal"] += 1
    elif change == "source":
        catalog["source_sha256"] = "changed"
    else:
        route["routed"][0]["tensor_types"] = ["tensor<99xf32>"] * 3
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    (tmp_path / "route.json").write_text(json.dumps(route))
    with pytest.raises(ValueError):
        MXBatchRHSReusePostOffload(tmp_path / "catalog.json")(source, tmp_path / "route.json", tmp_path / "rewrite")
    assert not (tmp_path / "rewrite").exists()


def test_unselected_integer_packing_cannot_admit_batch_reuse(tmp_path):
    from mx_gemmini_support.merlin_catalog_builder import build_merlin_catalog

    with pytest.raises(ValueError, match="explicit integer group packing"):
        build_merlin_catalog(tmp_path / "missing", tmp_path / "out", profile={}, riscv_root=tmp_path,
                              reuse_batch_rhs=True)
