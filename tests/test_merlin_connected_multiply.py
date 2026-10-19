"""Whole-stage rewrites retain source policy, ownership, and arithmetic."""

import json
import shutil
import subprocess
from math import prod
from pathlib import Path

import pytest
from merlin.frontends.linalg_mlir import parse_mlir_file, parse_mlir_text

from mx_gemmini_support.merlin_connected_multiply import (
    emit_chain_kernel,
    emit_chain_shim,
    multiply_pairs,
)
from mx_gemmini_support.merlin_vpu_elementwise import (
    MXVPUElementwisePostOffload,
    selected_elementwise_sites,
)
from mx_gemmini_support.target_profile import load_profile
from tools.capture_broadcast_multiply_probe import broadcast_source

ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")
SOURCE = (ROOT / "docs/evidence/connected_model_section_20261016/tiny_mlp/source.mlir").read_text()


def pairs(source=SOURCE, enabled=None, *, include_broadcasts=True):
    module = parse_mlir_text(source)
    sites = selected_elementwise_sites(source, PROFILE)
    if enabled is not None:
        sites = [site for site in sites if site["region"] in enabled]
    return multiply_pairs(module, sites, include_broadcasts=include_broadcasts)


@pytest.mark.parametrize("shape", ["1x1x137", "1x3x2731", "1x64x5632"])
def test_source_pair_uses_shared_legality_and_existing_bf16_adapter(shape):
    pair, = pairs(SOURCE.replace("1x64x5632", shape))
    code, removed = emit_chain_kernel(pair, PROFILE)
    assert removed > 0
    assert "if(!safe)" in code
    assert pair["kernel_symbol"] in code
    assert "run_chain(" not in code
    assert "_mlir_ciface_" + pair["symbol"] in emit_chain_shim(pair)


def test_placing_either_multiply_on_host_prevents_connection():
    assert not pairs(enabled={"mul_0#mul0"})
    assert not pairs(enabled={"mul_1#mul0"})


@pytest.mark.parametrize("change", ["arithmetic", "operand", "fastmath"])
def test_names_cannot_override_actual_arithmetic(change):
    if change == "arithmetic":
        source = SOURCE.replace('"arith.mulf"', '"arith.addf"', 1)
    elif change == "operand":
        source = SOURCE.replace('"arith.mulf"(%5, %6)', '"arith.mulf"(%5, %7)', 1)
    else:
        source = SOURCE.replace("#arith.fastmath<none>", "#arith.fastmath<fast>", 1)
    assert not pairs(source)


def test_full_rewriter_emits_one_call_and_accounts_for_both_sites(tmp_path):
    source, sidecar = tmp_path / "source.mlir", tmp_path / "sidecar.json"
    source.write_text(SOURCE)
    sidecar.write_text("{}")
    result = MXVPUElementwisePostOffload(PROFILE, connect_multiplies=True)(source, sidecar, tmp_path / "out")
    module = parse_mlir_file(result)
    module.verify()
    calls = [op for op in module.walk() if op.name == "func.call"]
    assert len(calls) == 1 and len(calls[0].arguments) == 4
    record = json.loads((result.parent / "mx_vpu_elementwise_rewrite.json").read_text())
    assert len(record["regions"]) == 2
    assert record["connected_multiplies"][0]["regions"] == ["mul_0#mul0", "mul_1#mul0"]
    assert not [op for op in module.walk() if op.name == "arith.mulf"]


def test_connection_is_opt_in(tmp_path):
    source, sidecar = tmp_path / "source.mlir", tmp_path / "sidecar.json"
    source.write_text(SOURCE)
    sidecar.write_text("{}")
    result = MXVPUElementwisePostOffload(PROFILE)(source, sidecar, tmp_path / "out")
    assert len([op for op in parse_mlir_file(result).walk() if op.name == "func.call"]) == 2


@pytest.mark.parametrize("middle_index,first_swapped", [(0, False), (1, False), (0, True), (1, True)])
def test_broadcast_pairs_keep_source_order_and_pack_actual_input_coordinates(middle_index, first_swapped):
    pair, = pairs(broadcast_source(middle_index=middle_index, first_swapped=first_swapped))
    from mx_gemmini_support.merlin_connected_multiply import chain_adapter

    graph, adapter = chain_adapter(pair, PROFILE)
    assert graph.nodes[1].inputs == (["m", "c"] if middle_index == 0 else ["c", "m"])
    code, removed = emit_chain_kernel(pair, PROFILE)
    assert removed == ((3 * 137 + 7) // 8) * 32
    assert " / 137u % 3u" in code and " % 137u" in code
    assert "if(!safe)" in code
    expected = "chain_mul_0(b,a,middle);" if first_swapped else "chain_mul_0(a,b,middle);"
    assert expected in adapter.lower(set()).source
    assert "chain_mul_1(middle,c,output);" in adapter.lower(set()).source
    assert "merlin_lmemref_1d" in emit_chain_shim(pair)
    assert "merlin_disjoint" in emit_chain_shim(pair)


def test_mixed_rank_rewrite_and_compiled_descriptor_abi(tmp_path):
    if not shutil.which("cc"):
        pytest.skip("host C compiler unavailable")
    source = tmp_path / "source.mlir"
    sidecar = tmp_path / "sidecar.json"
    source.write_text(broadcast_source())
    sidecar.write_text("{}")
    pair, = pairs(source.read_text())
    result = MXVPUElementwisePostOffload(PROFILE, connect_multiplies=True,
                                        connect_broadcast_multiplies=True)(source, sidecar, tmp_path / "out")
    module = parse_mlir_file(result)
    module.verify()
    call, = [op for op in module.walk() if op.name == "func.call"]
    assert [tuple(v.type.get_shape()) for v in call.arguments] == [(1, 3, 137), (1, 3, 1), (137,), (1, 3, 137)]
    shapes = (*pair["input_shapes"], pair["shape"])
    descriptors = []
    for name, shape in zip("abcd", shapes, strict=True):
        strides = [prod(shape[i + 1:]) for i in range(len(shape))]
        descriptors.append(f'''float {name}[{prod(shape)}]={{0}};
            merlin_lmemref_{len(shape)}d {name}_desc={{ {name},{name},0,
              {{{','.join(map(str, shape))}}}, {{{','.join(map(str, strides))}}} }};''')
    path, binary = tmp_path / "abi.c", tmp_path / "abi"
    path.write_text(emit_chain_shim(pair) + f'''
      static int called=0;
      void {pair['kernel_symbol']}(void *a,void *b,void *c,void *o) {{
        const float *x=a,*scale=b,*weight=c; float *y=o; ++called;
        for(unsigned i=0;i<411;++i)y[i]=x[i]+scale[i/137]+weight[i%137];
      }}
      int main(void) {{
        {''.join(descriptors)}
        for(unsigned i=0;i<411;++i)a[i]=(float)i;
        for(unsigned i=0;i<3;++i)b[i]=(float)(i*10);
        for(unsigned i=0;i<137;++i)c[i]=(float)(i*100);
        merlin_lmemref_3d result;
        _mlir_ciface_{pair['symbol']}(&result,&a_desc,&b_desc,&c_desc,&d_desc);
        if(called!=1 || result.aligned!=d || result.offset)return 1;
        for(unsigned i=0;i<411;++i)if(d[i]!=a[i]+b[i/137]+c[i%137])return 2;
        c_desc.sizes[0]=136;
        _mlir_ciface_{pair['symbol']}(&result,&a_desc,&b_desc,&c_desc,&d_desc);
        if(called!=1 || result.aligned)return 3;
        return 0;
      }}''')
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", str(path), "-o", str(binary)],
                   check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("change", ["fanout", "init", "fastmath", "arithmetic", "operand"])
def test_broadcast_fusion_refuses_observable_or_changed_arithmetic(change):
    source = broadcast_source()
    if change == "fanout":
        source = source.replace("func.return %o", "%extra = tensor.extract %m[%zero, %zero, %zero] : tensor<1x3x137xf32>\nfunc.return %o")
        source = source.replace("%e1 =", "%zero = arith.constant 0 : index\n%e1 =")
    elif change == "init":
        source = source.replace('"arith.mulf"(%x, %y)', '"arith.mulf"(%x, %init)', 1)
    elif change == "fastmath":
        source = source.replace("#arith.fastmath<none>", "#arith.fastmath<fast>", 1)
    elif change == "arithmetic":
        source = source.replace('"arith.mulf"', '"arith.addf"', 1)
    else:
        source = source.replace('"arith.mulf"(%x, %y)', '"arith.mulf"(%x, %x)', 1)
    assert not pairs(source)


def test_broadcast_selection_is_separate_and_requires_the_original_connection_option(tmp_path):
    assert not pairs(broadcast_source(), include_broadcasts=False)
    assert len(pairs(SOURCE, include_broadcasts=False)) == 1
    with pytest.raises(ValueError, match="require connected multiplies"):
        MXVPUElementwisePostOffload(PROFILE, connect_broadcast_multiplies=True)
    source, sidecar = tmp_path / "source.mlir", tmp_path / "sidecar.json"
    source.write_text(broadcast_source())
    sidecar.write_text("{}")
    result = MXVPUElementwisePostOffload(PROFILE, connect_multiplies=True)(source, sidecar, tmp_path / "out")
    assert len([op for op in parse_mlir_file(result).walk() if op.name == "func.call"]) == 2
