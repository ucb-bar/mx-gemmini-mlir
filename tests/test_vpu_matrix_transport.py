"""Compact reverse boundaries preserve padding bytes and checked private edges."""

import ctypes
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mx_gemmini_support.dense_contraction import plan_dense_shape
from mx_gemmini_support.dense_kernel import emit_activation_preparation
from mx_gemmini_support.merlin_vpu_matrix_transport import (
    MXVPUMatrixTransportPostOffload,
)
from mx_gemmini_support.planned_runtime_issuer import emit_planned_runtime_issuer
from mx_gemmini_support.runtime_group_preparation import emit_runtime_group_preparation
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


@pytest.mark.parametrize("precision", ["FP4", "FP8"])
@pytest.mark.parametrize("logical", [(1, 17, 37), (19, 23, 37), (32, 32, 64)])
def test_compact_preparation_pads_exact_words_with_half_the_scratch(tmp_path_factory, precision, logical):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler required")
    directory = tmp_path_factory.mktemp("compact_prepare_image")
    profile = load_profile(PROFILE)
    plan = plan_dense_shape(profile, logical, precision=precision)
    issuer = emit_planned_runtime_issuer(profile, plan.padded_shape_mnk, plan.tile_mnk, precision=precision)
    original, size, old_scratch = emit_activation_preparation(issuer, logical, symbol="original")
    compact, new_size, new_scratch = emit_activation_preparation(issuer, logical, symbol="compact", input_dtype="bf16")
    assert size == new_size and old_scratch == 2 * new_scratch
    paths = []
    for name, source in [("original", original), ("compact", compact)]:
        path = directory / (name + ".c")
        path.write_text(source)
        paths.append(str(path))
    (directory / "runtime_group_prepare.h").write_text(emit_runtime_group_preparation(bf16_input=True))
    image = directory / "prepare.so"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
                    "-DMX_QUANT_GROUP_WORDS", "-DMX_QUANT_INLINE_GROUP", "-DMX_QUANT_BF16_INPUT",
                    "-I", str(directory), "-I", str(ROOT / "mx_gemmini_support"), *paths,
                    str(ROOT / "mx_gemmini_support/runtime_quant.c"), "-o", str(image)], check=True)
    library = ctypes.CDLL(str(image))
    for name in ["original", "compact"]:
        getattr(library, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    rng = np.random.default_rng(7410)
    for _ in range(2):
        words = rng.integers(0, 65536, (logical[0], logical[2]), dtype=np.uint16)
        words[(words & 0x7F80) == 0x7F80] = 0x8000
        words.flat[:4] = [0, 0x8000, 1, 0x8001]
        unchanged = words.copy()
        wide = (words.astype(np.uint32) << 16).view(np.float32)
        old, new = [np.full(size + 128, 0xA5, dtype=np.uint8) for _ in range(2)]
        library.original(wide.ctypes.data, old.ctypes.data + 64)
        library.compact(words.ctypes.data, new.ctypes.data + 64)
        assert old.tobytes() == new.tobytes()
        assert (new[:64] == 0xA5).all() and (new[-64:] == 0xA5).all()
        assert words.tobytes() == unchanged.tobytes()


def test_unimplemented_implicit_zero_compact_route_refuses():
    profile = load_profile(PROFILE)
    plan = plan_dense_shape(profile, (1, 17, 37), precision="FP8")
    issuer = emit_planned_runtime_issuer(profile, plan.padded_shape_mnk, plan.tile_mnk, precision="FP8")
    with pytest.raises(ValueError, match="implicit zero rows"):
        emit_activation_preparation(issuer, (1, 17, 37), symbol="prepare",
                                    input_dtype="bf16", virtual_zero_padding=True)


@pytest.mark.parametrize("disabled", ["connect_multiplies", "share_activation_packing", "integer_group_packing"])
def test_catalog_transport_requires_explicit_compatible_selections(tmp_path, disabled):
    from mx_gemmini_support.merlin_catalog_builder import build_merlin_catalog

    flags = dict(connect_multiplies=True, share_activation_packing=True, integer_group_packing=True)
    flags[disabled] = False
    with pytest.raises(ValueError, match="VPU matrix transport requires"):
        build_merlin_catalog(tmp_path / "missing.mlir", tmp_path / "out", profile={}, riscv_root=tmp_path,
                              vpu_matrix_transport=True, **flags)


def fixture(tmp_path, *, live=False, barrier=""):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, StringAttr

    result = "(tensor<2x16xf32>,tensor<32xi8>)" if live else "tensor<32xi8>"
    returns = "%p,%q:tensor<2x16xf32>,tensor<32xi8>" if live else "%q:tensor<32xi8>"
    module = parse_mlir_text(f'''module {{
      func.func private @produce(tensor<2x16xf32>,tensor<2x16xf32>)->tensor<2x16xf32>
      func.func private @prepare(tensor<2x16xf32>,tensor<32xi8>)->tensor<32xi8>
      func.func private @unknown()
      func.func @forward(%a:tensor<2x16xf32>)->{result} {{
        %e=tensor.empty():tensor<2x16xf32>
        %p=func.call @produce(%a,%e):(tensor<2x16xf32>,tensor<2x16xf32>)->tensor<2x16xf32>
        %dead=tensor.collapse_shape %p [[0,1]]:tensor<2x16xf32> into tensor<32xf32>
        %f=tensor.collapse_shape %p [[0,1]]:tensor<2x16xf32> into tensor<32xf32>
        %v=tensor.expand_shape %f [[0,1]] output_shape [2,16]:tensor<32xf32> into tensor<2x16xf32>
        {barrier}
        %d=tensor.empty():tensor<32xi8>
        %q=func.call @prepare(%v,%d):(tensor<2x16xf32>,tensor<32xi8>)->tensor<32xi8>
        func.return {returns}
      }}
    }}''')
    for function in module.body.block.ops:
        if function.name == "func.func" and function.sym_name.data in {"produce", "prepare"}:
            function.properties["arg_attrs"] = ArrayAttr([
                DictionaryAttr({"bufferization.access": StringAttr(mode)}) for mode in ["read", "write"]])
    source, catalog, sidecar = [tmp_path / name for name in ["source.mlir", "catalog.json", "routing.json"]]
    source.write_text(text(module))
    first = {"symbol": "produce_bf16", "original_argument_types": ["tensor<2x16xf32>"] * 2,
             "completion": "synchronous_native_issuer"}
    second = {"symbol": "prepare_bf16", "original_argument_types": ["tensor<2x16xf32>", "tensor<32xi8>"],
              "completion": "synchronous_native_issuer"}
    data = {"source_sha256": "source-id", "vpu_matrix_transport": "experimental_private_bf16",
            "bindings": [], "vpu_elementwise": {
                "connected_multiplies": [{"symbol": "produce", "private_bf16_output": first}]},
            "kernels": [{"batched": False, "prepacked_checkpoint_weight": True,
                         "activation_preparation": {"prepare_symbol": "prepare", "private_bf16_input": second}}]}
    catalog.write_text(json.dumps(data))
    sidecar.write_text(json.dumps({"catalog_manifest": str(catalog), "model_sha256": "source-id", "routed": []}))
    return source, catalog, sidecar, data


def test_exact_private_graph_edge_changes_only_internal_representation(tmp_path):
    source, catalog, sidecar, _ = fixture(tmp_path)
    before = source.read_bytes()
    result = MXVPUMatrixTransportPostOffload(catalog)(source, sidecar, tmp_path / "rewrite")
    assert source.read_bytes() == before
    assert "func.call @produce_bf16" in result.read_text() and "func.call @prepare_bf16" in result.read_text()
    receipt = json.loads((result.parent / "mx_vpu_matrix_transport_rewrite.json").read_text())
    assert receipt["removed_unused_static_views"] == len(receipt["edges"]) == 1
    assert receipt["edges"][0]["transport_type"] == "tensor<2x16xbf16>"


@pytest.mark.parametrize("kwargs", [{"live": True}, {"barrier": "func.call @unknown():()->()"}])
def test_observed_outputs_and_effect_barriers_retain_original_endpoints(tmp_path, kwargs):
    source, catalog, sidecar, _ = fixture(tmp_path, **kwargs)
    result = MXVPUMatrixTransportPostOffload(catalog)(source, sidecar, tmp_path / "rewrite")
    assert "func.call @produce_bf16" not in result.read_text()
    assert not json.loads((result.parent / "mx_vpu_matrix_transport_rewrite.json").read_text())["edges"]


@pytest.mark.parametrize("changed", ["producer_type", "consumer_type", "producer_completion", "both_completion", "source"])
def test_changed_endpoint_contract_or_source_refuses(tmp_path, changed):
    source, catalog, sidecar, data = fixture(tmp_path)
    first = data["vpu_elementwise"]["connected_multiplies"][0]["private_bf16_output"]
    second = data["kernels"][0]["activation_preparation"]["private_bf16_input"]
    if changed == "producer_type":
        first["original_argument_types"][0] = "tensor<32xf32>"
    elif changed == "consumer_type":
        second["original_argument_types"][0] = "tensor<32xf32>"
    elif changed == "producer_completion":
        first["completion"] = "unknown"
    elif changed == "both_completion":
        first["completion"] = second["completion"] = "unknown"
    else:
        data["source_sha256"] = "different"
    catalog.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        MXVPUMatrixTransportPostOffload(catalog)(source, sidecar, tmp_path / "rewrite")
