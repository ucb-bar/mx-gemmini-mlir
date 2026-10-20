"""Portable host arithmetic checks must reject contradictory observations."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "host_softmax_observation_reader",
    ROOT / "docs/evidence/radiance_rocket_optimization_20261019/verify.py",
)
READER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READER)
PATH = "build/device_host_abi/mx_host_softmax_rewrite.json"


def observations(policy="libm_f32_ordered_finite"):
    source, output, sidecar = b"source\n", b"output\n", b"{}\n"
    proof = {
        "numeric_policy": policy,
        "finite_input_contract": True,
        "finite_maximum_specialization": True,
        "profile_sha256": "a" * 64,
        "sidecar_sha256": READER.sha(sidecar),
        "source_sha256": READER.sha(source),
        "output_sha256": READER.sha(output),
        "regions": [{"region": "row0", "shape": [1, 1, 2, 160]}],
    }
    llvm = "@initial = private constant [2 x float] [float 0xFFF0000000000000, float 0xFFF0000000000000]\n"
    llvm += "call float @llvm.maxnum.f32(float %a, float %b)\n"
    llvm += (
        "call float @expf(float %x)\n"
        if policy == "libm_f32_ordered_finite"
        else "call float @fmaf(float %x, float %y, float %z)\n" * 8
    )
    files = {
        PATH: json.dumps(proof).encode(),
        "build/device_signatures.json": sidecar,
        "build/device_host_abi/source.mlir": source,
        "build/device_host_abi/mx_host_softmax.mlir": output,
        "build/lower/model.ll": llvm.encode(),
    }
    probe = {
        "host_softmax_policy": policy,
        "host_softmax_rewrite": proof,
        "profile_sha256": "a" * 64,
        "execution_policy": {"vpu_placements": {"softmax": {"row0": "host"}}},
        "vpu_softmax_inventory": {
            "profile_sha256": "a" * 64,
            "regions": [
                {
                    "region": "row0",
                    "input_shape": [1, 1, 2, 160],
                    "output_shape": [1, 1, 2, 160],
                    "input_dtype": "f32",
                    "output_dtype": "f32",
                }
            ],
        },
    }
    return files, probe


@pytest.mark.parametrize(
    "policy", ["libm_f32_ordered_finite", "torch_2_10_cpu_softmax_f32_avx2"]
)
def test_bound_host_observations(policy):
    files, probe = observations(policy)
    result = READER.check_host_softmax(files, probe)
    assert result["policy"] == policy and result["regions"] == 1
    assert result["negative_infinity_initializer_sizes"] == [2]


@pytest.mark.parametrize(
    "field,value",
    [
        ("numeric_policy", "automatic"),
        ("finite_input_contract", False),
        ("finite_maximum_specialization", False),
        ("source_sha256", "b" * 64),
        ("output_sha256", "b" * 64),
        ("sidecar_sha256", "b" * 64),
        ("regions", []),
        ("regions", [{"region": "row0", "shape": [1, 1, 3, 160]}]),
    ],
)
def test_changed_source_bindings_refuse(field, value):
    files, probe = observations()
    proof = copy.deepcopy(probe["host_softmax_rewrite"])
    proof[field] = value
    files[PATH] = json.dumps(proof).encode()
    probe["host_softmax_rewrite"] = proof
    with pytest.raises(AssertionError):
        READER.check_host_softmax(files, probe)


@pytest.mark.parametrize(
    "before,after",
    [
        ("@llvm.maxnum.f32", "@llvm.maximum.f32"),
        ("@expf", "@different_exp"),
        ("0xFFF0000000000000", "0x41EFF00000000000"),
    ],
)
def test_changed_lowered_arithmetic_refuses(before, after):
    files, probe = observations()
    files["build/lower/model.ll"] = files["build/lower/model.ll"].replace(
        before.encode(), after.encode()
    )
    with pytest.raises(AssertionError):
        READER.check_host_softmax(files, probe)


def test_polynomial_request_with_libm_lowering_refuses():
    files, probe = observations("torch_2_10_cpu_softmax_f32_avx2")
    files["build/lower/model.ll"] = files["build/lower/model.ll"].replace(
        b"@fmaf", b"@expf"
    )
    with pytest.raises(AssertionError):
        READER.check_host_softmax(files, probe)


def test_unrequested_host_rewrite_refuses():
    assert READER.check_host_softmax({}, {}) is None
    files, probe = observations()
    probe["host_softmax_policy"] = None
    with pytest.raises(AssertionError):
        READER.check_host_softmax(files, probe)
