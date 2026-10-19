"""Whole-model connections bind private source arithmetic and selected sites."""

import hashlib
import json
import tarfile
from pathlib import Path

import pytest
from merlin.frontends.linalg_mlir import parse_mlir_file, parse_mlir_text

from mx_gemmini_support.merlin_connected_sigmoid import (
    MXConnectedSigmoidPostOffload,
    chain_record,
    sigmoid_chains,
)
from mx_gemmini_support.merlin_vpu_elementwise import selected_elementwise_sites
from mx_gemmini_support.merlin_vpu_unary import selected_unary_sites
from mx_gemmini_support.target_profile import load_profile, profile_sha256

ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json")


@pytest.fixture(scope="module")
def source():
    with tarfile.open(ROOT / "docs/evidence/sigmoid_transport_20261018/executions.tar.xz", "r:xz") as archive:
        return archive.extractfile("tail137/capture/source.mlir").read().decode()


def selected(source, *, enabled_unary=True, enabled_elements=True):
    return sigmoid_chains(parse_mlir_text(source),
                          selected_elementwise_sites(source, PROFILE) if enabled_elements else [],
                          selected_unary_sites(source, PROFILE) if enabled_unary else [])


@pytest.mark.parametrize("shape", ["137", "1x3x2731", "1x64x5632"])
def test_match_uses_typed_source_at_multiple_ranks_and_tails(source, shape):
    rank = len(shape.split("x"))
    source = source.replace("1x137", shape).replace("d0, d1", ", ".join(f"d{i}" for i in range(rank)))
    source = source.replace('["parallel", "parallel"]', json.dumps(["parallel"] * rank))
    chain, = selected(source)
    assert chain["shape"] == list(map(int, shape.split("x")))
    assert chain["regions"] == ["mul_0#mul0", "mul_1#mul0"]
    assert chain["unary_region"] == "sigmoid_0#sigmoid0"


@pytest.mark.parametrize("kind", ["unary", "elements"])
def test_host_placement_of_a_member_prevents_connection(source, kind):
    assert not selected(source, **{"enabled_" + kind: False})


def test_provenance_cannot_override_changed_arithmetic_or_observable_intermediate(source):
    assert not selected(source.replace("1.000000e+00", "2.000000e+00", 1))
    assert not selected(source.replace("ins(%0, %3", "ins(%1, %3", 1))
    assert not selected(source.replace("ins(%12, %1", "ins(%12, %3", 1))


def test_unknown_call_blocks_pure_source_movement(source):
    source = source.replace("func.func @forward", "func.func private @effect()\n  func.func @forward", 1)
    source = source.replace("%11 = tensor.empty()", "func.call @effect() : () -> ()\n    %11 = tensor.empty()", 1)
    assert not selected(source)


def test_operand_order_has_a_distinct_endpoint_for_exceptional_fallback(source):
    normal, = selected(source)
    commuted, = selected(source.replace("ins(%0, %3", "ins(%3, %0", 1))
    assert normal["gate_first"] and not commuted["gate_first"]
    assert normal["symbol"] != commuted["symbol"]


def test_second_multiply_operand_order_requires_its_own_sigmoid_qualification(source):
    assert not selected(source.replace("ins(%12, %1", "ins(%1, %12", 1))


@pytest.mark.parametrize("missing", ["connect_multiplies", "prepack_immutable_weights", "enabled_vpu_regions"])
def test_direct_catalog_api_requires_explicit_experiment_prerequisites(tmp_path, missing):
    from mx_gemmini_support.merlin_catalog_builder import build_merlin_catalog

    options = {"connect_multiplies": True, "prepack_immutable_weights": True, "enabled_vpu_regions": {}}
    options[missing] = None if missing == "enabled_vpu_regions" else False
    with pytest.raises(ValueError, match="sigmoid chains require"):
        build_merlin_catalog(tmp_path / "unread.mlir", tmp_path / "out", profile=PROFILE,
                             riscv_root=tmp_path, connect_sigmoid_multiplies=True, **options)
    assert not (tmp_path / "out").exists()


def bound_files(tmp_path, source):
    prepared, sidecar, catalog = (tmp_path / name for name in ("source.mlir", "sidecar.json", "catalog.json"))
    prepared.write_text(source)
    record = chain_record(selected(source)[0])
    catalog.write_text(json.dumps({
        "source_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
        "profile_sha256": profile_sha256(PROFILE), "connect_sigmoid_multiplies": True,
        "resolved_vpu_regions": {"elementwise": record["regions"], "unary": [record["unary_region"]]},
        "vpu_elementwise": {"connected_sigmoid_multiplies": [record]}}))
    sidecar.write_text(json.dumps({"catalog_manifest": str(catalog),
                                 "model_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest()}))
    return prepared, sidecar, catalog


def test_source_rewrite_has_one_checked_dps_call_and_three_source_obligations(tmp_path, source):
    prepared, sidecar, catalog = bound_files(tmp_path, source)
    output, records = MXConnectedSigmoidPostOffload(catalog, PROFILE)(prepared, sidecar, tmp_path / "out")
    module = parse_mlir_file(output)
    module.verify()
    calls = [op for op in module.walk() if op.name == "func.call"]
    assert len(calls) == 1 and len(calls[0].arguments) == 3
    assert not [op for op in module.walk() if op.name == "linalg.generic"]
    assert records == [chain_record(selected(source)[0])]
    assert prepared.read_text() == source
    receipt = json.loads(output.with_name("mx_connected_sigmoid_rewrite.json").read_text())
    assert receipt["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", ["profile", "missing", "symbol", "host", "source"])
def test_catalog_mismatch_refuses_before_rewriting(tmp_path, source, mutation):
    prepared, sidecar, catalog = bound_files(tmp_path, source)
    data = json.loads(catalog.read_text())
    if mutation == "profile":
        data["profile_sha256"] = "0" * 64
    elif mutation == "missing":
        data["vpu_elementwise"]["connected_sigmoid_multiplies"] = []
    elif mutation == "symbol":
        data["vpu_elementwise"]["connected_sigmoid_multiplies"][0]["symbol"] = "uncompiled"
    elif mutation == "host":
        data["resolved_vpu_regions"]["unary"] = []
    else:
        prepared.write_text(source.replace("1.000000e+00", "2.000000e+00", 1))
    catalog.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        MXConnectedSigmoidPostOffload(catalog, PROFILE)(prepared, sidecar, tmp_path / "out")
    assert not (tmp_path / "out/mx_connected_sigmoid.mlir").exists()
