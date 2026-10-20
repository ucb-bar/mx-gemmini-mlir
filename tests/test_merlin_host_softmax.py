"""Explicit host arithmetic must bind a complete, closed softmax decomposition."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from mx_gemmini_support.merlin_host_softmax import MXHostSoftmaxPostOffload
from mx_gemmini_support.merlin_vpu_softmax import selected_softmax_sites
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
)
POLICY = "torch_2_10_cpu_softmax_f32_avx2"


@pytest.fixture(autouse=True)
def merlin_available():
    pytest.importorskip("merlin.llvmlower.cpu_softmax")


def rewrite(tmp_path, source, regions=None, policy=POLICY):
    pytest.importorskip("merlin.frontends.linalg_mlir")
    prepared = tmp_path / "source.mlir"
    prepared.write_text(source)
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text("{}")
    if regions is None:
        regions = frozenset(
            row["region"] for row in selected_softmax_sites(source, PROFILE)
        )
    return MXHostSoftmaxPostOffload(PROFILE, policy=policy, enabled_regions=regions)(
        prepared, sidecar, tmp_path / "rewrite"
    )


@pytest.fixture
def source():
    return gzip.decompress(
        (
            ROOT
            / "docs/evidence/full_model_compile_preflight_a042643_748b984/tinyllama.mlir.gz"
        ).read_bytes()
    ).decode()


def test_source_policy_replaces_complete_regions(tmp_path, source):
    from merlin.frontends.linalg_mlir import parse_mlir_file, parse_mlir_text

    original_exp = sum(op.name == "math.exp" for op in parse_mlir_text(source).walk())
    output = rewrite(tmp_path, source)
    module = parse_mlir_file(output)
    module.verify()
    assert not any(
        getattr(op.attributes.get("prov.op"), "data", None) == "softmax"
        for op in module.walk()
    )
    assert any(op.name == "math.fma" for op in module.walk())
    assert sum(op.name == "arith.maxnumf" for op in module.walk()) == 2
    assert sum(op.name == "math.exp" for op in module.walk()) == original_exp - 2
    record = json.loads((output.parent / "mx_host_softmax_rewrite.json").read_text())
    assert record["numeric_policy"] == POLICY
    assert record["finite_input_contract"]
    assert record["finite_maximum_specialization"]
    assert len(record["regions"]) == 2


def test_unselected_regions_are_preserved(tmp_path, source):
    regions = frozenset([selected_softmax_sites(source, PROFILE)[0]["region"]])
    output = rewrite(tmp_path, source, regions)
    assert len(selected_softmax_sites(output.read_text(), PROFILE)) == 1


def test_libm_policy_preserves_every_other_source_operation(tmp_path, source):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    original = text(parse_mlir_text(source))
    assert original.count("arith.maximumf") == 2
    output = rewrite(tmp_path, source, policy="libm_f32_ordered_finite")
    assert output.read_text() == original.replace("arith.maximumf", "arith.maxnumf")


def test_rne_policy_guards_only_exp_underflow(tmp_path, source):
    from merlin.frontends.linalg_mlir import parse_mlir_file

    output = rewrite(tmp_path, source, policy="libm_f32_ordered_rne")
    module = parse_mlir_file(output)
    module.verify()
    branches = [op for op in module.walk() if op.name == "scf.if"]
    assert len(branches) == 2
    for branch in branches:
        assert [op.name for op in branch.true_region.block.ops] == ["scf.yield"]
        assert [op.name for op in branch.false_region.block.ops] == [
            "math.exp",
            "scf.yield",
        ]
    record = json.loads((output.parent / "mx_host_softmax_rewrite.json").read_text())
    assert record["exponential_contract"] == {
        "domain": "nonpositive",
        "rounding_mode": "rne",
        "floating_effects": "unobserved",
        "underflow_cutoff": -104.0,
        "underflow_result_bits": 0,
    }


@pytest.mark.parametrize(
    "before,after",
    [
        ("arith.maximumf", "arith.minimumf"),
        ("arith.subf", "arith.addf"),
        ("math.exp", "math.absf"),
        ("arith.divf", "arith.mulf"),
    ],
)
def test_rejects_changed_scalar_arithmetic(tmp_path, source, before, after):
    # Limit the mutation to a softmax block, retaining the rest of the model.
    index = source.index('prov.region_id = "softmax_0"')
    start = source.index(before, index)
    mutated = source[:start] + source[start:].replace(before, after, 1)
    with pytest.raises(ValueError, match="arithmetic changed"):
        rewrite(tmp_path, mutated)


def test_rejects_unknown_region(tmp_path, source):
    with pytest.raises(ValueError, match="missing source regions"):
        rewrite(tmp_path, source, frozenset(["missing"]))


def test_rejects_implicit_or_unknown_policy():
    with pytest.raises(ValueError, match="unsupported"):
        MXHostSoftmaxPostOffload(
            PROFILE, policy="automatic", enabled_regions=frozenset()
        )
