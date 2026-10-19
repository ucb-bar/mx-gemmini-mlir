"""Host placement is explicit, source-bound, and independently re-proven after preparation."""

import copy
import json

import pytest
from test_merlin_bf16_prepare import PROFILE, ROOT

from mx_gemmini_support.execution_policy import make_policy, validate_policy
from mx_gemmini_support.merlin_catalog_plan import plan_merlin_catalog
from mx_gemmini_support.merlin_singleton_prepare import MXSingletonReductionPreparation


def source(k=1, region="arbitrary_region"):
    return f'''builtin.module attributes {{prov.level = "linalg-on-tensors"}} {{
      func.func @forward(%a:tensor<1x7x{k}xf32>,%b:tensor<1x{k}x19xf32>) -> tensor<1x7x19xf32> {{
        %zero=arith.constant 0.0:f32
        %init=tensor.splat %zero:tensor<1x7x19xf32>
        %r=linalg.generic {{indexing_maps=[affine_map<(b,m,n,k)->(b,m,k)>,
          affine_map<(b,m,n,k)->(b,k,n)>,affine_map<(b,m,n,k)->(b,m,n)>],
          iterator_types=["parallel","parallel","parallel","reduction"]}}
          ins(%a,%b:tensor<1x7x{k}xf32>,tensor<1x{k}x19xf32>) outs(%init:tensor<1x7x19xf32>)
          attrs={{prov.region_id="{region}",prov.op="batch_matmul",prov.family="contraction",
                 prov.aten="aten.bmm.default"}} {{
          ^bb0(%x:f32,%y:f32,%z:f32):
            %p=arith.mulf %x,%y:f32
            %s=arith.addf %z,%p:f32
            linalg.yield %s:f32
          }} -> tensor<1x7x19xf32>
        func.return %r:tensor<1x7x19xf32>
      }}
    }}'''


@pytest.mark.parametrize("region", ["arbitrary_region", "renamed_with_no_model_identity"])
def test_v2_policy_selects_semantically_proven_singleton_only(region):
    raw = source(region=region)
    original = make_policy(raw, PROFILE)
    assert original["schema"].endswith(".v1")
    assert original["matrix_formats"] == {region: "mxfp8"}
    selected = make_policy(raw, PROFILE, preserve_singleton_reductions=True)
    assert selected["matrix_formats"] == {region: "host"}
    assert validate_policy(selected, raw, PROFILE)["host_singleton_regions"] == {region}
    invalid = copy.deepcopy(selected)
    invalid["schema"] = "mx_gemmini.execution_policy.v1"
    with pytest.raises(ValueError, match="matrix site formats"):
        validate_policy(invalid, raw, PROFILE)
    nonunit = source(2, region=region)
    candidate = make_policy(nonunit, PROFILE, preserve_singleton_reductions=True)
    assert candidate["matrix_formats"] == {region: "mxfp8"}
    candidate["matrix_formats"][region] = "host"
    with pytest.raises(ValueError, match="matrix site formats"):
        validate_policy(candidate, nonunit, PROFILE)


def test_prepared_matrix_is_removed_from_device_catalog_and_fully_accounted(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(ROOT))
    model = tmp_path / "source.mlir"
    model.write_text(source())
    out = MXSingletonReductionPreparation(frozenset({"arbitrary_region"}))(
        model, tmp_path / "prepare")
    assert model.read_text() == source()
    report = json.loads((out.parent / "singleton_reductions.json").read_text())
    row, = report["regions"]
    assert row["region"] == "arbitrary_region"
    assert row["removed_reduction_axes"] == [3]
    assert row["scalar_operations"] == ["arith.mulf", "arith.addf", "linalg.yield"]
    plan = plan_merlin_catalog(out, PROFILE)
    assert plan["bindings"] == [] and plan["unbound_precision_proposals"] == []


@pytest.mark.parametrize("k,regions", [(2, {"arbitrary_region"}), (1, {"missing"}),
                                      (1, {"arbitrary_region", "missing"})])
def test_prepared_source_must_reprove_exact_host_coverage(tmp_path, k, regions):
    model = tmp_path / "source.mlir"
    model.write_text(source(k))
    with pytest.raises(ValueError, match="unique proven prepared singleton"):
        MXSingletonReductionPreparation(frozenset(regions))(model, tmp_path / "prepare")
    assert not (tmp_path / "prepare/singleton_reductions.json").exists()


def test_downstream_receives_only_verified_transformed_source(tmp_path):
    model = tmp_path / "source.mlir"
    model.write_text(source())
    calls = []

    def downstream(path, directory):
        assert "\"reduction\"" not in path.read_text()
        calls.append((path, directory))
        return path

    out = MXSingletonReductionPreparation(frozenset({"arbitrary_region"}), downstream=downstream)(
        model, tmp_path / "prepare")
    assert calls == [(out, tmp_path / "prepare/bf16")]
