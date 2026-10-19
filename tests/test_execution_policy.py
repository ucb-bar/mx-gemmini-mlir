"""The model policy must bind every selected MX/VPU site to its capture."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import re

import pytest

from mx_gemmini_support.execution_policy import (
    make_policy, reconcile_vpu_region_sets, validate_policy,
)
from mx_gemmini_support.merlin_catalog_builder import select_routed_vpu_sites
from mx_gemmini_support.merlin_vpu_add import MXVPUAddPostOffload, MXVPUPostOffload
from mx_gemmini_support.target_profile import load_profile
from tools.compare_mx_model_quality import _kl, compare_smolvla, compare_tinyllama
from tools.report_stage_profile import parse_gemmini_perf

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROFILE = load_profile(ROOT / "profiles/gemmini-mx-cleanup-266c593/"
                       "MxE4M3Fp4VpuGemminiRocketConfig.json")
SOURCE = gzip.decompress((ROOT / "docs/evidence/"
                          "full_model_compile_preflight_a042643_748b984/"
                          "tinyllama.mlir.gz").read_bytes()).decode()


def test_policy_binds_all_matrix_and_vpu_sites(monkeypatch) -> None:
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(ROOT))
    policy = make_policy(SOURCE, PROFILE)
    resolved = validate_policy(policy, SOURCE, PROFILE)
    assert len(policy["matrix_formats"]) == 20
    assert len(resolved["precision_overrides"]) == 15
    assert len(resolved["enabled_vpu_regions"]["add"]) == 13
    changed = copy.deepcopy(policy)
    changed["vpu_placements"]["add"].pop(next(iter(changed["vpu_placements"]["add"])))
    with pytest.raises(ValueError, match="VPU site placements"):
        validate_policy(changed, SOURCE, PROFILE)
    changed = copy.deepcopy(policy)
    changed["source_mlir_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="captured source"):
        validate_policy(changed, SOURCE, PROFILE)
    changed = copy.deepcopy(policy)
    changed["matrix_formats"]["matmul_0"] = "host"
    with pytest.raises(ValueError, match="matrix site formats"):
        validate_policy(changed, SOURCE, PROFILE)


@pytest.mark.parametrize("region", ("opaque", "opaque#add_metadata", "opaque#mul_metadata"))
@pytest.mark.parametrize("operation,primitive,placement", (
    ("mul", "arith.mulf", "vpu"),
    ("sub", "arith.subf", "vpu"),
    ("div", "arith.divf", "vpu"),
    ("add", "arith.addf", "host"),
))
def test_default_elementwise_placement_uses_arithmetic_not_region_label(
        region, operation, primitive, placement):
    source = f'''module attributes {{prov.level = "linalg-on-tensors"}} {{
      func.func @forward(%a:tensor<1x2x1xf32>,%b:tensor<1x1x3xf32>)->tensor<1x2x3xf32> {{
        %e=tensor.empty():tensor<1x2x3xf32>
        %r=linalg.generic {{
          indexing_maps=[affine_map<(a,b,c)->(a,b,0)>,affine_map<(a,b,c)->(a,0,c)>,affine_map<(a,b,c)->(a,b,c)>],
          iterator_types=["parallel","parallel","parallel"]
        }} ins(%a,%b:tensor<1x2x1xf32>,tensor<1x1x3xf32>) outs(%e:tensor<1x2x3xf32>)
        attrs = {{prov.region_id="{region}",prov.op="{operation}",prov.aten="aten.{operation}.Tensor"}} {{
          ^bb0(%x:f32,%y:f32,%out:f32):
            %v={primitive} %x,%y:f32
            linalg.yield %v:f32
        }}->tensor<1x2x3xf32>
        func.return %r:tensor<1x2x3xf32>
      }}
    }}'''
    policy = make_policy(source, PROFILE)
    selected = f"{region}#{operation}0"
    assert policy["vpu_placements"]["elementwise"] == {selected: placement}
    resolved = validate_policy(policy, source, PROFILE)
    assert resolved["enabled_vpu_regions"]["elementwise"] == (
        {selected} if placement == "vpu" else set())


def test_add_rewriter_obeys_exact_site_selection(tmp_path: Path) -> None:
    from mx_gemmini_support.merlin_vpu_add import selected_add_sites

    sites = selected_add_sites(SOURCE, PROFILE)
    selected = frozenset((sites[0]["region"],))
    source = tmp_path / "model.mlir"
    sidecar = tmp_path / "sidecar.json"
    source.write_text(SOURCE)
    sidecar.write_text("{}")
    out = MXVPUAddPostOffload(PROFILE, enabled_regions=selected)(
        source, sidecar, tmp_path / "rewritten")
    assert out != source
    assert out.read_text().count('func.call @mx_vpu_add_') == 1
    with pytest.raises(ValueError, match="unknown source regions"):
        MXVPUAddPostOffload(PROFILE, enabled_regions=frozenset(("missing",)))(
            source, sidecar, tmp_path / "bad")


def test_catalog_emits_only_routed_vpu_sites() -> None:
    sites = [{"region": "add_0", "symbol": "a"},
             {"region": "add_1", "symbol": "b"}]
    assert select_routed_vpu_sites("add", sites, None) == sites
    assert select_routed_vpu_sites(
        "add", sites, {"add": frozenset(("add_1",))}) == sites[1:]
    assert select_routed_vpu_sites("add", sites, {"add": frozenset()}) == []
    with pytest.raises(ValueError, match="prepared source"):
        select_routed_vpu_sites("add", sites,
                                {"add": frozenset(("missing",))})


def test_prepared_scalar_add_keeps_its_source_route() -> None:
    available = {name: set() for name in
                 ("softmax", "add", "elementwise", "unary", "reduce")}
    available["elementwise"] = {"add_2#add0", "mul_0#mul0"}
    enabled = {name: frozenset() for name in available}
    enabled["add"] = frozenset(("add_2",))
    enabled["elementwise"] = frozenset(("mul_0#mul0",))
    routed, aliases = reconcile_vpu_region_sets(available, enabled)
    assert routed["add"] == frozenset()
    assert routed["elementwise"] == frozenset(
        ("add_2#add0", "mul_0#mul0"))
    assert aliases == {"add_2": "add_2#add0"}
    available["elementwise"].remove("add_2#add0")
    with pytest.raises(ValueError, match="prepared equivalent"):
        reconcile_vpu_region_sets(available, enabled)


def test_connected_vpu_rewrite_obeys_policy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(ROOT))
    policy = make_policy(SOURCE, PROFILE)
    routes = validate_policy(policy, SOURCE, PROFILE)["enabled_vpu_regions"]
    source = tmp_path / "model.mlir"
    sidecar = tmp_path / "sidecar.json"
    source.write_text(SOURCE)
    sidecar.write_text("{}")
    rewritten = MXVPUPostOffload(PROFILE, enabled_site_regions=routes)(
        source, sidecar, tmp_path / "rewritten")
    assert rewritten != source
    for name, selected in routes.items():
        receipt = tmp_path / "rewritten" / f"mx_vpu_{name}_rewrite.json"
        if selected:
            assert {row["region"] for row in
                    json.loads(receipt.read_text())["regions"]} == selected
        else:
            assert not receipt.exists()


def test_tiny_quality_gate_uses_kl_and_top_tokens() -> None:
    reference = np.array([[4.0, 0.0], [0.0, 4.0]])
    source = np.array([[3.0, 1.0], [1.0, 3.0]])
    assert compare_tinyllama(reference, source, source)["passed"]
    degraded = np.array([[1.0, 3.0], [3.0, 1.0]])
    report = compare_tinyllama(reference, source, degraded)
    assert not report["passed"]
    assert not report["checks"]["top1_agreement"]


def test_archived_nicolas_late_logits_define_quality_target() -> None:
    archive = ROOT / "docs/evidence/nicolas_tinyllama_onepass160_same_input_20261014"
    reference = np.load(archive / "fp32_late_logits.npy", allow_pickle=False)
    bf16 = np.fromfile(archive / "source_reference_late_bf16.bin",
                       dtype="<u2").reshape(33, 32000)
    source = (bf16.astype("<u4") << 16).view("<f4")
    kl = _kl(reference, source)
    assert np.mean(kl) == pytest.approx(0.0641712695, abs=1e-8)
    assert np.max(kl) == pytest.approx(0.1896034181, abs=1e-8)
    assert np.count_nonzero(np.argmax(reference, axis=1) ==
                            np.argmax(source, axis=1)) == 27


def test_full_depth_selected_vpu_control_misses_quality_gate() -> None:
    source_dir = ROOT / "docs/evidence/nicolas_tinyllama_full_source_20261014"
    compiler_dir = ROOT / "docs/evidence/tinyllama_full_safe_vpu_20261014"
    float_logits = np.load(source_dir / "pytorch_last_logits.npy",
                           allow_pickle=False).reshape(1, -1)
    source_log = gzip.decompress((source_dir / "source_spike.log.gz").read_bytes())
    source_hex = re.search(rb"^last_logits_bf16 ([0-9a-f]+)$", source_log,
                           re.MULTILINE).group(1)
    source_bits = np.fromiter((int(source_hex[i:i + 4], 16)
                               for i in range(0, len(source_hex), 4)),
                              dtype="<u4")
    source_logits = (source_bits << 16).view("<f4").reshape(1, -1)
    compiler_log = gzip.decompress((compiler_dir / "spike.log.gz").read_bytes())
    compiler_bits = [int(word) for line in compiler_log.splitlines()
                     if line.startswith(b"OUT_CHUNK f32 ")
                     for word in line.split()[4:]]
    compiler_logits = np.asarray(compiler_bits, dtype="<u4").view("<f4")
    report = compare_tinyllama(float_logits, source_logits,
                               compiler_logits.reshape(1, -1))
    archived = json.loads((compiler_dir / "quality_gate.json").read_text())
    for key in ("source", "compiler", "checks", "passed"):
        assert report[key] == archived[key]
    assert not report["passed"]


def test_smol_quality_gate_checks_each_action_dimension() -> None:
    reference = np.ones((50, 32))
    source = reference + 0.01
    compiler = source.copy()
    assert compare_smolvla(reference, source, compiler)["passed"]
    compiler[:, 31] += 0.01
    report = compare_smolvla(reference, source, compiler)
    assert not report["passed"]
    assert not report["checks_by_dimension"]["rel_rmse_by_dim"][31]


def test_spike_perf_counters_are_available_for_profile_attribution() -> None:
    source = ROOT / "docs/evidence/nicolas_tinyllama_onepass160_same_input_20261014"
    values = parse_gemmini_perf((source / "compiler_perf.txt").read_text())
    assert values["mode"] == "e4m3_vpu"
    assert values["mesh_tiles"] == 40970280
    assert values["vpu_commands"] == 44872
    assert values["load_bytes"] == 4407640384
