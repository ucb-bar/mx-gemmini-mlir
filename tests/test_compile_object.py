"""The public object driver selects existing lowerers from typed MX graphs."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from mx_gemmini_support.target_profile import load_profile
from tools.compile_object import _resident_pair_precision, classify


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"


@pytest.mark.parametrize(("mlir_path", "profile_name", "family", "resources"), [
    ("docs/evidence/radiance_plain_mx_profile_trio_266c593/fp4/payload_bound.mlir",
     "MxGemminiRocketConfig", "source_contract", 4),
    ("docs/evidence/nicolas_plain_chain_prefix_ladder_4cf23ef/m96/connected_chain.mlir.gz",
     "MxGemminiRocketConfig", "resident_pair", 0),
    ("docs/evidence/nicolas_fp4_connected_resident_266c593/chain/connected.mlir",
     "MxGemminiRocketConfig", "resident_pair", 0),
    ("docs/evidence/nicolas_fp6_connected_resident_64_266c593/chain/connected.mlir",
     "MxGemminiRocketConfig", "resident_pair", 0),
    ("docs/evidence/nicolas_connected_chain_upstream_e9ded36_20261010/connected_bound.mlir",
     "MxE4M3Fp4VpuGemminiRocketConfig", "resident_vpu_pair", 0),
    ("docs/evidence/nicolas_chain_pipelined_full_266c593/connected.mlir",
     "MxE4M3Fp4VpuGemminiRocketConfig", "full_vpu_branch", 0),
    ("docs/evidence/nicolas_vpu_elementwise_compiled_266c593/add/bound.mlir",
     "MxE4M3VpuGemminiRocketConfig", "vpu_elementwise", 0),
    ("docs/evidence/nicolas_vpu_fused_compiled_266c593/expsum/bound.mlir",
     "MxE4M3VpuGemminiRocketConfig", "vpu_elementwise", 0),
])
def test_compiler_selects_the_verified_graph_family(mlir_path, profile_name,
                                                     family, resources):
    path = ROOT / mlir_path
    content = path.read_bytes()
    if path.name.endswith(".gz"):
        content = gzip.decompress(content)
    selected, report = classify(content.decode(), load_profile(PROFILES / f"{profile_name}.json"))
    assert selected == family
    assert report["source_resources"] == resources
    if "nicolas_fp4_connected_resident" in mlir_path:
        assert _resident_pair_precision(content.decode()) == "fp4_e2m1"
    if "nicolas_fp6_connected_resident" in mlir_path:
        assert _resident_pair_precision(content.decode()) == "fp6_e3m2"


@pytest.mark.parametrize(("name", "profile_name", "resources"), [
    ("mxgemm.fp4.singletile.tm128tn128tk128.requant",
     "MxE4M3Fp4VpuGemminiRocketConfig", 4),
    ("mxgemm.fp6.singletile.tm128tn128tk128.requant",
     "MxE3M2OnlyGemminiRocketConfig", 7),
    ("mxgemm.fp8.singletile.tm128tn128tk128.requant",
     "MxE4M3Fp4VpuGemminiRocketConfig", 4),
])
def test_compiler_accepts_source_bound_host_requant_object(name, profile_name,
                                                           resources):
    path = (ROOT / "docs/evidence/radiance_mx_gemm_latest_e9ded36_ee22/spike" /
            name / "payload_bound.mlir")
    profile = load_profile(PROFILES / f"{profile_name}.json")
    family, report = classify(path.read_text(), profile)
    assert family == "source_contract"
    assert report["source_resources"] == resources


def test_compiler_rejects_ambiguous_or_unsupported_binding():
    profile = load_profile(PROFILES / "MxGemminiRocketConfig.json")
    source = (ROOT / "docs/evidence/radiance_plain_mx_profile_trio_266c593/fp4/"
              "payload_bound.mlir").read_text()
    with pytest.raises(ValueError, match="exactly one payload binding scheme"):
        classify(source.replace("module attributes {", (
            'module attributes {mx.runtime_resources_sha256 = "'
            + "0" * 64 + '\", '), 1), profile)
    frontend = (ROOT / "docs/evidence/model2mlir_nicolas_chain_two_site_profile_bound_20261009.mlir").read_text()
    vpu_profile = load_profile(PROFILES / "MxE4M3Fp4VpuGemminiRocketConfig.json")
    with pytest.raises(ValueError, match="no qualified lowering|exactly one payload binding"):
        classify(frontend, vpu_profile)


def test_compiler_selects_exact_dual_fp4_requant_graph():
    graph = (ROOT / "docs/evidence/nicolas_spad_requant_fp4_266c593/"
             "fp4_capable/connected.mlir").read_text()
    profile = load_profile(PROFILES / "MxE4M3Fp4VpuGemminiRocketConfig.json")
    family, report = classify(graph, profile)
    assert family == "fp4_dual_requant"
    assert report["spad_requants"] == 2
    altered = graph.replace('scale_buffer = "scales_hw2"',
                            'scale_buffer = "scales_hw"')
    with pytest.raises(ValueError, match="output binding differs"):
        classify(altered, profile)
