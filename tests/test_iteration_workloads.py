"""Candidate capture sites stay visible to the selected FP8 adapter."""

from pathlib import Path

import pytest

from examples.iteration_workloads import make_case
from mx_gemmini_support.m2m_adapter import apply


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,quantized,functional", [
    ("linear_seam", 2, 0),
    ("decoder_block", 6, 2),
    ("vision_patches", 2, 0),
    ("policy_fusion", 6, 2),
])
def test_candidate_fp8_site_census(name, quantized, functional):
    model, inputs = make_case(name)
    graph, manifest = apply(
        model, inputs,
        contract_bytes=(ROOT / "contracts/software-spec.yaml").read_bytes(),
        policy_bytes=(ROOT / "examples/default-policy.yaml").read_bytes(),
    )
    sites = manifest["sites"]
    assert len(sites) == quantized
    assert all(site["status"] == "quantized" and site["format"] == "mxfp8"
               for site in sites)
    assert sum(site["site_id"].startswith("functional:") for site in sites) == functional
    assert graph.module()(*inputs).shape == model(*inputs).shape
