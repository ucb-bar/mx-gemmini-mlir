"""Candidate capture sites stay visible to the selected FP8 adapter."""

from pathlib import Path

import pytest
import torch

from examples.iteration_workloads import make_case
from mx_gemmini_support.handoff import validate_handoff
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
    original = torch.export.export(model, inputs)
    contractions = {torch.ops.aten.linear.default, torch.ops.aten.matmul.default,
                    torch.ops.aten.mm.default, torch.ops.aten.bmm.default}
    for node in original.graph_module.graph.nodes:
        if node.target in contractions:
            assert all(operand.meta["val"].is_contiguous() for operand in node.args[:2])
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


@pytest.mark.parametrize("name,site_count", [
    ("linear_seam", 2),
    ("decoder_block", 6),
    ("vision_patches", 2),
    ("policy_fusion", 6),
])
def test_candidate_model2mlir_handoff(name, site_count):
    m2m = pytest.importorskip("m2m")
    from m2m.capture.external_quantization import ExternalQuantizationConfig
    from m2m.coverage import opaque_report

    contract = ROOT / "contracts/software-spec.yaml"
    policy = ROOT / "examples/default-policy.yaml"
    model, inputs = make_case(name)
    result = m2m.convert(
        model, inputs,
        quantization=ExternalQuantizationConfig("mx_gemmini", contract, policy),
        backend="fx_importer", capture_trace=True,
    )
    assert result.ok, result.diagnostics
    assert result.capture_trace["status"] == "complete", result.capture_trace["blockers"]
    assert not opaque_report(result.mlir_text), opaque_report(result.mlir_text)
    assert len(result.quantization_manifest["sites"]) == site_count
    assert all(site["status"] == "quantized"
               for site in result.quantization_manifest["sites"])
    validate_handoff(result, contract.read_bytes(), policy.read_bytes())
