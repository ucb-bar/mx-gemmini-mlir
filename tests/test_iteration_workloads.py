"""Candidate capture sites stay visible to the selected FP8 adapter."""

from pathlib import Path

import pytest
import torch
import yaml

from examples.iteration_workloads import make_case
from mx_gemmini_support.handoff import validate_handoff
from mx_gemmini_support.m2m_adapter import apply, derive_site_inventory


ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = (
    "contracts/software-spec.yaml",
    "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml",
)


@pytest.mark.parametrize("contract_path", CONTRACTS)
def test_selective_mixed_policy_keeps_host_activations(contract_path):
    m2m = pytest.importorskip("m2m")
    from m2m.capture.external_quantization import ExternalQuantizationConfig

    model, inputs = make_case("linear_seam")
    contract = ROOT / contract_path
    policy = ROOT / "examples/selective-policy.yaml"
    result = m2m.convert(
        model, inputs,
        quantization=ExternalQuantizationConfig("mx_gemmini", contract, policy),
        backend="fx_importer", capture_trace=True,
    )
    assert result.ok, result.diagnostics
    assert result.capture_trace["status"] == "complete", result.capture_trace["blockers"]
    assert [(row["site_id"], row["status"], row.get("format"))
            for row in result.quantization_manifest["sites"]] == [
                ("module:expand", "quantized", "mxfp8"),
                ("module:project", "quantized", "mxfp4"),
            ]
    validate_handoff(result, contract.read_bytes(), policy.read_bytes())


@pytest.mark.parametrize("contract_path", CONTRACTS)
def test_site_inventory_derives_all_formats_without_mutating_model(contract_path):
    model, inputs = make_case("decoder_block")
    inventory = derive_site_inventory(model, inputs,
                                      contract_bytes=(ROOT / contract_path).read_bytes())
    assert derive_site_inventory(model, inputs,
                                 contract_bytes=(ROOT / contract_path).read_bytes()) == inventory
    assert inventory["schema"] == "mx_gemmini.site_inventory.v1"
    assert inventory["status"] == "structural_candidates_only"
    assert len(inventory["source_graph_sha256"]) == 64
    assert len(inventory["sites"]) == 6
    assert {site["kind"] for site in inventory["sites"]} == {"linear", "functional"}
    assert all(site["eligible_formats"] == ["mxfp4", "mxfp6", "mxfp8"]
               for site in inventory["sites"])
    assert isinstance(model.qkv, torch.nn.Linear)


def test_site_inventory_requires_eval_model():
    model, inputs = make_case("linear_seam")
    model.train()
    with pytest.raises(ValueError, match="requires an eval"):
        derive_site_inventory(model, inputs,
                              contract_bytes=(ROOT / CONTRACTS[0]).read_bytes())
    assert model.training


def test_fp6_attention_and_fp4_fp8_linears_use_exact_sites():
    from m2m.capture.trace import capture_frontend_snapshot

    model, inputs = make_case("decoder_block")
    contract = (ROOT / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml").read_bytes()
    original = capture_frontend_snapshot(model, inputs)
    inventory = derive_site_inventory(model, inputs, contract_bytes=contract)
    functional = [row["site_id"] for row in inventory["sites"] if row["kind"] == "functional"]
    assert len(functional) == 2
    # Fixture codes exercise the mechanism; model-accuracy review must choose real codebooks.
    policy = yaml.safe_dump({
        "schema": "mx_gemmini.quantization_policy.v1",
        "default_format": "host",
        "source_graph_sha256": original["sha256"],
        "module_overrides": {"qkv": "mxfp8", "up": "mxfp4"},
        "functional_overrides": {site_id: "mxfp6" for site_id in functional},
        "fp6_codebooks": {"default": {"status": "reviewed",
                                      "activation": list(range(16)),
                                      "weight": list(range(16))}},
    }).encode()
    graph, manifest = apply(model, inputs, contract_bytes=contract,
                            policy_bytes=policy, original_frontend_snapshot=original)
    sites = {row["site_id"]: row for row in manifest["sites"]}
    assert sites["module:qkv"]["format"] == "mxfp8"
    assert sites["module:up"]["format"] == "mxfp4"
    assert all(sites[site_id]["format"] == "mxfp6" for site_id in functional)
    assert sites["module:attn_out"]["status"] == "host"
    assert sites["module:down"]["status"] == "host"
    assert graph.module()(*inputs).shape == model(*inputs).shape


@pytest.mark.parametrize("contract_path", CONTRACTS)
@pytest.mark.parametrize("name,quantized,functional", [
    ("linear_seam", 2, 0),
    ("decoder_block", 6, 2),
    ("vision_patches", 2, 0),
    ("policy_fusion", 6, 2),
])
def test_candidate_fp8_site_census(contract_path, name, quantized, functional):
    model, inputs = make_case(name)
    original = torch.export.export(model, inputs)
    contractions = {torch.ops.aten.linear.default, torch.ops.aten.matmul.default,
                    torch.ops.aten.mm.default, torch.ops.aten.bmm.default}
    for node in original.graph_module.graph.nodes:
        if node.target in contractions:
            assert all(operand.meta["val"].is_contiguous() for operand in node.args[:2])
    graph, manifest = apply(
        model, inputs,
        contract_bytes=(ROOT / contract_path).read_bytes(),
        policy_bytes=(ROOT / "examples/default-policy.yaml").read_bytes(),
    )
    sites = manifest["sites"]
    assert len(sites) == quantized
    assert all(site["status"] == "quantized" and site["format"] == "mxfp8"
               for site in sites)
    assert sum(site["site_id"].startswith("functional:") for site in sites) == functional
    assert graph.module()(*inputs).shape == model(*inputs).shape


@pytest.mark.parametrize("contract_path", CONTRACTS)
@pytest.mark.parametrize("name,site_count", [
    ("linear_seam", 2),
    ("decoder_block", 6),
    ("vision_patches", 2),
    ("policy_fusion", 6),
])
def test_candidate_model2mlir_handoff(contract_path, name, site_count):
    m2m = pytest.importorskip("m2m")
    from m2m.capture.external_quantization import ExternalQuantizationConfig
    from m2m.coverage import opaque_report

    contract = ROOT / contract_path
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
