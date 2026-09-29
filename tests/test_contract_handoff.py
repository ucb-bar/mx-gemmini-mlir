"""A mixed-site capture reaches the digest-gated OOT dialect boundary."""

from pathlib import Path

import pytest
import torch
from torch import nn

from mx_gemmini_support.contract import compile_contract
from mx_gemmini_support.handoff import render_handoff, validate_handoff
from mx_gemmini_support.m2m_adapter import apply


SPEC = Path(__file__).resolve().parents[1] / "contracts/software-spec.yaml"


class _Mixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(32, 32)
        self.b = nn.Parameter(torch.randn(32, 32))

    def forward(self, value):
        return self.a(value) @ self.b


def _policy():
    codes = ", ".join(str(value) for value in range(16))
    return ("schema: mx_gemmini.quantization_policy.v1\n"
            "default_format: mxfp8\n"
            "module_overrides: {a: mxfp6}\n"
            "functional_overrides: {functional:matmul: mxfp4}\n"
            "output_chains:\n"
            "  module:a: {consumer: 'functional:matmul', format: mxfp4}\n"
            "fp6_codebooks:\n"
            "  default:\n"
            "    status: reviewed\n"
            f"    activation: [{codes}]\n"
            f"    weight: [{codes}]\n")


def test_mixed_adapter_requires_reviewed_fp6_codebooks():
    model = _Mixed().eval()
    with pytest.raises(ValueError, match="reviewed"):
        apply(model, (torch.randn(32, 32),), contract_bytes=SPEC.read_bytes(),
              policy_bytes=b"schema: mx_gemmini.quantization_policy.v1\ndefault_format: mxfp6\n")


def test_host_module_does_not_reenter_functional_quantization():
    graph, manifest = apply(
        _Mixed().eval(), (torch.randn(32, 32),), contract_bytes=SPEC.read_bytes(),
        policy_bytes=(b"schema: mx_gemmini.quantization_policy.v1\n"
                      b"default_format: mxfp8\nmodule_overrides: {a: host}\n"),
    )
    assert {row["site_id"]: row["status"] for row in manifest["sites"]} == {
        "module:a": "host", "functional:matmul": "quantized"}
    assert graph(torch.randn(32, 32)).shape == (32, 32)


def test_mixed_capture_and_resident_chain_handoff(tmp_path):
    m2m = pytest.importorskip("m2m")
    from m2m.capture.external_quantization import ExternalQuantizationConfig
    from m2m.capture.trace import capture_frontend_snapshot

    torch.manual_seed(3)
    model = _Mixed().eval()
    inputs = (torch.randn(32, 32),)
    original = capture_frontend_snapshot(model, inputs)
    assert original["status"] == "complete"
    policy_text = _policy().replace(
        "functional_overrides:",
        f"source_graph_sha256: {original['sha256']}\nfunctional_overrides:",
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text(policy_text)
    result = m2m.convert(model, inputs,
                         quantization=ExternalQuantizationConfig("mx_gemmini", SPEC, policy),
                         backend="fx_importer", capture_trace=True,
                         original_frontend_snapshot=original)
    assert result.ok, result.diagnostics
    assert any("0 opaque" in row for row in result.diagnostics)
    selected = validate_handoff(result, SPEC.read_bytes(), policy.read_bytes())
    assert selected["contract"]["formats"].keys() == {"mxfp8", "mxfp6", "mxfp4"}
    sites = result.quantization_manifest["sites"]
    assert {(row["site_id"], row.get("format")) for row in sites} == {
        ("module:a", "mxfp6"), ("functional:matmul", "mxfp4")}
    rendered = render_handoff(result, SPEC.read_bytes(), policy.read_bytes())
    assert '"mx_gemmini.requantize"' in rendered
    assert '"mx_gemmini.contract"(%resident, %resident_scales' in rendered
    with pytest.raises(ValueError, match="contract_sha256"):
        validate_handoff(result, SPEC.read_bytes() + b"\n# altered\n", policy.read_bytes())


def test_source_contract_projection_has_output_requantization():
    projection = compile_contract(SPEC.read_bytes())
    assert projection["zero_block_scale_e8m0"] == 104
    assert projection["output_requantization"]["scale_resident"] is True
    assert set(projection["output_requantization"]["formats"]) == {"mxfp8", "mxfp6", "mxfp4"}
