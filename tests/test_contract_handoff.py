"""A mixed-site capture reaches the digest-gated OOT dialect boundary."""

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from mx_gemmini_support.contract import compile_contract
from mx_gemmini_support.handoff import render_handoff, validate_handoff
from mx_gemmini_support.legality import shape_reason
from mx_gemmini_support.m2m_adapter import apply, derive_site_inventory


SPEC = Path(__file__).resolve().parents[1] / "contracts/software-spec.yaml"
CANDIDATE = Path(__file__).resolve().parents[1] / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"


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
    assert graph.module()(torch.randn(32, 32)).shape == (32, 32)


def test_host_default_selects_only_named_mixed_precision_modules():
    class FourSites(nn.Module):
        def __init__(self):
            super().__init__()
            self.fp6 = nn.Linear(32, 32)
            self.fp4 = nn.Linear(32, 32)
            self.fp8 = nn.Linear(32, 32)
            self.host = nn.Linear(32, 32)
            self.weight = nn.Parameter(torch.randn(32, 32))

        def forward(self, value):
            return self.host(self.fp8(self.fp4(self.fp6(value)))) @ self.weight

    codes = ", ".join(str(value) for value in range(16))
    policy = ("schema: mx_gemmini.quantization_policy.v1\n"
              "default_format: host\n"
              "module_overrides: {fp6: mxfp6, fp4: mxfp4, fp8: mxfp8}\n"
              "fp6_codebooks:\n"
              "  default:\n"
              "    status: reviewed\n"
              f"    activation: [{codes}]\n"
              f"    weight: [{codes}]\n")
    model = FourSites().eval()
    graph, manifest = apply(model, (torch.randn(32, 32),),
                            contract_bytes=SPEC.read_bytes(), policy_bytes=policy.encode())
    assert [(site["site_id"], site["status"], site.get("format"))
            for site in manifest["sites"]] == [
                ("module:fp6", "quantized", "mxfp6"),
                ("module:fp4", "quantized", "mxfp4"),
                ("module:fp8", "quantized", "mxfp8"),
                ("module:host", "host", None),
                ("functional:matmul", "host", None),
            ]
    assert graph.module()(torch.randn(32, 32)).shape == (32, 32)


def test_explicit_ineligible_linear_fails_before_torchao_mutation():
    model = nn.Sequential(nn.Linear(32, 32)).eval()
    with pytest.raises(ValueError, match="explicit MX site module:0 is ineligible: M=113"):
        apply(model, (torch.randn(113, 32),), contract_bytes=SPEC.read_bytes(),
              policy_bytes=(b"schema: mx_gemmini.quantization_policy.v1\n"
                            b"default_format: host\nmodule_overrides: {'0': mxfp6}\n"))
    assert isinstance(model[0], nn.Linear)


def test_explicit_ineligible_functional_fails_before_torchao_mutation():
    from m2m.capture.trace import capture_frontend_snapshot

    class Strided(nn.Module):
        def forward(self, value):
            return value @ value.transpose(0, 1)

    model = Strided().eval()
    inputs = (torch.randn(32, 32),)
    original = capture_frontend_snapshot(model, inputs)
    policy = ("schema: mx_gemmini.quantization_policy.v1\n"
              "default_format: host\n"
              f"source_graph_sha256: {original['sha256']}\n"
              "functional_overrides: {'functional:matmul': mxfp6}\n")
    with pytest.raises(ValueError, match="explicit MX site functional:matmul is ineligible: noncontiguous"):
        apply(model, inputs, contract_bytes=SPEC.read_bytes(),
              policy_bytes=policy.encode(), original_frontend_snapshot=original)


def test_shape_rule_comes_from_selected_contract():
    contract = compile_contract(SPEC.read_bytes())
    assert shape_reason(contract, "mxfp8", 32, 32, 32) is None
    contract["formats"]["mxfp8"]["shape_bounds"]["M"]["min"] = 64
    assert shape_reason(contract, "mxfp8", 32, 32, 32) == "M=32 outside mxfp8 shape bounds"


def test_torchao_handler_uses_selected_contract_shape_rule():
    source = yaml.safe_load(SPEC.read_bytes())
    bounds = source["operations"]["contraction_mxfp8"]["shape_bounds"]["M"]
    bounds.update(min=64, multiple_of=64)
    contract = yaml.safe_dump(source).encode()
    policy = b"schema: mx_gemmini.quantization_policy.v1\ndefault_format: mxfp8\n"
    small = nn.Sequential(nn.Linear(32, 32)).eval()
    _, skipped = apply(small, (torch.randn(32, 32),),
                       contract_bytes=contract, policy_bytes=policy)
    assert skipped["sites"][0]["status"] == "skipped"
    large = nn.Sequential(nn.Linear(32, 32)).eval()
    graph, selected = apply(large, (torch.randn(64, 32),),
                            contract_bytes=contract, policy_bytes=policy)
    assert selected["sites"][0]["status"] == "quantized"
    assert large[0].m_rule == (64, 64)
    assert graph.module()(torch.randn(64, 32)).shape == (64, 32)


def test_inventory_reports_fp8_only_shape():
    model = nn.Sequential(nn.Linear(32, 16)).eval()
    inventory = derive_site_inventory(model, (torch.randn(16, 32),),
                                      contract_bytes=SPEC.read_bytes())
    assert inventory["sites"] == [{
        "site_id": "module:0", "kind": "linear", "eligible_formats": ["mxfp8"],
        "refusals": {
            "mxfp4": "N=16 outside mxfp4 shape bounds",
            "mxfp6": "N=16 outside mxfp6 shape bounds",
        },
    }]


@pytest.mark.parametrize("rows,status", [(113, "skipped"), (32, "quantized")])
def test_linear_selection_checks_captured_activation_rows(rows, status):
    class OneLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(32, 32)

        def forward(self, value):
            return self.a(value)

    graph, manifest = apply(
        OneLinear().eval(), (torch.randn(rows, 32),),
        contract_bytes=SPEC.read_bytes(),
        policy_bytes=b"schema: mx_gemmini.quantization_policy.v1\ndefault_format: mxfp8\n",
    )
    assert graph.module()(torch.randn(rows, 32)).shape == (rows, 32)
    assert len(manifest["sites"]) == 1
    assert manifest["sites"][0]["status"] == status
    if status == "quantized":
        assert manifest["sites"][0]["shape"] == [rows, 32, 32]
    else:
        assert "M=113" in manifest["sites"][0]["reason"]


def test_reused_linear_requires_distinct_call_site_ids():
    class Reused(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(32, 32)

        def forward(self, value):
            return self.a(value) + self.a(value)

    with pytest.raises(ValueError, match="multiple static calls"):
        apply(Reused().eval(), (torch.randn(32, 32),),
              contract_bytes=SPEC.read_bytes(),
              policy_bytes=b"schema: mx_gemmini.quantization_policy.v1\ndefault_format: mxfp8\n")


def test_noncontiguous_linear_is_reported_as_skipped():
    class StridedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(32, 32)

        def forward(self, value):
            return self.a(value.transpose(0, 1))

    _, manifest = apply(
        StridedLinear().eval(), (torch.randn(32, 32),),
        contract_bytes=SPEC.read_bytes(),
        policy_bytes=b"schema: mx_gemmini.quantization_policy.v1\ndefault_format: mxfp8\n",
    )
    assert [(site["status"], site.get("reason")) for site in manifest["sites"]] == [
        ("skipped", "noncontiguous or unknown Linear operand layout")]


def test_noncontiguous_functional_operand_is_reported_as_skipped():
    class StridedMatmul(nn.Module):
        def forward(self, value):
            return value @ value.transpose(0, 1)

    _, manifest = apply(
        StridedMatmul().eval(), (torch.randn(32, 32),),
        contract_bytes=SPEC.read_bytes(),
        policy_bytes=b"schema: mx_gemmini.quantization_policy.v1\ndefault_format: mxfp8\n",
    )
    assert [(site["status"], site.get("reason")) for site in manifest["sites"]] == [
        ("skipped", "noncontiguous operand layout")]


def test_sdpa_exposes_qk_and_pv_with_host_softmax():
    from mx_gemmini_support.torchao_quant import expose_sdpa_contractions

    class Attention(nn.Module):
        def forward(self, query, key, value, mask):
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, mask, scale=0.125)

    model = Attention().eval()
    query = torch.randn(1, 2, 32, 64)
    key = torch.randn(1, 2, 32, 64)
    value = torch.randn(1, 2, 32, 64)
    mask = torch.ones(1, 1, 32, 32, dtype=torch.bool)
    mask[..., 0, :] = False
    inputs = (query, key, value, mask)
    exposed = expose_sdpa_contractions(torch.export.export(model, inputs))
    targets = [node.target for node in exposed.graph_module.graph.nodes]
    assert targets.count(torch.ops.aten.matmul.default) == 2
    assert torch.ops.aten._safe_softmax.default in targets
    assert torch.ops.aten.scaled_dot_product_attention.default not in targets
    for node in exposed.graph_module.graph.nodes:
        if node.target == torch.ops.aten.matmul.default:
            assert all(arg.meta["val"].is_contiguous() for arg in node.args[:2])
    torch.testing.assert_close(exposed.module()(*inputs), model(*inputs))

    graph, manifest = apply(
        model, inputs, contract_bytes=SPEC.read_bytes(),
        policy_bytes=b"schema: mx_gemmini.quantization_policy.v1\ndefault_format: mxfp8\n",
    )
    assert [site["status"] for site in manifest["sites"]] == ["quantized", "quantized"]
    assert [site["shape"] for site in manifest["sites"]] == [[32, 32, 64], [32, 64, 32]]
    assert torch.isfinite(graph.module()(*inputs)).all()

    m2m = pytest.importorskip("m2m")
    from m2m.capture.external_quantization import ExternalQuantizationConfig
    result = m2m.convert(
        Attention().eval(), inputs,
        quantization=ExternalQuantizationConfig(
            "mx_gemmini", SPEC, Path(__file__).resolve().parents[1] / "examples/default-policy.yaml"),
        backend="fx_importer", capture_trace=True,
    )
    assert result.ok, result.diagnostics
    assert any("0 opaque" in row for row in result.diagnostics)
    assert result.capture_trace["status"] == "complete", result.capture_trace["blockers"]
    assert len(validate_handoff(
        result, SPEC.read_bytes(),
        (Path(__file__).resolve().parents[1] / "examples/default-policy.yaml").read_bytes(),
    )["manifest"]["sites"]) == 2


def test_sdpa_rejects_unreviewed_causal_mode():
    from mx_gemmini_support.torchao_quant import expose_sdpa_contractions

    class Causal(nn.Module):
        def forward(self, query, key, value):
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True)

    value = torch.randn(1, 2, 32, 64)
    exported = torch.export.export(Causal().eval(), (value, value, value))
    with pytest.raises(ValueError, match="causal mode"):
        expose_sdpa_contractions(exported)


@pytest.mark.parametrize("contract_path", [SPEC, CANDIDATE])
def test_mixed_capture_and_resident_chain_handoff(tmp_path, contract_path):
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
                         quantization=ExternalQuantizationConfig("mx_gemmini", contract_path, policy),
                         backend="fx_importer", capture_trace=True,
                         original_frontend_snapshot=original)
    assert result.ok, result.diagnostics
    assert any("0 opaque" in row for row in result.diagnostics)
    assert result.capture_trace["status"] == "complete", result.capture_trace["blockers"]
    selected = validate_handoff(result, contract_path.read_bytes(), policy.read_bytes())
    assert selected["contract"]["formats"].keys() == {"mxfp8", "mxfp6", "mxfp4"}
    sites = result.quantization_manifest["sites"]
    assert {(row["site_id"], row.get("format")) for row in sites} == {
        ("module:a", "mxfp6"), ("functional:matmul", "mxfp4")}
    rendered = render_handoff(result, contract_path.read_bytes(), policy.read_bytes())
    assert '"mx_gemmini.requantize"' in rendered
    assert '"mx_gemmini.contract"(%resident, %resident_scales' in rendered
    with pytest.raises(ValueError, match="contract_sha256"):
        validate_handoff(result, contract_path.read_bytes() + b"\n# altered\n", policy.read_bytes())
    other = CANDIDATE if contract_path == SPEC else SPEC
    with pytest.raises(ValueError, match="contract_sha256"):
        validate_handoff(result, other.read_bytes(), policy.read_bytes())


def test_source_contract_projection_has_output_requantization():
    projection = compile_contract(SPEC.read_bytes())
    assert projection["rtl_config_class"] == "GemminiMxFPStandaloneConfig"
    assert projection["zero_block_scale_e8m0"] == 104
    assert projection["output_requantization"]["scale_resident"] is True
    assert set(projection["output_requantization"]["formats"]) == {"mxfp8", "mxfp6", "mxfp4"}


@pytest.mark.parametrize("field,replacement", [
    ("rtl_commit", "0" * 40),
    ("mxgen_commit", "0" * 40),
    ("rtl_config", "GemminiMxFPConfigs.e4m3SingleNoLutMxFPConfig"),
    ("rtl_config_class", "GemminiMxFPSingleNoLutConfig"),
])
def test_operand_kernel_rejects_unselected_revision_or_config(field, replacement):
    from mx_gemmini_support.torchao_quant import verify_kernel_contract

    projection = compile_contract(SPEC.read_bytes())
    projection[field] = replacement
    with pytest.raises(ValueError, match="needs review"):
        verify_kernel_contract(projection)


def test_candidate_operand_kernel_requires_exact_revision_pair():
    from mx_gemmini_support.torchao_quant import verify_kernel_contract

    candidate = compile_contract(CANDIDATE.read_bytes())
    assert candidate["status"] == "unreviewed"
    verify_kernel_contract(candidate)
    candidate["mxgen_commit"] = compile_contract(SPEC.read_bytes())["mxgen_commit"]
    with pytest.raises(ValueError, match="needs review"):
        verify_kernel_contract(candidate)
