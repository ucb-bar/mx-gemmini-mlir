"""Optional Torch tensor handoff keeps model2MLIR's logical axes intact."""

from hashlib import sha256
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mx_gemmini_support.model2mlir import plan_linear_operands, plan_rank2_operands
from mx_gemmini_support.diagnostic_program import emit_single_window_baremetal_c


@pytest.mark.parametrize("fmt,code", [("mxfp8", 0x38), ("mxfp6", 0x0c), ("mxfp4", 0x02)])
def test_rank2_linear_handoff_packs_codes_and_e8m0(fmt, code):
    scales = torch.tensor([[127] if row < 16 else [128] for row in range(32)], dtype=torch.uint8)
    operands = SimpleNamespace(
        format=fmt,
        activation_codes=torch.full((32, 32), code, dtype=torch.uint8),
        weight_codes=torch.full((32, 32), code, dtype=torch.uint8).T,
        activation_scales=scales,
        weight_scales=scales.clone(),
    )
    lut = [[0, code] + [0] * 14 for _ in range(16)] if fmt == "mxfp6" else None
    payload = plan_linear_operands(operands, activation_lut=lut, weight_lut=lut)
    assert (payload.m, payload.k, payload.n) == (32, 32, 32)
    assert len(payload.waves) == 1
    assert payload.waves[0].activation_scale_bytes == bytes([127] * 16 + [128] * 16)
    assert payload.waves[0].weight_scale_bytes == bytes([127] * 16 + [128] * 16)
    assert len(payload.waves[0].activation_bytes) == (1024 if fmt == "mxfp8" else 512)


def test_rank2_linear_handoff_rejects_non_byte_codes():
    operands = SimpleNamespace(
        format="mxfp8",
        activation_codes=torch.ones(32, 32),
        weight_codes=torch.ones(32, 32, dtype=torch.uint8),
        activation_scales=torch.full((32, 1), 127, dtype=torch.uint8),
        weight_scales=torch.full((32, 1), 127, dtype=torch.uint8),
    )
    with pytest.raises(TypeError, match="activation_codes"):
        plan_linear_operands(operands)


@pytest.mark.parametrize("fmt,zero_block,source_hash", [
    ("mxfp8", False, "3beeef63b9273924d62a60b5a6052538b0f7b3f551efca989b609a06926dfdbb"),
    ("mxfp6", False, "525520867b90a533013e071803ca02f0cda5ba0ddbf23d583d8aa4e57395e055"),
    ("mxfp4", False, "dc07f3dac57a9c1ae289e230507d243167ff61d5d3cfc619783bd48fc4bf1cee"),
    ("mxfp8", True, "61efeae2809454c810f3721b7261f4678c1f42dafb52e129e9009646808bf3e3"),
    ("mxfp6", True, "b415026448cb3868fad9cc86c79f957a34616112dfe0e8553b03340e6e652993"),
    ("mxfp4", True, "93ddb08fba4827e18ac8a7300a077f146ad358daaa2b8c3bf236c806018063ee"),
])
def test_torchao_linear_to_rtl_test_source(fmt, zero_block, source_hash):
    """Reproduce the exact C source previously run on the selected RTL simulator."""
    pytest.importorskip("torchao")
    pytest.importorskip("m2m")
    from m2m.capture import mx_gemmini_quant
    from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization

    if not hasattr(mx_gemmini_quant, "linear_contraction_operands"):
        pytest.skip("the installed model2MLIR lacks the rank-2 MX operand handoff")

    class OneLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project = torch.nn.Linear(32, 32, bias=False)

        def forward(self, value):
            return self.project(value)

    model = OneLinear().eval()
    with torch.no_grad():
        model.project.weight[:16].fill_(0.0 if zero_block else 1.0)
        model.project.weight[16:].fill_(0.0 if zero_block else 2.0)
    activation = torch.zeros(32, 32) if zero_block else torch.ones(32, 32)
    if not zero_block:
        activation[16:].fill_(2.0)
    captured = apply_quantization(
        model, QuantizationConfig(scheme=f"mx_gemmini_{fmt[2:]}"),
        example_inputs=(activation,),
    )
    assert captured._m2m_quantization_stats["torchao_linear_modules"] == 1
    assert isinstance(model.project, mx_gemmini_quant.MXGemminiLinear)
    operands = mx_gemmini_quant.linear_contraction_operands(model.project, activation)
    code = 0 if zero_block else {"mxfp8": 0x38, "mxfp6": 0x0c, "mxfp4": 0x02}[fmt]
    if zero_block:
        assert set(operands.activation_codes.flatten().tolist()) == {0}
        assert set(operands.weight_codes.flatten().tolist()) == {0}
        assert set(operands.activation_scales.flatten().tolist()) == {104}
        assert set(operands.weight_scales.flatten().tolist()) == {104}
    lut = (
        [list(range(16)) for _ in range(16)] if zero_block else
        [[0, code] + [0] * 14 for _ in range(16)]
    ) if fmt == "mxfp6" else None
    payload = plan_linear_operands(operands, activation_lut=lut, weight_lut=lut)
    expected = [[0] * 32 for _ in range(32)] if zero_block else [
        [0x4200 if row < 16 and col < 16 else
         0x4280 if row < 16 or col < 16 else 0x4300
         for col in range(32)]
        for row in range(32)
    ]
    source = emit_single_window_baremetal_c(payload, expected)
    assert sha256(source.encode()).hexdigest() == source_hash
    functional = mx_gemmini_quant.functional_contraction_operands(
        activation, model.project.weight.T, fmt
    )
    functional_payload = plan_rank2_operands(functional, activation_lut=lut, weight_lut=lut)
    assert functional_payload == payload
