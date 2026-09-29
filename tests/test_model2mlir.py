"""Optional Torch tensor handoff keeps model2MLIR's logical axes intact."""

from hashlib import sha256
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mx_gemmini_support.model2mlir import (
    iter_spatial_tiles,
    plan_independent_batches,
    plan_linear_operands,
    plan_rank2_operands,
)
from mx_gemmini_support.diagnostic_program import (
    emit_independent_batches_baremetal_c,
    emit_single_window_baremetal_c,
    emit_spatial_tiles_baremetal_c,
)


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


@pytest.mark.parametrize("fmt,one,two", [
    ("mxfp8", 0x38, 0x40), ("mxfp6", 0x0c, 0x10), ("mxfp4", 0x02, 0x04),
])
def test_spatial_tiles_preserve_batch_and_global_row_column_axes(fmt, one, two):
    activation = torch.full((2, 1, 64, 32), one, dtype=torch.uint8)
    activation[:, :, 32:].fill_(two)
    weight = torch.full((2, 1, 32, 64), one, dtype=torch.uint8)
    weight[:, :, :, 32:].fill_(two)
    activation_scales = torch.full((2, 1, 64, 1), 127, dtype=torch.uint8)
    activation_scales[1].fill_(128)
    weight_scales = torch.full((2, 1, 64, 1), 127, dtype=torch.uint8)
    operands = SimpleNamespace(
        format=fmt, activation_codes=activation, weight_codes=weight,
        activation_scales=activation_scales, weight_scales=weight_scales,
    )
    first_lut = [[0, one, two] + [0] * 13 for _ in range(16)]
    second_lut = [[0, two, one] + [0] * 13 for _ in range(16)]
    lut = first_lut + second_lut if fmt == "mxfp6" else None
    tiles = list(iter_spatial_tiles(operands, activation_lut=lut, weight_lut=lut))
    assert [(t.batch_index, t.m_start, t.n_start) for t in tiles] == [
        (batch, m, n) for batch in ((0, 0), (1, 0)) for m in (0, 32) for n in (0, 32)
    ]
    assert all((t.m_stop - t.m_start, t.n_stop - t.n_start) == (32, 32) for t in tiles)
    assert all((t.payload.m, t.payload.k, t.payload.n) == (32, 32, 32) for t in tiles)
    assert all(t.payload.waves[0].activation_scale_bytes == bytes([128] * 32)
               for t in tiles[4:])
    if fmt == "mxfp6":
        assert tiles[0].payload.waves[0].activation_bytes == tiles[2].payload.waves[0].activation_bytes
        assert tiles[0].payload.waves[0].weight_bytes == tiles[1].payload.waves[0].weight_bytes
        assert tiles[0].payload.activation_lut_bytes != tiles[2].payload.activation_lut_bytes
        assert tiles[0].payload.weight_lut_bytes != tiles[1].payload.weight_lut_bytes
    else:
        assert set(tiles[0].payload.waves[0].activation_bytes) != set(
            tiles[2].payload.waves[0].activation_bytes)
        assert set(tiles[0].payload.waves[0].weight_bytes) != set(
            tiles[1].payload.waves[0].weight_bytes)


def test_spatial_tiles_refuse_misaligned_geometry_and_scale_axes():
    operands = SimpleNamespace(
        format="mxfp8",
        activation_codes=torch.ones(32, 32, dtype=torch.uint8),
        weight_codes=torch.ones(32, 32, dtype=torch.uint8),
        activation_scales=torch.ones(32, 2, dtype=torch.uint8),
        weight_scales=torch.ones(32, 1, dtype=torch.uint8),
    )
    with pytest.raises(ValueError, match="activation E8M0 axes"):
        list(iter_spatial_tiles(operands))
    operands.activation_scales = torch.ones(32, 1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="aligned values"):
        list(iter_spatial_tiles(operands, tile_m=24))


def test_spatial_tiles_keep_large_linear_scale_windows_bounded():
    operands = SimpleNamespace(
        format="mxfp8",
        activation_codes=torch.full((32, 2048), 0x38, dtype=torch.uint8),
        weight_codes=torch.full((2048, 2048), 0x38, dtype=torch.uint8),
        activation_scales=torch.full((32, 64), 127, dtype=torch.uint8),
        weight_scales=torch.full((2048, 64), 127, dtype=torch.uint8),
    )
    tiles = list(iter_spatial_tiles(operands))
    assert len(tiles) == 64
    assert [tile.n_start for tile in tiles] == list(range(0, 2048, 32))
    assert all((tile.payload.m, tile.payload.k, tile.payload.n) == (32, 2048, 32)
               for tile in tiles)
    assert all(len(tile.payload.waves) == 1 for tile in tiles)
    assert all(len(tile.payload.waves[0].activation_scale_bytes) == 2048
               and len(tile.payload.waves[0].weight_scale_bytes) == 2048
               for tile in tiles)


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


@pytest.mark.parametrize("fmt,code", [("mxfp8", 0x38), ("mxfp6", 0x0c), ("mxfp4", 0x02)])
def test_functional_attention_batch_axes_are_packed_independently(fmt, code):
    pytest.importorskip("torchao")
    pytest.importorskip("m2m")
    from m2m.capture import mx_gemmini_quant

    if not hasattr(mx_gemmini_quant, "functional_contraction_operands"):
        pytest.skip("the installed model2MLIR lacks the functional MX operand handoff")
    lhs = torch.ones(2, 3, 32, 32)
    lhs[1].fill_(2.0)
    rhs = torch.ones(2, 3, 32, 32)
    rhs[..., 16:].fill_(2.0)
    operands = mx_gemmini_quant.functional_contraction_operands(lhs, rhs, fmt)
    lut = [[0, code] + [0] * 14 for _ in range(16)] if fmt == "mxfp6" else None
    packed = plan_independent_batches(operands, activation_lut=lut, weight_lut=lut)
    assert [row.batch_index for row in packed] == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)
    ]
    assert all((row.payload.m, row.payload.k, row.payload.n) == (32, 32, 32) for row in packed)
    assert packed[0].payload.waves[0].activation_scale_bytes == bytes([127] * 32)
    assert packed[-1].payload.waves[0].activation_scale_bytes == bytes([128] * 32)
    assert packed[-1].payload.waves[0].weight_scale_bytes == bytes([127] * 16 + [128] * 16)


@pytest.mark.parametrize("fmt,small,golden,normal_code,source_hash", [
    ("mxfp8", 2.0**-9, 0x3b00, 0x38, "4c7c7318eacad1e523383643d9fb9c0cad7b731be596d2c6c95bf92c22c902a1"),
    ("mxfp6", 2.0**-4, 0x3d80, 0x0c, "2899cb226035723ffd9cc52b503333bb22f6c214b7cba236d2873691a99ad767"),
    ("mxfp4", 2.0**-1, 0x3f00, 0x02, "eac9aaf43f2c30422048c9a614ca0905679ca4fc152e6c11fbc07a020deb6df0"),
])
def test_functional_element_subnormal_source_matches_rtl_run(fmt, small, golden, normal_code, source_hash):
    pytest.importorskip("torchao")
    pytest.importorskip("m2m")
    from m2m.capture import mx_gemmini_quant

    if not hasattr(mx_gemmini_quant, "functional_contraction_operands"):
        pytest.skip("the installed model2MLIR lacks the functional MX operand handoff")
    lhs = torch.zeros(32, 32)
    lhs[:, 0] = 1.0
    lhs[:, 1] = small
    rhs = torch.zeros(32, 32)
    rhs[1, :] = 1.0
    operands = mx_gemmini_quant.functional_contraction_operands(lhs, rhs, fmt)
    assert set(operands.activation_codes[:, 1].tolist()) == {1}
    assert set(operands.activation_codes[:, 0].tolist()) == {normal_code}
    assert set(operands.activation_scales.flatten().tolist()) == {127}
    assert set(operands.weight_scales.flatten().tolist()) == {127}
    lut = [list(range(16)) for _ in range(16)] if fmt == "mxfp6" else None
    payload = plan_rank2_operands(operands, activation_lut=lut, weight_lut=lut)
    source = emit_single_window_baremetal_c(payload, [[golden] * 32 for _ in range(32)])
    assert sha256(source.encode()).hexdigest() == source_hash


@pytest.mark.parametrize("fmt,code,source_hash", [
    ("mxfp8", 0x38, "6c10f5bf537e5c99346db32acb694d8f045e327bbe50b29a39f9ae715fb39b8e"),
    ("mxfp6", 0x0c, "f9c187480b1c2c5db2ed5ca2262b3e2999be2e73f3482cc4714d36d749724c77"),
    ("mxfp4", 0x02, "ac1bf3b30f67667b228f5da9793f52f1170c75019889786c045d870ed2b44437"),
])
def test_functional_independent_batches_emit_distinct_executions(fmt, code, source_hash):
    pytest.importorskip("torchao")
    pytest.importorskip("m2m")
    from m2m.capture.mx_gemmini_quant import functional_contraction_operands

    lhs = torch.ones(2, 32, 32)
    lhs[1].fill_(2.0)
    rhs = torch.ones(2, 32, 32)
    operands = functional_contraction_operands(lhs, rhs, fmt)
    lut = [[0, code] + [0] * 14 for _ in range(16)] if fmt == "mxfp6" else None
    indexed = plan_independent_batches(operands, activation_lut=lut, weight_lut=lut)
    assert [row.batch_index for row in indexed] == [(0,), (1,)]
    assert indexed[0].payload.waves[0].activation_scale_bytes == bytes([127] * 32)
    assert indexed[1].payload.waves[0].activation_scale_bytes == bytes([128] * 32)
    source = emit_independent_batches_baremetal_c([
        (indexed[0], [[0x4200] * 32 for _ in range(32)]),
        (indexed[1], [[0x4280] * 32 for _ in range(32)]),
    ])
    assert source.count("gemmini_loop_ws_spad(") == 2
    assert source.count("static int run_batch_") == 2
    assert "batch (0,)" in source and "batch (1,)" in source
    assert sha256(source.encode()).hexdigest() == source_hash


@pytest.mark.parametrize("fmt,code,source_hash", [
    ("mxfp8", 0x38, "434dab1e9b110ed50b24d398b4b1d7a556e0a708d241b684ed43fc53dcec31a7"),
    ("mxfp6", 0x0c, "0e0fcb0f25f633311d229e57c0e14a241c68a24d539f482b3cf0aa1ff37c7fd7"),
    ("mxfp4", 0x02, "028ad046e5833005e8aad29ef88301590253be6f3b6386c47493fecb61fe951b"),
])
def test_functional_spatial_tiles_emit_four_quadrants(fmt, code, source_hash):
    pytest.importorskip("torchao")
    pytest.importorskip("m2m")
    from m2m.capture.mx_gemmini_quant import functional_contraction_operands

    lhs = torch.ones(64, 32)
    lhs[32:].fill_(2.0)
    rhs = torch.ones(32, 64)
    rhs[:, 32:].fill_(2.0)
    operands = functional_contraction_operands(lhs, rhs, fmt)
    lut = [[0, code] + [0] * 14 for _ in range(32)] if fmt == "mxfp6" else None
    tiles = list(iter_spatial_tiles(operands, activation_lut=lut, weight_lut=lut))
    expected = [
        [0x4200 if row < 32 and col < 32 else
         0x4280 if row < 32 or col < 32 else 0x4300
         for col in range(64)]
        for row in range(64)
    ]
    source = emit_spatial_tiles_baremetal_c(tiles, expected)
    assert source.count("gemmini_loop_ws_spad(") == 4
    assert "tile (0, 0)" in source and "tile (32, 32)" in source
    assert sha256(source.encode()).hexdigest() == source_hash
