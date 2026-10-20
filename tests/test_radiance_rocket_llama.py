"""Reject incomplete port evidence and check its matched live-quantization policy."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.qualify_radiance_rocket_llama import parse_console

ROOT = Path(__file__).resolve().parents[1]


def console():
    return "\n".join(
        [
            "PORT_PHASE 0 1 10",
            "PORT_PHASES_COMPLETED 1",
            "PORT_TOTAL 12 guards=0",
            "PORT_BYTES 0 0 2 80 3f",
            "PORT_DONE",
        ]
    )


def test_complete_port_output():
    total, phases, tensors = parse_console(console(), [{"bytes": 2}], 1, True)
    assert total == 12 and phases == [{"phase": 0, "operation": 1, "cycles": 10}]
    assert tensors == {0: bytes.fromhex("803f")}


@pytest.mark.parametrize(
    "before,after",
    [
        ("PORT_DONE", ""),
        ("PORT_PHASES_COMPLETED 1", "PORT_PHASES_COMPLETED 0"),
        ("guards=0", "guards=1"),
        ("PORT_TOTAL 12", "PORT_TOTAL 0"),
        ("PORT_BYTES 0 0 2 80 3f", ""),
        ("PORT_BYTES 0 0 2 80 3f", "PORT_BYTES 0 1 2 80 3f"),
        ("PORT_BYTES 0 0 2 80 3f", "PORT_BYTES 0 0 2 100 3f"),
        ("PORT_PHASE 0 1 10", "PORT_PHASE 1 1 10"),
        ("PORT_DONE", "gemmini perf: MODEL DEADLOCK\nPORT_DONE"),
    ],
)
def test_incomplete_corrupt_or_deadlocked_port_refuses(before, after):
    with pytest.raises(ValueError):
        parse_console(console().replace(before, after), [{"bytes": 2}], 1, True)


def test_phase_instrumentation_must_match_build():
    with pytest.raises(ValueError, match="phase timings"):
        parse_console(console(), [{"bytes": 2}], 1, False)


def test_port_quantizer_matches_published_bf16_input_policy(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("host C compiler unavailable")
    (tmp_path / "rocket_data.h").write_text(
        "\n".join(f"#define RK_REGION{i}_BYTES 64u" for i in (1, 2, 3))
    )
    output = tmp_path / "quant.so"
    subprocess.run(
        [
            cc,
            "-O2",
            "-shared",
            "-fPIC",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-DRK_QUANT_TEST",
            "-DRK_ROUND_QUANT_INPUTS",
            "-DMX_QUANT_REFERENCE",
            "-I",
            str(tmp_path),
            str(ROOT / "tools/fixtures/radiance_rocket_llama.c"),
            str(ROOT / "mx_gemmini_support/runtime_quant.c"),
            "-lm",
            "-o",
            str(output),
        ],
        check=True,
    )
    library = ctypes.CDLL(str(output))
    p_float, p_byte = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8)
    library.rk_quant_test.argtypes = [p_float, p_byte, p_byte]
    library.mx_pack_activation_f32.argtypes = [
        p_float,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        p_byte,
        p_byte,
    ]
    rng = np.random.default_rng(192)
    groups = [np.zeros(32, np.float32), np.full(32, -0.0, np.float32)]
    groups += [
        np.ldexp(rng.normal(size=32).astype(np.float32), exponent)
        for exponent in (-120, -25, -23, -7, 0, 10, 110)
    ]
    # Adjacent F32 values around BF16 and FP8 rounding boundaries.
    for center in (1.0625, 1.1875, 1.99609375, 0.001953125):
        group = np.linspace(center - 0.008, center + 0.008, 32, dtype=np.float32)
        group[0], group[1] = np.nextafter(np.float32(center), -np.inf), center
        group[2] = np.nextafter(np.float32(center), np.inf)
        groups.append(group)
    for group in groups:
        codes, scales = (ctypes.c_uint8 * 32)(), (ctypes.c_uint8 * 1)()
        expected, expected_scale = (ctypes.c_uint8 * 512)(), (ctypes.c_uint8 * 16)()
        library.rk_quant_test(group.ctypes.data_as(p_float), codes, scales)
        tiled = np.tile(group, (16, 1))
        assert (
            library.mx_pack_activation_f32(
                tiled.ctypes.data_as(p_float), 16, 32, 8, expected, expected_scale
            )
            == 0
        )
        assert bytes(codes) == bytes(expected)[:32]
        assert bytes(scales) == bytes(expected_scale)[:1]


def test_quality_failure_outside_late_rows_withholds_speedup():
    from tools.compare_radiance_rocket_forward import quality_and_ratio

    reference = np.zeros((160, 32000), dtype=np.float32)
    compiler = reference.copy()
    compiler[0, 0] = 10
    result = quality_and_ratio(reference, reference, compiler, 200, 100)
    assert result["quality_late_33"]["passed"]
    assert not result["quality_all_160"]["passed"]
    assert result["qualified_speedup"] is None
    assert result["unqualified_source_to_compiler_cycle_ratio"] == 2


def test_equal_quality_full_outputs_allow_speedup():
    from tools.compare_radiance_rocket_forward import quality_and_ratio

    values = np.zeros((160, 32000), dtype=np.float32)
    result = quality_and_ratio(values, values, values, 200, 100)
    assert result["numerical_gate_passed"]
    assert result["qualified_speedup"] == 2
    assert result["qualified_cycle_reduction"] == 0.5


def test_complete_outputs_do_not_excuse_missing_modeled_work():
    from tools.qualify_radiance_rocket_llama import check_accelerator_work

    program = np.zeros((1, 32), dtype=np.uint32)
    program[0, [0, 14, 15, 16, 17, 18, 19]] = [1, 1, 1, 1, 2, 16, 32]
    perf = (
        "gemmini perf [e4m3_vpu]: 32 commands, 2 loops, 2 fences\n"
        "  mesh: 16 tiles, 256 busy cycles; VPU: 0 commands\n"
        "  load: 5120 bytes in 80 Gets; scales: 160 bytes\n"
    )
    assert check_accelerator_work(program, perf)["passed"]
    with pytest.raises(ValueError, match="accelerator work"):
        check_accelerator_work(program, perf.replace("16 tiles", "8 tiles"))
    with pytest.raises(ValueError, match="accelerator work"):
        check_accelerator_work(program, perf.replace("5120 bytes", "2560 bytes"))
