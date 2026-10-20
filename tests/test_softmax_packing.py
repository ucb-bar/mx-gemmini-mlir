"""Streamed probabilities preserve hybrid arithmetic and per-matrix scale layout."""

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_hybrid_softmax import EMULATOR

from mx_gemmini_support import vpu_row_softmax
from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.runtime_group_preparation import emit_runtime_group_preparation
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
)


@pytest.mark.parametrize("batch,m,k", [(2, 16, 32), (5, 32, 160), (1, 160, 160), (32, 160, 160)])
def test_complete_packed_bytes_across_vector_tile_and_matrix_boundaries(
    tmp_path, monkeypatch, batch, m, k
):
    def trace(sequence, *, symbol, buffers, **unused):
        body = "\n".join(
            f"issue({c.funct}, {c.rs1.c_expr()}, {c.rs2.c_expr()});"
            for c in sequence
            if isinstance(c, Command)
        )
        return f"static void {symbol}(const void *score,const void *result) {{\n{body}\n}}\n"

    monkeypatch.setattr(vpu_row_softmax, "emit_c", trace)
    kernel = vpu_row_softmax.emit_vpu_row_softmax(
        load_profile(PROFILE),
        logical_rows=batch * m,
        row_width=k,
        symbol="packed",
        implementation="f32_reduce",
        packed_matrix_rows=m,
    )
    source = tmp_path / "packed.c"
    source.write_text(EMULATOR + kernel.source)
    (tmp_path / "runtime_group_prepare.h").write_text(emit_runtime_group_preparation())
    library = tmp_path / "packed.so"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-DMX_QUANT_GROUP_WORDS",
            "-DMX_QUANT_INLINE_GROUP",
            "-I",
            str(tmp_path),
            "-I",
            str(ROOT / "mx_gemmini_support"),
            str(source),
            str(ROOT / "mx_gemmini_support/runtime_quant.c"),
            "-o",
            str(library),
        ],
        check=True,
    )
    reference = tmp_path / "reference.so"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-DMX_QUANT_REFERENCE",
            str(ROOT / "mx_gemmini_support/runtime_quant.c"),
            "-o",
            str(reference),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(library))
    ref = ctypes.CDLL(str(reference))
    dll.packed.argtypes = [ctypes.c_void_p] * 2
    ref.mx_pack_activation_f32.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    values = (
        -np.random.default_rng(53).integers(0, 4, (batch * m, k)).astype(np.float32)
    )
    values[:, 0] = 0
    values[:, -1] = -np.inf
    exp_words = {0: 0x3F80, -1: 0x3EBC, -2: 0x3E0B, -3: 0x3D4C, -np.inf: 0}
    exponentials = (
        np.array([exp_words[float(x)] << 16 for x in values.flat], np.uint32)
        .view(np.float32)
        .reshape(values.shape)
    )
    probability = exponentials.copy()
    for row in range(batch * m):
        total = np.float32(0)
        for value in exponentials[row]:
            total = np.float32(total + value)
        probability[row] = exponentials[row] / total
    expected = np.empty(batch * m * k * 33 // 32, np.uint8)
    for b in range(batch):
        offset = b * m * k * 33 // 32
        assert (
            ref.mx_pack_activation_f32(
                probability[b * m :].ctypes.data,
                m,
                k,
                8,
                expected[offset:].ctypes.data,
                expected[offset + m * k :].ctypes.data,
            )
            == 0
        )
    for sentinel in (0xA5, 0x5A):
        output = np.full(expected.size + 128, sentinel, np.uint8)
        dll.packed(values.ctypes.data, output[64:].ctypes.data)
        assert np.array_equal(output[64:-64], expected)
        assert np.all(output[:64] == sentinel) and np.all(output[-64:] == sentinel)


def test_packed_softmax_refuses_unrepresentable_matrix_layout():
    for options in (
        {"implementation": "bf16", "packed_matrix_rows": 16},
        {"implementation": "f32_reduce", "packed_matrix_rows": 17},
        {"implementation": "f32_reduce", "packed_matrix_rows": True},
    ):
        with pytest.raises(ValueError):
            vpu_row_softmax.emit_vpu_row_softmax(
                load_profile(PROFILE),
                logical_rows=32,
                row_width=160,
                symbol="packed",
                **options,
            )
