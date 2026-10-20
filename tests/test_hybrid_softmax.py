"""Exercise hybrid softmax tiling and FP32 normalization through its DMA stream."""

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mx_gemmini_support import vpu_row_softmax
from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.target_profile import load_profile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
)

# Deliberately simple command emulator: known exp words for integer shifts.
# Numerical qualification of the hardware exponential is a separate Spike run.
EMULATOR = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static uint16_t scratch[16384 * 8];
static unsigned exp_calls;
unsigned observed_exp_calls(void) { return exp_calls; }
static void issue(unsigned op, uint64_t a, uint64_t b) {
  if (op == 0 || op == 7) return;
  if (op == 2 || op == 3) {
    unsigned start=(unsigned)b, rows=b>>48, bytes=(b>>32)&0xffff;
    if (bytes!=16 || start+rows>16384) abort();
    if (op==2) memcpy(scratch+start*8,(const void *)(uintptr_t)a,rows*16);
    else memcpy((void *)(uintptr_t)a,scratch+start*8,rows*16);
    return;
  }
  if (op != 33 || (b & 31) != 5) abort();
  unsigned src=a&0x3fff, dst=(a>>28)&0x3fff, rows=(a>>42)&0xffff;
  if (src+rows>16384 || dst+rows>16384) abort();
  ++exp_calls;
  for (unsigned i=0;i<rows*8;i++) {
    uint16_t x=scratch[src*8+i], y;
    switch(x) {
      case 0: case 0x8000: y=0x3f80;break;
      case 0xbf80: y=0x3ebc;break;
      case 0xc000: y=0x3e0b;break;
      case 0xc040: y=0x3d4c;break;
      case 0xff80: y=0;break;
      default: abort();
    }
    scratch[dst*8+i]=y;
  }
}
"""


@pytest.mark.parametrize("rows,width", [(3, 1), (7, 9), (345, 17), (55, 160)])
def test_batched_exp_and_tail_keep_rows_independent(tmp_path, monkeypatch, rows, width):
    commands = []

    def trace(sequence, *, symbol, buffers, **unused):
        commands.extend(sequence)
        body = "\n".join(
            f"issue({c.funct}, {c.rs1.c_expr()}, {c.rs2.c_expr()});"
            for c in sequence
            if isinstance(c, Command)
        )
        return f"static void {symbol}(const void *score, const void *result) {{\n{body}\n}}\n"

    monkeypatch.setattr(vpu_row_softmax, "emit_c", trace)
    generated = vpu_row_softmax.emit_vpu_row_softmax(
        load_profile(PROFILE),
        logical_rows=rows,
        row_width=width,
        symbol="hybrid",
        implementation="f32_reduce",
    )
    source = tmp_path / "hybrid.c"
    source.write_text(EMULATOR + generated.source)
    library = tmp_path / "hybrid.so"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-fno-fast-math",
            "-ffp-contract=off",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(library))
    dll.hybrid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    values = -np.random.default_rng(42).integers(0, 4, (rows, width)).astype(np.float32)
    values[:, 0] = 0
    if width > 1:
        values[:, -1] = -np.inf
    output = np.full(rows * width + 2, np.float32(123), dtype=np.float32)
    dll.hybrid(values.ctypes.data, output[1:].ctypes.data)
    words = {0: 0x3F80, -1: 0x3EBC, -2: 0x3E0B, -3: 0x3D4C, -np.inf: 0}
    exp = (
        np.array([words[float(x)] << 16 for x in values.flat], np.uint32)
        .view(np.float32)
        .reshape(values.shape)
    )
    expected = exp.copy()
    for row in range(rows):
        total = np.float32(0)
        for value in exp[row]:
            total = np.float32(total + value)
        expected[row] = exp[row] / total
    assert np.array_equal(
        output[1:-1].view(np.uint32), expected.ravel().view(np.uint32)
    )
    assert output[0] == output[-1] == 123
    assert (
        dll.observed_exp_calls()
        == (rows + generated.rows_per_chunk - 1) // generated.rows_per_chunk
    )
    assert generated.vpu_steps == ("exp",)
    assert all(
        c.funct != 33 or (c.rs2.immediate & 31) == 5
        for c in commands
        if isinstance(c, Command)
    )
