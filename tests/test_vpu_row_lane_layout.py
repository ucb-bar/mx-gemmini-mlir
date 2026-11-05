"""Source-row layout checks for the parameterized VPU reduction issuer."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import unittest

from mx_gemmini_support.command_ir import _mx_rows, vpu_command
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.vpu_row_reduce import emit_vpu_row_reduce


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
LEGACY_SHA256 = "46264af19e08ed5da482e5d4e8d18a72f8e1af369fbe62e933afc97193309cbe"


class VpuRowLaneLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.legacy = load_profile(PROFILE)

    def selected(self, row_bytes: int) -> dict:
        profile = copy.deepcopy(self.legacy)
        profile["geometry"]["mesh_rows"] = row_bytes
        profile["geometry"]["mesh_columns"] = row_bytes
        return profile

    def test_existing_dim16_generated_c_is_byte_identical(self) -> None:
        kernel = emit_vpu_row_reduce(
            self.legacy, logical_rows=16, row_width=768,
            operation="rsum", symbol="mx_reduce_probe_kernel")
        self.assertEqual(kernel.lanes, 8)
        self.assertEqual(hashlib.sha256(kernel.source.encode()).hexdigest(), LEGACY_SHA256)

    def test_dim8_uses_four_bf16_values_per_spad_row_and_15_bit_addresses(self) -> None:
        profile = self.selected(8)
        kernel = emit_vpu_row_reduce(
            profile, logical_rows=7, row_width=10,
            operation="rmax", symbol="reduce4", lanes=4)
        self.assertEqual((kernel.lanes, kernel.padded_width, kernel.rows_per_chunk),
                         (4, 12, 7))
        self.assertIn("static uint16_t reduce4_input[84]", kernel.source)
        self.assertIn("static uint16_t reduce4_result[28]", kernel.source)
        self.assertIn("reduce4_result[row*4]", kernel.source)
        self.assertIn("reduce4_input[row*12+col]", kernel.source)
        self.assertIn("(uintptr_t)input + UINT64_C(128)", kernel.source)
        cmd = vpu_command(profile, kind="rsum", src1_row=1, src2_row=0,
                          dst_row=0x2000, rows=3, reduction_length=3)
        self.assertEqual(cmd.rs1.immediate,
                         1 | (0x2000 << 30) | (3 << 45))
        with self.assertRaisesRegex(ValueError, "only been derived for DIM16"):
            _mx_rows(profile)  # SPAD_REQUANT is still DIM16-only.

    def test_dim4_uses_two_lanes_and_four_byte_transfers(self) -> None:
        profile = self.selected(4)
        kernel = emit_vpu_row_reduce(
            profile, logical_rows=5, row_width=5,
            operation="rsum", symbol="reduce2", lanes=2)
        self.assertEqual((kernel.lanes, kernel.padded_width), (2, 6))
        self.assertIn("static uint16_t reduce2_input[30]", kernel.source)
        self.assertIn("static uint16_t reduce2_result[10]", kernel.source)
        self.assertIn("reduce2_result[row*2]", kernel.source)
        self.assertIn("UINT64_C(0x0005000400002000)", kernel.source)

    def test_dim32_uses_sixteen_lanes_and_result_window_inside_spad(self) -> None:
        profile = self.selected(32)
        kernel = emit_vpu_row_reduce(
            profile, logical_rows=3, row_width=33,
            operation="rsum", symbol="reduce16", lanes=16)
        self.assertEqual((kernel.lanes, kernel.padded_width), (16, 48))
        self.assertIn("reduce16_result[row*16]", kernel.source)
        self.assertIn("static uint16_t reduce16_result[48]", kernel.source)
        self.assertIn("UINT64_C(0x0000250000000000)", kernel.source)
        self.assertIn("UINT64_C(0x0003002000001000)", kernel.source)

    def test_missing_or_inconsistent_source_lane_selection_refuses(self) -> None:
        profile = self.selected(8)
        kwargs = dict(logical_rows=7, row_width=10, operation="rsum",
                      symbol="reduce_bad")
        with self.assertRaisesRegex(ValueError, "explicit source-derived"):
            emit_vpu_row_reduce(profile, **kwargs)
        with self.assertRaisesRegex(ValueError, "must match"):
            emit_vpu_row_reduce(profile, lanes=8, **kwargs)
        with self.assertRaisesRegex(ValueError, "DIM4, DIM8, DIM16, or DIM32"):
            emit_vpu_row_reduce(self.selected(6), lanes=3, **kwargs)
        with self.assertRaisesRegex(ValueError, "static extents"):
            emit_vpu_row_reduce(profile, lanes=4, **{**kwargs, "row_width": 4093})


if __name__ == "__main__":
    unittest.main()
