"""Structural softmax VPU row layout checks for explicit lane configurations."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mx_gemmini_support.command_ir import Command
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support import vpu_row_softmax


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
LEGACY_BF16_SHA256 = "99015ca009e3690e7842b8518805fd4ac105c820f805fb42f987a3e37ec3a5e7"
LEGACY_HYBRID_SHA256 = "aaffba6845490bf1a570d235f84bf09c2b3c21754b8c348fdd6b2eb5e23b05e3"


class VpuSoftmaxLaneLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.legacy = load_profile(PROFILE)

    def selected(self, row_bytes: int) -> dict:
        profile = copy.deepcopy(self.legacy)
        profile["geometry"]["mesh_rows"] = row_bytes
        profile["geometry"]["mesh_columns"] = row_bytes
        return profile

    def traced(self, profile: dict, *, logical_rows: int, row_width: int,
               lanes: int, implementation: str = "bf16"):
        commands = []
        original = vpu_row_softmax.emit_c

        def record(items, **kwargs):
            commands.extend(items)
            return original(items, **kwargs)

        with patch.object(vpu_row_softmax, "emit_c", side_effect=record):
            kernel = vpu_row_softmax.emit_vpu_row_softmax(
                profile, logical_rows=logical_rows, row_width=row_width,
                symbol="softmax_lane_probe", implementation=implementation,
                lanes=lanes)
        return kernel, [cmd for cmd in commands if isinstance(cmd, Command)]

    def test_existing_dim16_sources_remain_byte_identical(self) -> None:
        for implementation, expected in (("bf16", LEGACY_BF16_SHA256),
                                         ("f32_reduce", LEGACY_HYBRID_SHA256)):
            with self.subTest(implementation=implementation):
                kernel = vpu_row_softmax.emit_vpu_row_softmax(
                    self.legacy, logical_rows=16, row_width=113,
                    symbol="softmax_probe", implementation=implementation)
                self.assertEqual(kernel.lanes, 8)
                self.assertEqual(hashlib.sha256(kernel.source.encode()).hexdigest(), expected)

    def test_dim4_two_lanes_use_four_byte_dma_and_disjoint_windows(self) -> None:
        kernel, commands = self.traced(self.selected(4), logical_rows=5,
                                       row_width=9, lanes=2)
        self.assertEqual((kernel.lanes, kernel.padded_width, kernel.rows_per_chunk),
                         (2, 10, 5))
        self.assertIn("static uint16_t softmax_lane_probe_score[50]", kernel.source)
        loads = [cmd for cmd in commands if cmd.funct == 2]
        stores = [cmd for cmd in commands if cmd.funct == 3]
        self.assertEqual([cmd.rs1.byte_offset for cmd in loads], [0, 64])
        self.assertEqual([cmd.rs1.byte_offset for cmd in stores], [0, 64])
        self.assertTrue(all(((cmd.rs2.immediate >> 32) & 0xffff) == 4
                            for cmd in loads + stores))
        vpu = [cmd for cmd in commands if cmd.funct == 33]
        self.assertEqual(len(vpu), 6)
        # 5 rows * ceil(9/2) = 25 source rows. The five distinct windows
        # are score [0,25), max [25,30), exp [30,55), sum [55,60),
        # and final result [60,85).
        self.assertEqual([(cmd.rs1.immediate >> 32) & 0xffff for cmd in vpu],
                         [25, 30, 30, 55, 55, 60])

    def test_dim8_four_lanes_use_fifteen_bit_vpu_addresses(self) -> None:
        kernel, commands = self.traced(self.selected(8), logical_rows=5,
                                       row_width=10, lanes=4)
        self.assertEqual((kernel.lanes, kernel.padded_width), (4, 12))
        self.assertIn("static uint16_t softmax_lane_probe_score[60]", kernel.source)
        vpu = [cmd for cmd in commands if cmd.funct == 33]
        # DIM8 scratchpad has 32768 rows, so address fields are 15 bits.
        self.assertEqual([(cmd.rs1.immediate >> 30) & 0x7fff for cmd in vpu],
                         [15, 20, 20, 35, 35, 40])

    def test_dim32_sixteen_lanes_and_hybrid_exp_layout(self) -> None:
        kernel, commands = self.traced(self.selected(32), logical_rows=3,
                                       row_width=33, lanes=16)
        self.assertEqual((kernel.lanes, kernel.padded_width), (16, 48))
        self.assertIn("softmax_lane_probe_result[row * 48 + col]", kernel.source)
        self.assertEqual(((next(cmd for cmd in commands if cmd.funct == 33).rs1.immediate
                           >> 28) & 0x3fff), 9)
        hybrid, hybrid_commands = self.traced(self.selected(8), logical_rows=5,
                                               row_width=10, lanes=4,
                                               implementation="f32_reduce")
        self.assertEqual((hybrid.lanes, hybrid.padded_width), (4, 12))
        exp = [cmd for cmd in hybrid_commands if cmd.funct == 33]
        self.assertEqual(len(exp), 1)
        self.assertEqual((exp[0].rs1.immediate >> 30) & 0x7fff, 15)
        self.assertTrue(all(((cmd.rs2.immediate >> 32) & 0xffff) == 8
                            for cmd in hybrid_commands if cmd.funct in {2, 3}))

    def test_refusals_name_missing_lanes_fixed_layouts_and_capacity(self) -> None:
        profile = self.selected(8)
        kwargs = dict(logical_rows=5, row_width=10, symbol="softmax_bad")
        with self.assertRaisesRegex(ValueError, "explicit source-derived"):
            vpu_row_softmax.emit_vpu_row_softmax(profile, **kwargs)
        with self.assertRaisesRegex(ValueError, "must match"):
            vpu_row_softmax.emit_vpu_row_softmax(profile, lanes=8, **kwargs)
        with self.assertRaisesRegex(ValueError, "DIM4, DIM8, DIM16, or DIM32"):
            vpu_row_softmax.emit_vpu_row_softmax(self.selected(6), lanes=3, **kwargs)
        with self.assertRaisesRegex(ValueError, "DIM16-only"):
            vpu_row_softmax.emit_vpu_row_softmax(
                profile, lanes=4, implementation="f32_striped16", **kwargs)
        with self.assertRaisesRegex(ValueError, "DIM16-only"):
            vpu_row_softmax.emit_vpu_row_softmax(
                profile, lanes=4, implementation="f32_reduce",
                packed_matrix_rows=16, **kwargs)
        tiny = self.selected(32)
        tiny["resources"]["scratchpad_bytes"] = 1024
        with self.assertRaisesRegex(ValueError, "windows exceed scratchpad"):
            vpu_row_softmax.emit_vpu_row_softmax(
                tiny, lanes=16, logical_rows=3, row_width=33,
                symbol="softmax_tiny")

    def test_non_dim16_generated_c_parses_for_full_and_hybrid_modes(self) -> None:
        temp_root = ROOT.parent / ".tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as scratch:
            for row_bytes, lanes, implementation in ((4, 2, "bf16"),
                                                      (8, 4, "f32_reduce"),
                                                      (32, 16, "bf16")):
                with self.subTest(row_bytes=row_bytes, implementation=implementation):
                    kernel = vpu_row_softmax.emit_vpu_row_softmax(
                        self.selected(row_bytes), logical_rows=3, row_width=33,
                        symbol="softmax_syntax", implementation=implementation,
                        lanes=lanes)
                    source = Path(scratch) / f"softmax_{row_bytes}_{implementation}.c"
                    source.write_text(kernel.source)
                    subprocess.run(["cc", "-std=c11", "-fsyntax-only", str(source)],
                                   check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
