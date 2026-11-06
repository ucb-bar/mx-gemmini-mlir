"""The selected coupled mesh/lane profiles share a typed binary VPU issuer."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from mx_gemmini_support.vpu_add_kernel import (
    elementwise_issue_commands, emit_vpu_elementwise,
)


def _profile(dim: int) -> dict:
    return {
        "schema": "mx_gemmini.target_profile.v3",
        "transport": "rocket_rocc",
        "geometry": {"mesh_rows": dim, "mesh_columns": dim},
        "resources": {
            "scratchpad_bytes": 131072 if dim == 4 else 262144,
            "scratchpad_row_bits": dim * 8,
            "scratchpad_banks": 4,
            "scratchpad_bank_rows": (131072 if dim == 4 else 262144) // (4 * dim),
            "vpu": True,
            "vpu_config": {"lanes": dim // 2},
        },
    }


class CoupledVPUElementwiseTest(unittest.TestCase):
    def test_selected_multiply_rows_and_chunked_dim4(self) -> None:
        for dim in (4, 8, 16, 32):
            with self.subTest(dim=dim):
                profile = _profile(dim)
                lanes = dim // 2
                chunk = min(2752, 1024 * lanes)
                commands = elementwise_issue_commands(
                    profile, elements=chunk, operation="mul")
                multiply = [cmd for cmd in commands if getattr(cmd, "funct", None) == 33]
                self.assertEqual(len(multiply), 1)
                width = max(14, (profile["resources"]["scratchpad_bytes"] // dim - 1).bit_length())
                mask = (1 << width) - 1
                word = multiply[0].rs1.immediate
                self.assertEqual(word & mask, 0)
                self.assertEqual((word >> width) & mask,
                                 0x400 if dim == 32 else 0x1000)
                self.assertEqual((word >> (2 * width)) & mask,
                                 0x800 if dim == 32 else 0x2000)
                self.assertEqual((word >> (3 * width)) & 0xffff,
                                 (chunk + lanes - 1) // lanes)
                with patch("mx_gemmini_support.vpu_add_kernel._broadcast_index",
                           return_value="base + index"):
                    emitted = emit_vpu_elementwise(
                        profile, result_shape=(1, 8, 344), rhs_shape=(1, 8, 344),
                        dtype="f32", operation="mul", symbol="slice_vpu_mul12")
                self.assertEqual(emitted.elements_per_chunk, chunk)
                if dim == 4:
                    self.assertIn("case 704:", emitted.source)

    def test_wrong_lane_or_undersized_window_refuses(self) -> None:
        profile = _profile(32)
        profile["resources"]["vpu_config"]["lanes"] = 8
        with patch("mx_gemmini_support.vpu_add_kernel._broadcast_index",
                   return_value="base + index"):
            with self.assertRaisesRegex(ValueError, "coupled mesh/lane"):
                emit_vpu_elementwise(
                    profile, result_shape=(1, 8), rhs_shape=(1, 8),
                    dtype="f32", operation="mul", symbol="wrong")
        profile = _profile(32)
        profile["resources"]["scratchpad_bytes"] = 2048 * 32
        profile["resources"]["scratchpad_bank_rows"] = 2048 // 4
        with self.assertRaisesRegex(ValueError, "scratchpad capacity"):
            elementwise_issue_commands(profile, elements=2752, operation="mul")


if __name__ == "__main__":
    unittest.main()
