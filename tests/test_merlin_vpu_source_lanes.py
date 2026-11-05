"""Model VPU lane routing against captured, real model2MLIR source."""

from __future__ import annotations

import copy
import gzip
import hashlib
from pathlib import Path
import unittest

from mx_gemmini_support.merlin_vpu_reduce import selected_row_reduce_sites
from mx_gemmini_support.merlin_vpu_softmax import selected_softmax_sites
from mx_gemmini_support.target_profile import load_profile
from mx_gemmini_support.vpu_row_reduce import emit_vpu_row_reduce
from mx_gemmini_support.vpu_row_softmax import emit_vpu_row_softmax


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/full_model_compile_preflight_a042643_748b984/tinyllama.mlir.gz"
PROFILES = ROOT / "profiles/gemmini-mx-cleanup-266c593"


class ModelVpuSourceLanesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = gzip.decompress(ARCHIVE.read_bytes()).decode()
        cls.vpu = load_profile(PROFILES / "MxE4M3Fp4VpuGemminiRocketConfig.json")
        cls.no_vpu = load_profile(PROFILES / "MxGemminiRocketConfig.json")

    def test_source_sites_carry_legacy_lane_layout_into_emitters(self) -> None:
        softmax = selected_softmax_sites(self.source, self.vpu)
        reductions = selected_row_reduce_sites(self.source, self.vpu)
        self.assertEqual(len(softmax), 2)
        self.assertEqual(len(reductions), 5)
        self.assertEqual({site["vpu_lanes"] for site in softmax + reductions}, {8})
        self.assertEqual({site["operation"] for site in reductions}, {"rsum"})

        softmax_site = softmax[0]
        explicit = emit_vpu_row_softmax(
            self.vpu, logical_rows=softmax_site["logical_rows"],
            row_width=softmax_site["row_width"], symbol=softmax_site["kernel_symbol"],
            lanes=softmax_site["vpu_lanes"])
        legacy = emit_vpu_row_softmax(
            self.vpu, logical_rows=softmax_site["logical_rows"],
            row_width=softmax_site["row_width"], symbol=softmax_site["kernel_symbol"])
        self.assertEqual(hashlib.sha256(explicit.source.encode()).digest(),
                         hashlib.sha256(legacy.source.encode()).digest())

        reduce_site = reductions[0]
        explicit = emit_vpu_row_reduce(
            self.vpu, logical_rows=reduce_site["logical_rows"],
            row_width=reduce_site["row_width"], operation=reduce_site["operation"],
            symbol=reduce_site["kernel_symbol"], lanes=reduce_site["vpu_lanes"])
        legacy = emit_vpu_row_reduce(
            self.vpu, logical_rows=reduce_site["logical_rows"],
            row_width=reduce_site["row_width"], operation=reduce_site["operation"],
            symbol=reduce_site["kernel_symbol"])
        self.assertEqual(hashlib.sha256(explicit.source.encode()).digest(),
                         hashlib.sha256(legacy.source.encode()).digest())

    def test_non_dim16_model_profiles_need_matching_source_lane_binding(self) -> None:
        for columns in (4, 8, 32, 6):
            with self.subTest(columns=columns):
                profile = copy.deepcopy(self.vpu)
                profile["geometry"]["mesh_columns"] = columns
                for selector in (selected_softmax_sites, selected_row_reduce_sites):
                    with self.assertRaisesRegex(ValueError,
                                                "non-DIM16 model VPU routing needs source-verified lanes"):
                        selector(self.source, profile)

    def test_no_vpu_has_no_model_sites(self) -> None:
        self.assertEqual(selected_softmax_sites(self.source, self.no_vpu), [])
        self.assertEqual(selected_row_reduce_sites(self.source, self.no_vpu), [])


if __name__ == "__main__":
    unittest.main()
