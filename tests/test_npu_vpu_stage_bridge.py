"""A captured npu VPU decision must reach every selected MX rewrite site."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from mx_gemmini_support.execution_policy import eligible_sites
from mx_gemmini_support.merlin_inventory import build_mx_worklist_from_merlin
from mx_gemmini_support.npu_vpu_stage_bridge import (
    bind_rewrite_to_stage_policy, rewrite_selected_vpu_sites)
from mx_gemmini_support.execution_policy import validate_policy
from mx_gemmini_support.target_profile import load_profile
from tests.test_model_quantization_admission import _plan


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"
SOURCE = ROOT / "docs/evidence/model2mlir_small_llama_precision_e9ded36_266c593/small_llama.model2mlir.mlir.gz"


class NpuVpuStageBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = gzip.decompress(SOURCE.read_bytes()).decode()
        cls.profile = load_profile(PROFILE)
        cls.worklist = build_mx_worklist_from_merlin(cls.source, cls.profile)
        _, cls.eligible, cls.singletons = eligible_sites(cls.source, cls.profile)
        if cls.singletons:
            raise AssertionError("the pinned small Llama fixture unexpectedly has singleton adds")

    def fixture(self) -> tuple[dict, dict, bytes]:
        plan = _plan(self.source, self.profile, self.worklist)
        policy = {"default": "vpu", "overrides": {}}
        plan["vpu"] = policy
        plan["vpu_sites"] = [
            {"operation_class": name, "region_id": region, "placement": "vpu",
             "explicit_override": False,
             "status": "structural_candidate_not_executable", "host_reason": None}
            for name in sorted(self.eligible) for region in sorted(self.eligible[name])
        ]
        run = {"vpu": policy}
        raw = (json.dumps(run, sort_keys=True, separators=(",", ":")) + "\n").encode()
        plan["run_recipe_sha256"] = hashlib.sha256(raw).hexdigest()
        plan_bytes = (json.dumps(plan, sort_keys=True) + "\n").encode()
        return plan, run, plan_bytes

    def test_all_eighty_three_selected_sites_reach_existing_rewriters(self) -> None:
        plan, run, plan_bytes = self.fixture()
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as scratch:
            receipt = rewrite_selected_vpu_sites(
                plan, plan_bytes, self.source, self.profile, run,
                Path(scratch) / "rewrite")
            self.assertEqual(sum(map(len, receipt["selected_vpu_regions"].values())), 83)
            self.assertEqual(len(receipt["selected_vpu_regions"]["softmax"]), 2)
            self.assertEqual({name: len(row["regions"]) for name, row in
                              receipt["rewrite_receipts"].items()},
                             {name: len(sites) for name, sites in self.eligible.items()})
            output = Path(scratch) / "rewrite" / receipt["rewritten_mlir"]
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(),
                             receipt["rewritten_mlir_sha256"])
            self.assertIn("no_model_elf_or_spike_parity", receipt["status"])

    def test_changed_or_missing_source_site_refused_before_rewrite(self) -> None:
        plan, run, plan_bytes = self.fixture()
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as scratch:
            out = Path(scratch) / "changed"
            with self.assertRaisesRegex(ValueError, "captured source or profile"):
                rewrite_selected_vpu_sites(plan, plan_bytes, self.source + "\n",
                                           self.profile, run, out)
            self.assertFalse(out.exists())
            changed = copy.deepcopy(plan)
            changed["vpu_sites"].pop()
            with self.assertRaisesRegex(ValueError, "VPU site census"):
                rewrite_selected_vpu_sites(changed, json.dumps(changed).encode(), self.source,
                                           self.profile, run, out)
            self.assertFalse(out.exists())
            with self.assertRaisesRegex(ValueError, "plan bytes differ"):
                rewrite_selected_vpu_sites(plan, b"{}", self.source,
                                           self.profile, run, out)

    def test_rewrite_binds_existing_stage_policy_at_original_capture(self) -> None:
        plan, run, plan_bytes = self.fixture()
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as scratch:
            rewrite_dir = Path(scratch) / "rewrite"
            rewrite_selected_vpu_sites(plan, plan_bytes, self.source,
                                       self.profile, run, rewrite_dir)
            out = Path(scratch) / "policy"
            receipt = bind_rewrite_to_stage_policy(
                plan, plan_bytes, self.source, self.profile, run, rewrite_dir, out)
            policy = json.loads((out / "execution_policy.json").read_text())
            resolved = validate_policy(policy, self.source, self.profile)
            self.assertEqual(receipt["matrix_sites"], 19)
            self.assertEqual(sum(receipt["vpu_sites"].values()), 83)
            self.assertEqual(len(resolved["enabled_vpu_regions"]["softmax"]), 2)
            self.assertEqual(set(policy["matrix_formats"].values()), {"mxfp8"})
            self.assertIn("quantizer_binding", receipt)
            self.assertIn("original_capture", receipt["stage_source"])

            rewritten = rewrite_dir / json.loads((rewrite_dir / "receipt.json").read_text())[
                "rewritten_mlir"]
            rewritten.write_bytes(rewritten.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "rewritten MLIR differs"):
                bind_rewrite_to_stage_policy(plan, plan_bytes, self.source,
                                             self.profile, run, rewrite_dir,
                                             Path(scratch) / "changed")


if __name__ == "__main__":
    unittest.main()
