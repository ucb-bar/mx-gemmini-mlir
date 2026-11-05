"""Real captured-model and adversarial tests for npu model-plan admission."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import unittest
from pathlib import Path

from mx_gemmini_support.model_quantization_admission import validate_model_quantization_plan
from mx_gemmini_support.target_profile import load_profile, profile_sha256


ROOT = Path(__file__).resolve().parents[1]
CAPTURE = ROOT / "docs/evidence/model2mlir_small_llama_precision_e9ded36_266c593"
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


def _plan(source: str, profile: dict, worklist: dict) -> dict:
    """A separately constructed 19-site npu v2 fixture from the captured source."""
    mode = next(row for row in profile["legal_compute"] if row["pe_mode"] == 8)
    requested = {
        "placement": "mesh", "activation_format": "fp8_e4m3",
        "weight_format": "fp8_e4m3", "activation_projection": "direct",
        "weight_projection": "direct", "activation_mode": "dynamic_per_call",
        "weight_source": "quantize_float", "output_format": "bf16",
    }
    sites = []
    for collection, shape_key in (("rank2_matmuls", "shape_mnk"),
                                  ("rank3_batch_matmuls", "shape_bmnk")):
        for site in worklist[collection]:
            shape = site[shape_key]
            m, n, k = shape[-3:]
            padded = [*shape[:-3], (m + 15) // 16 * 16,
                      (n + 15) // 16 * 16, (k + 31) // 32 * 32]
            sites.append({
                "region_id": site["region_id"], "fqn": site["fqn"],
                "source_op": site["source_op"], shape_key: shape,
                "requested": copy.deepcopy(requested), "explicit_override": False,
                "status": "structural_candidate_not_executable",
                "hardware_mode": copy.deepcopy(mode),
                "operand_encoders": {"activation": "fp8_e4m3", "weight": "fp8_e4m3"},
                "padded_" + shape_key: padded,
                "padding_required": padded[-3:] != shape[-3:],
            })
    return {
        "schema": "npu_exploration.model_quantization_plan.v2",
        "status": "planning_only_no_executable", "run_id": "a" * 16,
        "run_name": "test_fp8", "run_recipe_sha256": "b" * 64,
        "profile_name": profile["name"], "profile_sha256": profile_sha256(profile),
        "profile_source": profile["source"],
        "source_mlir_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "quantizer": {"kind": "external_wire", "provider": "custom.user.rule",
                      "implementation_sha256": "c" * 64,
                      "settings": {"user_selected_rule": "opaque_to_structural_admission"},
                      "activation_runtime": "host_prologue"},
        "sites": sites,
    }


class ModelQuantizationAdmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from mx_gemmini_support.merlin_inventory import build_mx_worklist_from_merlin
            cls.source = gzip.decompress((CAPTURE / "small_llama.model2mlir.mlir.gz").read_bytes()).decode()
            cls.profile = load_profile(PROFILE)
            cls.worklist = build_mx_worklist_from_merlin(cls.source, cls.profile)
        except (ImportError, ModuleNotFoundError) as error:
            raise unittest.SkipTest(f"Merlin model inventory is unavailable: {error}") from error

    def fixture(self):
        return _plan(self.source, copy.deepcopy(self.profile), self.worklist)

    def admit(self, plan, *, source=None, profile=None):
        return validate_model_quantization_plan(
            plan, self.source if source is None else source,
            self.profile if profile is None else profile)

    def test_real_nineteen_site_capture_and_user_selected_quantizer(self):
        receipt = self.admit(self.fixture())
        self.assertEqual(receipt["status"], "structural_admission_only_no_executable")
        self.assertEqual(receipt["site_count"], 19)
        self.assertEqual(receipt["site_status_counts"]["structural_candidate_not_executable"], 19)
        self.assertEqual(receipt["source_mlir_sha256"], self.worklist["source_mlir_sha256"])
        self.assertIs(receipt["rtl_source_checked_against_checkout"], False)
        self.assertEqual(receipt["run_recipe_digest_status"], "declared_not_recomputed")

    def test_selected_recipe_digest_is_recomputed_when_supplied(self):
        plan = self.fixture()
        recipe = {"schema": "npu_exploration.model_run.v2", "quantizer": {
            "block_size": 32, "scale_encoding": "e8m0", "method": "custom"}}
        canonical = (json.dumps(recipe, sort_keys=True, separators=(",", ":")) + "\n").encode()
        plan["run_recipe_sha256"] = hashlib.sha256(canonical).hexdigest()
        receipt = validate_model_quantization_plan(
            plan, self.source, self.profile, run_recipe=recipe)
        self.assertEqual(receipt["run_recipe_digest_status"], "recomputed_from_selected_recipe")
        recipe["quantizer"]["method"] = "different"
        with self.assertRaisesRegex(ValueError, "recipe differs"):
            validate_model_quantization_plan(plan, self.source, self.profile,
                                             run_recipe=recipe)

    def test_source_profile_census_shape_and_provenance_drift_refused(self):
        good = self.fixture()
        with self.assertRaisesRegex(ValueError, "captured source or profile"):
            self.admit(good, source=self.source + "\n")
        for key, changed in (("profile_sha256", "0" * 64),
                             ("profile_name", "wrong"),
                             ("profile_source", {})):
            bad = copy.deepcopy(good)
            bad[key] = changed
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "captured source or profile"):
                self.admit(bad)
        bad = copy.deepcopy(good)
        bad["sites"].pop()
        with self.assertRaisesRegex(ValueError, "site census"):
            self.admit(bad)
        bad = copy.deepcopy(good)
        bad["sites"][0]["shape_mnk"][0] += 1
        with self.assertRaisesRegex(ValueError, "shape differs"):
            self.admit(bad)
        bad = copy.deepcopy(good)
        bad["sites"][0]["fqn"] = "wrong"
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            self.admit(bad)

    def test_mode_wire_and_padding_mutations_refused(self):
        good = self.fixture()
        mutations = (
            ("hardware_mode", "pe_mode", 0, "absent"),
            ("operand_encoders", "weight", "fp4_e2m1", "wire encoders"),
            ("padded_shape_mnk", 0, 999, "physical padding"),
        )
        for field, subkey, value, message in mutations:
            bad = copy.deepcopy(good)
            bad["sites"][0][field][subkey] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                self.admit(bad)
        bad = copy.deepcopy(good)
        bad["sites"][0]["requested"]["weight_format"] = "fp4_e2m1"
        with self.assertRaisesRegex(ValueError, "invalid compute mode"):
            self.admit(bad)
        custom = copy.deepcopy(good)
        custom["quantizer"]["settings"]["different_numeric_rule"] = {"anything": 64}
        self.assertEqual(self.admit(custom)["site_count"], 19)

    def test_explicit_host_stays_host_without_numeric_assumption(self):
        plan = self.fixture()
        site = plan["sites"][0]
        site["requested"]["placement"] = "host"
        site["status"] = "host_selected"
        site["explicit_override"] = True
        site["hardware_mode"] = None
        for key in ("operand_encoders", "padding_required", "padded_shape_mnk"):
            del site[key]
        receipt = self.admit(plan)
        self.assertEqual(receipt["site_status_counts"]["host_selected"], 1)
        self.assertEqual(receipt["site_status_counts"]["structural_candidate_not_executable"], 18)

    def test_lut_packing_is_checked_but_numeric_rule_is_opaque(self):
        profile = copy.deepcopy(self.profile)
        lut_mode = {"activation_format": "fp8_e4m3", "weight_format": "fp8_e4m3",
                    "activation_projection": "lut", "weight_projection": "lut", "pe_mode": 9}
        profile["legal_compute"].append(lut_mode)
        profile["resources"]["lut"] = True
        profile["resources"]["lut_config"] = {"address_bits": 4, "read_data_bits": 8}
        plan = _plan(self.source, profile, self.worklist)
        site = plan["sites"][0]
        site["requested"]["activation_projection"] = "lut"
        site["requested"]["weight_projection"] = "lut"
        site["hardware_mode"] = lut_mode
        site["operand_encoders"] = {"activation": "fp8_e4m3_quad", "weight": "fp8_e4m3_quad"}
        shape = site["shape_mnk"]
        site["padded_shape_mnk"] = [(shape[0] + 31) // 32 * 32,
                                     (shape[1] + 31) // 32 * 32,
                                     (shape[2] + 31) // 32 * 32]
        site["padding_required"] = site["padded_shape_mnk"] != shape
        receipt = self.admit(plan, profile=profile)
        self.assertEqual(receipt["site_count"], 19)
        profile["resources"]["lut"] = False
        plan["profile_sha256"] = profile_sha256(profile)
        with self.assertRaisesRegex(ValueError, "LUT hardware resources"):
            self.admit(plan, profile=profile)
        profile["resources"]["lut"] = True
        profile["resources"]["lut_config"]["read_data_bits"] = 6
        plan["profile_sha256"] = profile_sha256(profile)
        with self.assertRaisesRegex(ValueError, "LUT hardware resources"):
            self.admit(plan, profile=profile)


if __name__ == "__main__":
    unittest.main()
