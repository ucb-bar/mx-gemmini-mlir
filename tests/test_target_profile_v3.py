"""Current RTL geometry and versioned profile admission checks."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from mx_gemmini_support.target_profile import (
    SCHEMA_V3, _capacity, _hardware_options, _int_field, _memory_rows,
    _resolve_array_configs, _revision, _SCALA_FILES_V3, export_profiles_v3,
    load_profile, model_vpu_lanes, source_vpu_lanes,
)
from mx_gemmini_support.merlin_profile_contract import contract_for_profile


ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT.parent / "gemmini"
LEGACY = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3Fp4VpuGemminiRocketConfig.json"


def _v3_fixture() -> dict:
    profile = json.loads(LEGACY.read_text())
    profile["schema"] = SCHEMA_V3
    resources = profile["resources"]
    dim = profile["geometry"]["mesh_columns"] * profile["geometry"]["tile_columns"]
    resources["scratchpad_row_bits"] = dim * 8
    resources["accumulator_row_bits"] = dim * 64
    resources["scratchpad_bank_rows"] = (resources["scratchpad_bytes"] * 8 //
                                          (resources["scratchpad_banks"] * dim * 8))
    resources["accumulator_bank_rows"] = (resources["accumulator_bytes"] * 8 //
                                           (resources["accumulator_banks"] * dim * 64))
    resources["accumulator_rows"] = (resources["accumulator_banks"] *
                                      resources["accumulator_bank_rows"])
    resources["vpu_config"]["lanes"] = 8
    profile["source"]["sha256"]["src/main/scala/gemmini/Scratchpad.scala"] = "0" * 64
    assert set(profile["source"]["sha256"]) == set(_SCALA_FILES_V3)
    return profile


class _JsonPath:
    def __init__(self, value: dict):
        self.value = value

    def read_text(self) -> str:
        return json.dumps(self.value)


class TargetProfileV3Test(unittest.TestCase):
    def test_current_named_config_rows_lanes_and_requantizer(self):
        if not RTL.joinpath("src/main/scala/gemmini/ConfigsFP.scala").is_file():
            self.skipTest("selected Gemmini RTL source is unavailable")
        src = RTL / "src/main/scala/gemmini"
        configs = _resolve_array_configs((src / "ConfigsFP.scala").read_text())
        formula = (src / "GemminiConfigs.scala").read_text()
        fragments = (src / "MxConfigFragments.scala").read_text()
        vpu_source = (src / "vpu/Vpu.scala").read_text()
        cases = (
            ("dim4Fp4VpuMxFPConfig", 4, 2, 8192, 2048),
            ("dim8Fp4VpuMxFPConfig", 8, 4, 8192, 1024),
            ("dim32E4M3VpuMxFPConfig", 32, 16, 2048, 256),
        )
        for name, dim, lanes, bank_rows, acc_rows in cases:
            with self.subTest(name=name):
                values = configs[name]
                geometry = {"mesh_columns": _int_field(values, "meshColumns"),
                            "tile_columns": _int_field(values, "tileColumns")}
                resources = {
                    "scratchpad_bytes": _capacity(values, "sp_capacity"),
                    "scratchpad_banks": _int_field(values, "sp_banks"),
                    "accumulator_bytes": _capacity(values, "acc_capacity"),
                    "accumulator_banks": _int_field(values, "acc_banks"),
                }
                rows = _memory_rows(values, geometry, resources, formula)
                self.assertEqual(geometry["mesh_columns"], dim)
                self.assertEqual(rows["scratchpad_bank_rows"], bank_rows)
                self.assertEqual(rows["accumulator_rows"], acc_rows)
                self.assertEqual(rows["scratchpad_row_bits"], lanes * 16)
                self.assertEqual(_hardware_options(configs, name, fragments,
                                  schema=SCHEMA_V3)["requantizer"]["pipeline_latency"], 2)
                source_profile = {
                    "schema": SCHEMA_V3, "gemmini_config": name,
                    "source": {"gemmini_revision": _revision(RTL), "sha256": {
                        relative: hashlib.sha256((RTL / relative).read_bytes()).hexdigest()
                        for relative in ("src/main/scala/gemmini/ConfigsFP.scala",
                                         "src/main/scala/gemmini/vpu/Vpu.scala")}},
                    "resources": {"vpu": True, "vpu_config": {
                        "units": 2, "exp_sub": True, "exp_sum": True, "lanes": lanes}},
                }
                self.assertEqual(source_vpu_lanes(source_profile, RTL), lanes)

    def test_v3_loader_rejects_missing_or_inconsistent_memory_facts(self):
        profile = _v3_fixture()
        selected = load_profile(_JsonPath(profile))
        self.assertEqual(model_vpu_lanes(selected), 8)
        self.assertEqual(contract_for_profile(selected)["mx_resources"]["vpu_config"]["lanes"], 8)
        changed = copy.deepcopy(profile)
        changed["resources"]["scratchpad_bank_rows"] += 1
        with self.assertRaisesRegex(ValueError, "memory rows disagree"):
            load_profile(_JsonPath(changed))
        changed = copy.deepcopy(profile)
        changed["resources"]["vpu_config"]["lanes"] = 4
        with self.assertRaisesRegex(ValueError, "VPU lanes disagree"):
            load_profile(_JsonPath(changed))
        changed = copy.deepcopy(profile)
        del changed["source"]["sha256"]["src/main/scala/gemmini/Scratchpad.scala"]
        with self.assertRaisesRegex(ValueError, "complete source file closure"):
            load_profile(_JsonPath(changed))

    def test_incomplete_mxgen_source_refuses_export(self):
        with self.assertRaisesRegex(ValueError, "source closure is incomplete"):
            export_profiles_v3(ROOT.parent / ".tmp")


if __name__ == "__main__":
    unittest.main()
