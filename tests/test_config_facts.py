"""Selected standalone config facts must bind the candidate's hardware claims."""

from copy import deepcopy

import pytest

from mx_gemmini_support.config_facts import check_selected_config, selected_config_facts


SOURCE = """
val defaultMxFPConfig = GemminiArrayConfig[MxFloat, Float, Float](
  meshRows = 16, meshColumns = 16, sp_banks = 4, acc_banks = 1,
  scaleSize = 32, spad_read_delay = 1, tile_latency = 1,
  acc_latency = 2, mesh_output_delay = 0,
  use_mx_scaling = true, enable_lut = true,
  has_nonlinear_activations = false, has_max_pool = false,
  has_normalizations = false, ex_read_from_acc = true,
  ex_write_to_spad = true,
  sp_capacity = CapacityInKilobytes(256),
  acc_capacity = CapacityInKilobytes(64),
  scale_mem = Some(MemConfig(sizeInBytes = 16, numBanks = 8)),
)
val standaloneMxFPConfig = defaultMxFPConfig.copy(
  acc_banks = 2, // selected standalone override
  spad_read_delay = 4, tile_latency = 0,
  acc_latency = 3, mesh_output_delay = 1,
  ex_read_from_acc = false,
  lut = Some(GemminiLUTConfig()),
)
"""


def test_selected_facts_include_inherited_switches_and_overrides():
    facts = selected_config_facts(SOURCE)
    assert facts["meshRows"] == {"value": 16, "origin": "default"}
    assert facts["acc_banks"] == {"value": 2, "origin": "standalone"}
    assert facts["lut_present"] == {"value": True, "origin": "standalone"}
    assert facts["has_nonlinear_activations"]["value"] is False
    assert facts["has_normalizations"]["value"] is False
    assert facts["has_max_pool"]["value"] is False
    assert facts["scaleSize"]["value"] == 32


@pytest.mark.parametrize("change,message", [
    ("scaleSize = 32", "scaleSize = 16"),
    ("use_mx_scaling = true", "use_mx_scaling = false"),
    ("meshRows = 16", "meshRows = 32"),
    ("enable_lut = true", "enable_lut = false"),
    ("lut = Some(GemminiLUTConfig())", "lut = None"),
])
def test_selected_config_refuses_claim_drift(change, message):
    facts = selected_config_facts(SOURCE.replace(change, message))
    with pytest.raises(ValueError):
        check_selected_config({"operations": {}}, {"block_size": 32}, facts)


@pytest.mark.parametrize("name,family", [
    ("elementwise_map", "elementwise_map"),
    ("norm_op", "normalization"),
    ("pool_op", "pooling"),
])
def test_disabled_accelerator_families_are_refused(name, family):
    facts = selected_config_facts(SOURCE)
    spec = {"operations": {name: {"families": [family], "placement": "host"}}}
    check_selected_config(spec, {"block_size": 32}, facts)
    bad = deepcopy(spec)
    bad["operations"][name]["placement"] = "accelerator"
    with pytest.raises(ValueError, match="selected RTL disables"):
        check_selected_config(bad, {"block_size": 32}, facts)


def test_nonliteral_selected_switch_is_refused():
    with pytest.raises(ValueError, match="has_max_pool.*literal Boolean"):
        selected_config_facts(SOURCE.replace("has_max_pool = false", "has_max_pool = enabled"))
