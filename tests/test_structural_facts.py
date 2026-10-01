"""The selected tile grid remains structural until compute is corroborated."""

import json

import pytest

from mx_gemmini_support.structural_facts import attach_mesh_observation


def _sample():
    from pathlib import Path
    contract = (Path(__file__).parents[1] / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml").read_bytes()
    from mx_gemmini_support.contract import compile_contract
    selected = compile_contract(contract)
    fir = "f" * 64
    facts = {"schema_version": "1", "inputs": {"target": "mx_gemmini", "fir_sha256": fir},
             "source_consistency": {"status": "verified", "config": selected["rtl_config_class"],
                                    "production": {"firrtl_sha256": fir}},
             "facts": {"target": "mx_gemmini", "arrays": [],
                       "source": {"config": selected["rtl_config_class"], "fir_sha256": fir}}}
    receipt = {"artifacts": {"firrtl": {"sha256": fir}}}
    audit = {"schema": "mx_gemmini.build_receipt_audit.v1", "status": "observed_consistency",
             "gemmini_commit": selected["rtl_commit"], "mxgen_commit": selected["mxgen_commit"],
             "artifacts": {"firrtl": {"sha256": fir}},
             "elaborated_mesh": {"rows": 16, "columns": 16, "tiles": 256},
             "mesh_compute_hierarchy": {"tiles_with_hierarchy": 256, "fused_units_per_tile": 4,
                                        "path_manifest_sha256": "a" * 64,
                                        "port_wiring": {"tiles_with_witnesses": 256,
                                                        "witness_manifest_sha256": "b" * 64},
                                        "connected_arithmetic_verified": False},
             "selected_config_features": {
                 name: {"value": value, "origin": "standalone"}
                 for name, value in {
                     "has_nonlinear_activations": False, "has_normalizations": False,
                     "has_max_pool": False, "enable_lut": True, "lut_present": True,
                 }.items()
             },
             "phase0_admitted": False, "historical_build_provenance_verified": False}
    return facts, audit, json.dumps(receipt).encode(), contract


def test_structural_observation_does_not_create_compute_array():
    facts, audit, receipt, contract = _sample()
    output = attach_mesh_observation(json.dumps(facts).encode(), audit, receipt, contract)
    assert output["facts"]["arrays"] == []
    assert output["facts"]["structural_observations"][0]["instances"] == 256
    assert output["facts"]["structural_observations"][0]["multiplier_hierarchy"]["fused_units_per_tile"] == 4
    assert output["facts"]["structural_observations"][0]["multiplier_hierarchy"]["port_wiring"]["tiles_with_witnesses"] == 256
    assert output["facts"]["structural_observations"][0]["selected_config_features"]["has_normalizations"]["value"] is False
    assert output["facts"]["structural_observations"][0]["compute_engine_established"] is False
    assert output["structural_observation_provenance"]["phase0_admitted"] is False


def test_structural_observation_refuses_other_firrtl_or_existing_array():
    facts, audit, receipt, contract = _sample()
    facts["inputs"]["fir_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="generic RTL facts"):
        attach_mesh_observation(json.dumps(facts).encode(), audit, receipt, contract)
    facts["inputs"]["fir_sha256"] = "f" * 64
    facts["facts"]["arrays"] = [{"name": "mesh"}]
    with pytest.raises(ValueError, match="already contain"):
        attach_mesh_observation(json.dumps(facts).encode(), audit, receipt, contract)
    facts["facts"]["arrays"] = []
    del audit["mesh_compute_hierarchy"]
    with pytest.raises(ValueError, match="multiplier hierarchy"):
        attach_mesh_observation(json.dumps(facts).encode(), audit, receipt, contract)
    facts, audit, receipt, contract = _sample()
    del audit["selected_config_features"]["has_normalizations"]
    with pytest.raises(ValueError, match="feature switches"):
        attach_mesh_observation(json.dumps(facts).encode(), audit, receipt, contract)
