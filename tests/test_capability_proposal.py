"""A capability proposal keeps structural facts separate from authored intent."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from mx_gemmini_support.capability_proposal import derive_proposal


def _inputs():
    spec = (Path(__file__).parents[1] /
            "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml").read_bytes()
    fir_sha = "a" * 64
    facts = {
        "inputs": {"target": "mx_gemmini", "fir_sha256": fir_sha},
        "source_consistency": {"status": "verified", "config": "GemminiMxFPStandaloneConfig",
                               "production": {"firrtl_sha256": fir_sha}},
        "structural_observation_provenance": {
            "schema": "mx_gemmini.structural_observation.v1",
            "build_receipt_sha256": "b" * 64, "generic_facts_sha256": "c" * 64,
            "software_spec_sha256": hashlib.sha256(spec).hexdigest(), "phase0_admitted": False,
        },
        "facts": {"target": "mx_gemmini", "source": {"fir_sha256": fir_sha,
                                                         "config": "GemminiMxFPStandaloneConfig"}, "arrays": [],
                  "structural_observations": [{"kind": "mesh_tile_grid", "source": "selected_firrtl",
                                               "firrtl_sha256": fir_sha, "rows": 16, "cols": 16,
                                               "instances": 256, "compute_engine_established": False,
                                               "multiplier_hierarchy": {
                                                   "tiles_with_hierarchy": 256,
                                                   "fused_units_per_tile": 4,
                                                   "path_manifest_sha256": "d" * 64,
                                                   "port_wiring": {"tiles_with_witnesses": 256,
                                                                   "witness_manifest_sha256": "e" * 64},
                                                   "connected_arithmetic_verified": False,
                                               },
                                               "selected_config_features": {
                                                   name: {"value": value, "origin": "standalone"}
                                                   for name, value in {
                                                       "has_nonlinear_activations": False,
                                                       "has_normalizations": False,
                                                       "has_max_pool": False,
                                                       "enable_lut": True,
                                                       "lut_present": True,
                                                   }.items()
                                               }}]},
    }
    return facts, spec


def test_proposal_separates_rtl_geometry_and_authored_formats():
    facts, spec = _inputs()
    result = derive_proposal(json.dumps(facts).encode(), spec)
    assert result["rtl_geometry"]["tiles"] == 256
    assert result["rtl_builtin_features"]["has_nonlinear_activations"]["value"] is False
    assert [row["format"] for row in result["authored_accelerator_intent"]] == ["mxfp4", "mxfp6", "mxfp8"]
    assert result["authored_accelerator_intent"][0]["site_modes"]["functional_matmul"]["lhs"] == "dynamic"
    assert "normalization" in result["authored_host_families"]
    assert result["status"] == "candidate_not_admitted"
    assert result["merlin_target_contract_selectable"] is False


def test_proposal_refuses_mismatched_source_or_unsupported_claim():
    facts, spec = _inputs()
    changed = deepcopy(facts)
    changed["facts"]["arrays"] = [{"name": "mesh"}]
    with pytest.raises(ValueError, match="one selected source"):
        derive_proposal(json.dumps(changed).encode(), spec)
    changed = deepcopy(facts)
    changed["facts"]["structural_observations"][0]["instances"] = 255
    with pytest.raises(ValueError, match="uncorroborated mesh grid"):
        derive_proposal(json.dumps(changed).encode(), spec)
    changed = deepcopy(facts)
    del changed["facts"]["structural_observations"][0]["selected_config_features"]["has_max_pool"]
    with pytest.raises(ValueError, match="features are incomplete"):
        derive_proposal(json.dumps(changed).encode(), spec)
    changed = yaml.safe_load(spec)
    del changed["operations"]["contraction_mxfp6"]
    altered_spec = yaml.safe_dump(changed).encode()
    facts["structural_observation_provenance"]["software_spec_sha256"] = hashlib.sha256(altered_spec).hexdigest()
    with pytest.raises(ValueError, match="mxfp6 operation must be a mapping"):
        derive_proposal(json.dumps(facts).encode(), altered_spec)
    changed = yaml.safe_load(spec)
    del changed["operations"]["contraction_mxfp4"]["aliasing"]
    altered_spec = yaml.safe_dump(changed).encode()
    facts["structural_observation_provenance"]["software_spec_sha256"] = hashlib.sha256(altered_spec).hexdigest()
    with pytest.raises(ValueError, match="legality declaration is incomplete"):
        derive_proposal(json.dumps(facts).encode(), altered_spec)
