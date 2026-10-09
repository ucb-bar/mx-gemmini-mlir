"""A named contraction must be admitted by one exact source-bound profile."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir


PROFILE = Path(__file__).resolve().parents[1] / "profiles/gemmini-mx-cleanup-266c593/MxAllAsymGemminiRocketConfig.json"
PROFILE_DATA = load_profile(PROFILE)


def _ir(*, activation="fp8_e4m3", weight="fp4_e2m1", projection="lut",
        mode=10, profile_digest=None) -> str:
    digest = profile_digest or profile_sha256(PROFILE_DATA)
    contract, policy, manifest = "a" * 64, "b" * 64, "c" * 64
    attrs = (f'site_id = "s", activation_format = "{activation}", weight_format = "{weight}", '
             f'activation_projection = "{projection}", weight_projection = "direct", '
             f'pe_mode = {mode} : i32, profile_sha256 = "{digest}", '
             f'contract_sha256 = "{contract}", policy_sha256 = "{policy}", '
             f'manifest_sha256 = "{manifest}"')
    readout = (f'site_id = "s", profile_sha256 = "{digest}", '
               f'contract_sha256 = "{contract}", policy_sha256 = "{policy}", '
               f'manifest_sha256 = "{manifest}"')
    return f'''module attributes {{mx.contract_sha256 = "{contract}",
  mx.policy_sha256 = "{policy}", prov.quantization_manifest_sha256 = "{manifest}",
  mx.profile_sha256 = "{digest}"}} {{
  func.func @test(%a: tensor<32x32xi8>, %as: tensor<32x1xi8>,
                  %b: tensor<32x32xi8>, %bs: tensor<32x1xi8>) -> tensor<32x32xbf16> {{
    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) {{{attrs}}} :
      (tensor<32x32xi8>, tensor<32x1xi8>, tensor<32x32xi8>, tensor<32x1xi8>)
      -> tensor<32x32xbf16>
    %out = "mx_gemmini.readout_bf16"(%acc) {{{readout}}} :
      (tensor<32x32xbf16>) -> tensor<32x32xbf16>
    func.return %out : tensor<32x32xbf16>
  }}
}}
'''


def test_named_asymmetric_contract():
    result = verify_ir(_ir(), PROFILE_DATA)
    assert result["contracts"] == 1
    assert result["status"] == "structural_unqualified"


@pytest.mark.parametrize("change", [
    {"mode": 11},
    {"projection": "direct", "mode": 10},
    {"activation": "fp6_e3m2", "mode": 10},
    {"profile_digest": "0" * 64},
])
def test_illegal_contract_rejected(change):
    with pytest.raises(ValueError):
        verify_ir(_ir(**change), PROFILE_DATA)


def test_native_mlir_verifier_accepts_named_contract(tmp_path):
    executable = Path(os.getenv("MX_GEMMINI_OPT", "build/tools/mx-gemmini-opt"))
    if not executable.exists():
        pytest.skip("build mx-gemmini-opt or set MX_GEMMINI_OPT")
    path = tmp_path / "named.mlir"
    path.write_text(_ir())
    subprocess.run([str(executable.resolve()), str(path), "-o", "/dev/null"], check=True)
