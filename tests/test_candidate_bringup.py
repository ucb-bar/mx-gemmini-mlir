"""The candidate generator must reproduce the C sources run on the latest RTL."""

import hashlib

import pytest

from mx_gemmini_support.candidate_bringup import canonical_scale_load_source


@pytest.mark.parametrize("fmt,variant,digest", [
    ("mxfp8", "pitched", "b9e88dc7f2ddbbf6c1ae163f6054d3791e9a6b2f5a13e6b7f630c15f9f2ecb54"),
    ("mxfp6", "pitched", "5c34cd7e95c0fbd20b1e92cc5544bee36d974e60070978a33f2577f164edd200"),
    ("mxfp4", "pitched", "4cdd1b818ddf72d03e65ea6802d360a1a5573cf69176d6fd3c8a5b7a32a95319"),
    ("mxfp8", "pitched-gated-alt-half", "7c59fa0798a1f60eab7260616c347a9f5539ed1e09ea4edf4ef0f97c45238a13"),
    ("mxfp8", "loop-managed", "6264b4957cde674ec9f8eeec98eb62ed308689bd1ed9d5426676f4a6695de651"),
    ("mxfp6", "loop-managed", "b9aa22f2788b49fb2119a0cb7a50b40bff90d42b4d527f15375c6650e884bd19"),
    ("mxfp4", "loop-managed", "435f1fd234f436544a365ff5330a9bee0e5de7f16c8d72d2b8bacfc76d7546ae"),
    ("mxfp8", "loop-managed-pitched-64", "610fa4d1ac8778fa3d3f7ea6ab39bd0d5db5fabd01bf1525426c60d8db4143b7"),
])
def test_candidate_source_matches_executed_program(fmt, variant, digest):
    source = canonical_scale_load_source(fmt, variant)
    assert hashlib.sha256(source.encode()).hexdigest() == digest
    if variant.startswith("loop-managed"):
        assert source.count("(uintptr_t)B_scales, 31);") == 1
    else:
        assert source.count("k_MX_LOAD_SCALES);") == 2
        assert "gemmini_mx_load_scales(" not in source
    if variant == "pitched-gated-alt-half":
        assert "(1ULL << 17) | (1ULL << 16)" in source


def test_candidate_source_rejects_unknown_variant():
    with pytest.raises(ValueError, match="declared MX format and variant"):
        canonical_scale_load_source("mxfp8", "unknown")
