"""The candidate generator must reproduce the C sources run on the latest RTL."""

import hashlib

import pytest

from mx_gemmini_support.candidate_bringup import canonical_scale_load_source


@pytest.mark.parametrize("fmt,variant,digest", [
    ("mxfp8", "pitched", "b9e88dc7f2ddbbf6c1ae163f6054d3791e9a6b2f5a13e6b7f630c15f9f2ecb54"),
    ("mxfp6", "pitched", "5c34cd7e95c0fbd20b1e92cc5544bee36d974e60070978a33f2577f164edd200"),
    ("mxfp4", "pitched", "4cdd1b818ddf72d03e65ea6802d360a1a5573cf69176d6fd3c8a5b7a32a95319"),
    ("mxfp8", "pitched-gated-alt-half", "7c59fa0798a1f60eab7260616c347a9f5539ed1e09ea4edf4ef0f97c45238a13"),
])
def test_candidate_source_matches_executed_program(fmt, variant, digest):
    source = canonical_scale_load_source(fmt, variant)
    assert hashlib.sha256(source.encode()).hexdigest() == digest
    assert source.count("k_MX_LOAD_SCALES);") == 2
    assert "gemmini_mx_load_scales(" not in source
    if variant == "pitched-gated-alt-half":
        assert "(1ULL << 17) | (1ULL << 16)" in source


def test_candidate_source_rejects_unknown_variant():
    with pytest.raises(ValueError, match="declared MX format and variant"):
        canonical_scale_load_source("mxfp8", "unknown")
