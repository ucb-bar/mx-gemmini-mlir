"""Pinned C source bytes for the RTL-tested diagnostic load images."""

from hashlib import sha256

import pytest

from mx_gemmini_support.bringup import canonical_bringup_source


@pytest.mark.parametrize("case,fmt,expected_sha256", [
    ("square32", "mxfp8", "6ada7eb2e93468ea9905cedb77fa6fd6547bba7ff5364a0d752d95dbc76a5cd7"),
    ("square32", "mxfp6", "0b9d02a4b961a40ea7a7467f59363dc6c8b367e35b712c4e8543c561066b1b04"),
    ("square32", "mxfp4", "034a4ca127927bf3337a76c8096e1b73365f2e30eecbdaabf29c3cee45741966"),
    ("square64", "mxfp8", "629b9080382d526d79e6c42f2133f4d05ab8814c00bdd9a5636e373d3f25efa7"),
    ("square64", "mxfp6", "dde663622142c4b9ef0c52c10ffd75182bef643d2d854e20fbfe9ced69876c5f"),
    ("square64", "mxfp4", "96d5d0e2ec4f4afb9458787b567eddde8440169bd40d2965db33fc99eaa74047"),
    ("split32x64", "mxfp8", "3244e190f3fc6d86c78ec16e81d313a34ac8a86478ba1b08d26c732eb138cd40"),
    ("split32x64", "mxfp6", "fb85fc8538dce128ae6d106409359598ddcd49934b44e62f2b8bb17bcf9f78d0"),
    ("split32x64", "mxfp4", "cea3ce74c060bb60c2499b6c9bcc392b9ed21ddffb9d56cd9f36b942bc44c868"),
])
def test_canonical_rtl_bringup_source(case, fmt, expected_sha256):
    source = canonical_bringup_source(fmt, case).encode()
    assert sha256(source).hexdigest() == expected_sha256
