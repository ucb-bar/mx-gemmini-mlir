"""Cross-check the PV handoff diagnostic against the pinned Muon C encoder."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from mx_gemmini_support.source_attention_pv import (
    MUON_REQUANT_SOURCE_SHA256, source_e4m3_scaled)


def test_source_muon_e4m3_encoder_matches_python_translation(tmp_path):
    root = os.getenv("RADIANCE_KERNELS_ROOT")
    if not root:
        pytest.skip("set RADIANCE_KERNELS_ROOT to the pinned Radiance checkout")
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host g++ is required to execute the source C helper")
    source = (Path(root) /
              "kernels/flash_attention_mx_gqa/flash_mx_impl.hpp").read_bytes()
    assert hashlib.sha256(source).hexdigest() == MUON_REQUANT_SOURCE_SHA256
    code = source.decode()
    begin = code.index("static inline uint8_t bf16_to_e4m3_scaled(")
    end = code.index("// 16-lane intra-warp tree reduction", begin)
    helper = code[begin:end]
    harness = tmp_path / "encoder.cpp"
    harness.write_text(
        "#include <cstdint>\n#include <cstdio>\n" + helper +
        "\nint main() { for (int se=-6; se<=0; ++se) "
        "for (int b=0; b<=0x7f7f; ++b) "
        "std::putchar(bf16_to_e4m3_scaled((uint16_t)b,se)); }\n")
    executable = tmp_path / "encoder"
    subprocess.run([compiler, "-O2", "-std=c++17", str(harness), "-o",
                    str(executable)], check=True)
    actual = subprocess.run([str(executable)], check=True,
                            stdout=subprocess.PIPE).stdout
    expected = bytes(source_e4m3_scaled(b, se)
                     for se in range(-6, 1) for b in range(0x7f80))
    assert actual == expected
