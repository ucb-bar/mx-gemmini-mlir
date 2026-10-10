"""GQA source selection and explicit candidate provenance checks."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from mx_gemmini_support.source_attention_qk import (
    HARDWARE_GENERATOR_SHA256, HARDWARE_MODEL_SHA256,
    derive_shifted_qk, read_first_gqa_qk, source_product_overflow_count)
from mx_gemmini_support.source_payload import (
    ATTENTION_QK_CANDIDATE_ORIGIN, validate_attention_qk_candidate)


def test_attention_candidate_requires_explicit_derivation():
    manifest = {
        "origin": ATTENTION_QK_CANDIDATE_ORIGIN,
        "source_header_sha256": "a" * 64,
        "precision": "FP8", "shape_mnk": [64, 64, 64],
        "tile_mnk": [64, 64, 64],
        "source_derivation": {
            "schema": "mx_gemmini.attention_qk_shift.v1",
            "stage": "gqa_qk_head0_block0", "e8m0_shift": 6,
            "oracle": "dim16_reduced_precision_product_and_accumulator",
            "source_header_sha256": "a" * 64,
            "source_arrays_sha256": {name: "b" * 64 for name in (
                "activation", "weight", "activation_scales", "weight_scales")},
        },
    }
    validate_attention_qk_candidate(manifest)
    for change in ({"origin": "radiance_source_header_specialization"},
                   {"source_derivation": {**manifest["source_derivation"],
                                          "e8m0_shift": 0}},
                   {"source_derivation": {**manifest["source_derivation"],
                                          "source_header_sha256": "c" * 64}}):
        altered = {**manifest, **change}
        with pytest.raises(ValueError, match="candidate"):
            validate_attention_qk_candidate(altered)


def test_first_gqa_qk_source_and_shifted_candidate():
    source = os.getenv("RADIANCE_KERNELS_ROOT")
    if not source:
        pytest.skip("set RADIANCE_KERNELS_ROOT to the pinned Radiance checkout")
    root = Path(source)
    if not (root / "kernels/flash_attention_mx_gqa/include/fa_data.h").exists():
        pytest.skip("generate the pinned GQA data header before this source test")
    tile = read_first_gqa_qk(root)
    assert source_product_overflow_count(tile) == 244154
    sys.path.insert(0, str(root / "lib/mxgemmini"))
    try:
        import torch
        import fp8_matmul_model as model
        if Path(model.__file__).resolve() != (root / "lib/mxgemmini/fp8_matmul_model.py").resolve():
            pytest.skip("the pinned Radiance MX model is not imported")
        with pytest.raises(ValueError, match="still overflows"):
            derive_shifted_qk(tile, 5, torch=torch, low_level_model=model)
        resources, policy = derive_shifted_qk(tile, 6, torch=torch,
                                              low_level_model=model)
        assert policy["source_arrays_sha256"] == tile.hashes()
        assert len(resources["golden_bf16"].data) == 8192
        assert resources["activation"].descriptor("activation")["sha256"] == (
            "6504b226ca3ddc4702a89d5f4ef3aff9129b372090da052fdea62048ddb5bd7b")
    finally:
        sys.path.pop(0)


def test_hardware_model_patch_applies_only_to_pinned_source(tmp_path):
    source = os.getenv("RADIANCE_KERNELS_ROOT")
    if not source:
        pytest.skip("set RADIANCE_KERNELS_ROOT to the pinned Radiance checkout")
    source_root = Path(source)
    original = source_root / "kernels/flash_attention_mx_gqa"
    copied = tmp_path / "kernels/flash_attention_mx_gqa"
    copied.mkdir(parents=True)
    names = ("fa_gen_data.py", "flash_attention_model.py")
    before = {}
    for name in names:
        before[name] = (original / name).read_bytes()
        shutil.copyfile(original / name, copied / name)
    patch = (Path(__file__).resolve().parents[1] /
             "docs/patches/radiance_gqa_hardware_model.patch")
    subprocess.run(["git", "apply", "--unidiff-zero", "--check", str(patch)],
                   cwd=tmp_path, check=True)
    subprocess.run(["git", "apply", "--unidiff-zero", str(patch)],
                   cwd=tmp_path, check=True)
    import hashlib
    assert hashlib.sha256((copied / names[0]).read_bytes()).hexdigest() == HARDWARE_GENERATOR_SHA256
    assert hashlib.sha256((copied / names[1]).read_bytes()).hexdigest() == HARDWARE_MODEL_SHA256
    assert all((original / name).read_bytes() == before[name] for name in names)
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(["git", "apply", "--unidiff-zero", "--check", str(patch)],
                       cwd=tmp_path, check=True, capture_output=True)
