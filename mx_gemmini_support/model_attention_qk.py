"""Bind one actual model2MLIR attention QK head to the ordinary MXFP8 path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from .model_projection import MODEL_MATH_SHA256
from .operands import pack_direct_operands
from .source_payload import (MODEL2MLIR_ATTENTION_QK_ORIGIN,
                             MODEL2MLIR_PROJECTION_GOLDEN_SHA256, Resource,
                             load_bundle, validate_model2mlir_attention_qk)
from .target_profile import profile_sha256, require_compute


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_model_attention_qk_bundle(directory: Path, *, worklist: dict, site: dict,
                                    batch_index: int, activation, weight,
                                    tile_k: int, profile: dict,
                                    reference_root: Path) -> dict:
    """Quantize a head's captured Q and K operands without changing its source site."""
    import numpy as np
    import torch
    from mxq.nn.operand_capture import functional_contraction_operands

    if (worklist.get("schema") != "mx_gemmini.model2mlir_contraction_worklist.v1" or
            site not in worklist.get("rank3_batch_matmuls", []) or
            site.get("source_op") != "batch_matmul" or
            ".self_attn" not in (site.get("fqn") or "") or
            site.get("element_type") != "f32" or
            worklist.get("profile_sha256") != profile_sha256(profile)):
        raise ValueError("attention QK site is absent from selected model2MLIR worklist")
    batch, m, n, k = site["shape_bmnk"]
    if (type(batch_index) is not int or not 0 <= batch_index < batch or
            not isinstance(activation, np.ndarray) or
            not isinstance(weight, np.ndarray) or
            activation.dtype != np.float32 or weight.dtype != np.float32 or
            activation.shape != (m, k) or weight.shape != (k, n) or
            not np.isfinite(activation).all() or not np.isfinite(weight).all() or
            k % 32 or n > k or type(tile_k) is not int or tile_k < 32 or
            tile_k % 32 or k % tile_k):
        raise ValueError("attention QK operands or tile differ from source batched site")
    if (profile["geometry"]["mesh_rows"] != 16 or
            profile["geometry"]["mesh_columns"] != 16):
        raise ValueError("attention QK currently needs DIM16")
    require_compute(profile, "fp8_e4m3", "fp8_e4m3", pe_mode=8,
                    activation_projection="direct", weight_projection="direct")
    padded_m, padded_n = ((m + 15) // 16) * 16, ((n + 31) // 32) * 32
    a = torch.zeros((padded_m, k), dtype=torch.bfloat16)
    b = torch.zeros((k, padded_n), dtype=torch.bfloat16)
    a[:m] = torch.from_numpy(activation).to(torch.bfloat16)
    b[:, :n] = torch.from_numpy(weight).to(torch.bfloat16)
    encoded = functional_contraction_operands(a, b, "mxfp8")
    packed = pack_direct_operands(
        "mxfp8", encoded.activation_codes.tolist(), encoded.weight_codes.tolist())
    resources = {
        "activation": Resource(packed.activation_bytes, (padded_m, k), 8,
                               "row_major_codes"),
        "weight": Resource(packed.weight_bytes, (k, padded_n), 8,
                           "row_major_codes"),
        "activation_scales": Resource(encoded.activation_scales.T.contiguous().numpy().tobytes(),
                                      (k // 32, padded_m), 8, "k_group_row_e8m0"),
        "weight_scales": Resource(encoded.weight_scales.T.contiguous().numpy().tobytes(),
                                  (k // 32, padded_n), 8, "k_group_column_e8m0"),
        "output_scales": Resource(bytes(padded_m * padded_n // 32),
                                  (padded_n // 32, padded_m), 8,
                                  "n_group_row_e8m0"),
    }
    model_cpp = reference_root / "lib/golden/mx_golden.cpp"
    model_math = reference_root / "lib/golden/mx_fp_math.h"
    if (_sha(model_cpp.read_bytes()) != MODEL2MLIR_PROJECTION_GOLDEN_SHA256 or
            _sha(model_math.read_bytes()) != MODEL_MATH_SHA256):
        raise ValueError("independent MX arithmetic reference differs from pinned source")
    if directory.exists():
        raise ValueError(f"refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    for name, resource in resources.items():
        (directory / f"{name}.bin").write_bytes(resource.data)
    with TemporaryDirectory(prefix="model-attention-reference-", dir=directory.parent) as temp:
        executable = Path(temp) / "mx_golden"
        subprocess.run(["g++", "-O2", "-std=c++17", "-I", str(model_cpp.parent),
                        str(model_cpp), "-o", str(executable)], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        golden_path = directory / "golden_bf16.bin"
        subprocess.run([str(executable), str(padded_m), str(padded_n), str(k),
                        *(str(directory / f"{name}.bin") for name in
                          ("activation", "weight", "activation_scales", "weight_scales")),
                        str(golden_path), "0"], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    resources["golden_bf16"] = Resource(golden_path.read_bytes(),
                                         (padded_m, padded_n), 16, "row_major_bf16")
    manifest = {
        "schema": "mx_gemmini.source_payload.v1",
        "site_id": "functional:matmul", "precision": "FP8",
        "shape_mnk": [padded_m, padded_n, k],
        "tile_mnk": [padded_m, 32, tile_k],
        "profile_sha256": profile_sha256(profile),
        "origin": MODEL2MLIR_ATTENTION_QK_ORIGIN,
        "model2mlir_attention_qk": {
            "source_mlir_sha256": worklist["source_mlir_sha256"],
            "region_id": site["region_id"], "fqn": site["fqn"],
            "source_op": "batch_matmul", "source_shape_bmnk": site["shape_bmnk"],
            "batch_index": batch_index,
            "input_sha256": _sha(activation.tobytes()),
            "weight_sha256": _sha(weight.tobytes()),
            "golden_model_sha256": MODEL2MLIR_PROJECTION_GOLDEN_SHA256,
        },
        "resources": {name: resource.descriptor(name)
                      for name, resource in sorted(resources.items())},
    }
    validate_model2mlir_attention_qk(manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    load_bundle(directory)
    return manifest
