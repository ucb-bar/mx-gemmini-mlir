"""Bind a gpt-oss MXFP4 checkpoint slice to a lossless Gemmini B tile.

The extractor receipt pins the original tensor bytes, expert index and source
range. This module verifies those facts before repacking. It does not quantize
activations, emit a GEMM, or claim that the selected Spike matches the target.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .gptoss_checkpoint import repack_expert_tile, restore_checkpoint_layout
from .target_profile import profile_sha256, require_compute


SCHEMA = "mx_gemmini.gptoss_native_expert_tile.v1"
_HEX = frozenset("0123456789abcdef")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object, length: int, label: str) -> str:
    if (not isinstance(value, str) or len(value) != length or
            any(char not in _HEX for char in value)):
        raise ValueError(f"{label} needs a lowercase {length}-digit SHA")
    return value


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safetensors header key {key!r}")
        result[key] = value
    return result


def read_single_u8_safetensors(path: Path) -> tuple[str, np.ndarray, str]:
    """Read exactly one U8 tensor, with no extra payload or pickle dependency."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("checkpoint slice needs a regular safetensors file")
    raw = path.read_bytes()
    if len(raw) < 10:
        raise ValueError("checkpoint slice is shorter than a safetensors header")
    header_bytes = int.from_bytes(raw[:8], "little")
    if header_bytes <= 0 or header_bytes > len(raw) - 8:
        raise ValueError("checkpoint slice has an invalid safetensors header size")
    header = json.loads(raw[8:8 + header_bytes], object_pairs_hook=_unique_json)
    if not isinstance(header, dict):
        raise ValueError("checkpoint slice has an invalid safetensors header")
    tensors = [(name, row) for name, row in header.items() if name != "__metadata__"]
    if len(tensors) != 1:
        raise ValueError("checkpoint slice must contain exactly one tensor")
    name, row = tensors[0]
    if (not isinstance(name, str) or not name or not isinstance(row, dict) or
            row.get("dtype") != "U8" or set(row) != {"dtype", "shape", "data_offsets"}):
        raise ValueError("checkpoint slice needs one named U8 tensor")
    shape, offsets = row["shape"], row["data_offsets"]
    payload = raw[8 + header_bytes:]
    if (not isinstance(shape, list) or not shape or
            any(type(value) is not int or value <= 0 for value in shape) or
            offsets != [0, len(payload)] or math.prod(shape) != len(payload)):
        raise ValueError("checkpoint slice shape, offset and payload disagree")
    return name, np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy(), _sha(raw)


def _selected_row(receipt: dict, name: str, expert_index: int) -> dict:
    tensors = receipt.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError("checkpoint extractor receipt has no tensor rows")
    selected = [row for row in tensors if isinstance(row, dict) and row.get("name") == name]
    if len(selected) != 1:
        raise ValueError(f"checkpoint extractor receipt has no unique tensor {name!r}")
    row = selected[0]
    source_shape, shape = row.get("source_shape"), row.get("selected_shape")
    if (row.get("dtype") != "U8" or row.get("axis0_index") != expert_index or
            not isinstance(source_shape, list) or not isinstance(shape, list) or
            len(source_shape) < 2 or len(shape) != len(source_shape) or
            shape != [1, *source_shape[1:]] or
            type(source_shape[0]) is not int or not 0 <= expert_index < source_shape[0]):
        raise ValueError(f"checkpoint tensor {name!r} expert index or shape differs")
    _digest(row.get("raw_sha256"), 64, f"checkpoint tensor {name!r} raw hash")
    source_range = row.get("source_byte_range")
    if (not isinstance(source_range, list) or len(source_range) != 2 or
            any(type(value) is not int or value < 0 for value in source_range) or
            source_range[1] < source_range[0] or
            source_range[1] - source_range[0] + 1 != math.prod(shape)):
        raise ValueError(f"checkpoint tensor {name!r} has an invalid inclusive source range")
    if (type(row.get("source_shard_bytes")) is not int or
            row["source_shard_bytes"] <= source_range[1] or
            not isinstance(row.get("shard"), str) or not row["shard"]):
        raise ValueError(f"checkpoint tensor {name!r} has an invalid source shard")
    if (not isinstance(row.get("file"), str) or not row["file"] or
            Path(row["file"]).name != row["file"] or
            not row["file"].endswith(".safetensors")):
        raise ValueError(f"checkpoint tensor {name!r} has no extracted file")
    return row


def bind_native_expert_tile(extractor_receipt: dict, receipt_bytes: bytes,
                            fixture_dir: Path, *, projection: str,
                            expert_index: int, output_start: int,
                            output_count: int, block_start: int,
                            block_count: int, activation_bf16: bytes,
                            activation_m: int, expected_revision: str,
                            profile: dict | None = None
                            ) -> tuple[bytes, bytes, dict]:
    """Verify pinned checkpoint slices and return B codes/scales plus a receipt."""
    if (not isinstance(extractor_receipt, dict) or
            extractor_receipt.get("schema") != "mx_checkpoint_ranges.v1" or
            extractor_receipt.get("model") not in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}):
        raise ValueError("expected a gpt-oss checkpoint range receipt")
    if json.loads(receipt_bytes) != extractor_receipt:
        raise ValueError("checkpoint receipt bytes do not match parsed receipt")
    revision = _digest(extractor_receipt.get("revision"), 40, "checkpoint revision")
    if revision != _digest(expected_revision, 40, "expected checkpoint revision"):
        raise ValueError("checkpoint revision differs from pinned recipe revision")
    if projection not in {"gate_up", "down"}:
        raise ValueError("projection must be gate_up or down")
    if type(expert_index) is not int or expert_index < 0:
        raise ValueError("expert index must be nonnegative")
    prefix = "model.layers.0.mlp.experts."
    names = {"blocks": prefix + projection + "_proj_blocks",
             "scales": prefix + projection + "_proj_scales"}
    arrays, source = {}, {}
    for kind, name in names.items():
        row = _selected_row(extractor_receipt, name, expert_index)
        tensor_path = fixture_dir / row["file"]
        header_name, values, file_sha = read_single_u8_safetensors(tensor_path)
        if (header_name != name or list(values.shape) != row["selected_shape"] or
                _sha(values.tobytes()) != row["raw_sha256"]):
            raise ValueError(f"checkpoint {kind} file differs from selected source tensor")
        arrays[kind] = values[0]
        source[kind] = {"name": name, "raw_sha256": row["raw_sha256"],
                        "file_sha256": file_sha, "source_byte_range": row.get("source_byte_range"),
                        "source_shard": row.get("shard")}
    if (arrays["blocks"].ndim != 3 or arrays["scales"].ndim != 2 or
            arrays["blocks"].shape[:2] != arrays["scales"].shape or
            arrays["blocks"].shape[-1] != 16):
        raise ValueError("checkpoint blocks and E8M0 scales disagree")
    packed = repack_expert_tile(arrays["blocks"], arrays["scales"],
                                output_start=output_start, output_count=output_count,
                                block_start=block_start, block_count=block_count)
    selected_blocks = arrays["blocks"][output_start:output_start + output_count,
                                        block_start:block_start + block_count]
    selected_scales = arrays["scales"][output_start:output_start + output_count,
                                        block_start:block_start + block_count]
    restored_blocks, restored_scales = restore_checkpoint_layout(packed)
    if (not np.array_equal(restored_blocks, selected_blocks) or
            not np.array_equal(restored_scales, selected_scales)):
        raise ValueError("Gemmini B tile does not restore checkpoint bytes exactly")
    if (type(activation_m) is not int or activation_m <= 0 or
            not isinstance(activation_bf16, bytes) or
            len(activation_bf16) != activation_m * packed.k * 2):
        raise ValueError("activation operand must be M x selected K row-major BF16 bytes")
    mode = None
    if profile is not None:
        if profile.get("schema") != "mx_gemmini.target_profile.v2":
            raise ValueError("native tile needs an MX target profile v2")
        candidates = [row for row in profile.get("legal_compute", []) if
                      row.get("activation_format") == "fp4_e2m1" and
                      row.get("weight_format") == "fp4_e2m1" and
                      row.get("activation_projection") == "direct" and
                      row.get("weight_projection") == "direct"]
        if len(candidates) != 1:
            raise ValueError("selected profile has no unique direct FP4 A/B mesh mode")
        mode = candidates[0]
        require_compute(profile, "fp4_e2m1", "fp4_e2m1",
                        pe_mode=mode["pe_mode"], activation_projection="direct",
                        weight_projection="direct")
    manifest = {
        "schema": SCHEMA,
        "status": "source_bound_b_tile_and_bf16_activation_no_executable_or_spike_parity",
        "execution_refusal": (
            "The direct FP4 mesh mode requires packed MXFP4 A codes and E8M0 A scales; "
            "this fixture supplies explicit BF16 A bytes and no authorized activation "
            "quantization policy. No direct BF16 x checkpoint-MXFP4 GEMM lowering is "
            "available in this compiler backend."
        ),
        "model": extractor_receipt["model"], "checkpoint_revision": revision,
        "extractor_receipt_sha256": _sha(receipt_bytes),
        "projection": projection, "expert_index": expert_index,
        "output_start": output_start, "output_count": output_count,
        "block_start": block_start, "block_count": block_count,
        "shape_mnk": [activation_m, packed.n, packed.k], "source_tensors": source,
        "activation": {"format": "bf16", "layout": "row_major_mk",
                       "file": "a_bf16.bin",
                       "sha256": _sha(activation_bf16),
                       "bytes": len(activation_bf16)},
        "b_codes_sha256": _sha(packed.weight_bytes),
        "b_scales_sha256": _sha(packed.scale_bytes),
        "b_codes_bytes": len(packed.weight_bytes),
        "b_scales_bytes": len(packed.scale_bytes),
        "profile_sha256": profile_sha256(profile) if profile is not None else None,
        "hardware_mode": mode,
    }
    return packed.weight_bytes, packed.scale_bytes, manifest
