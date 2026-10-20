"""Corrupted linked constants must fail even when their source headers look correct."""

import hashlib
import json
import struct

import numpy as np
import pytest
from test_elf_payload import write_payload_elf

from tools.audit_merlin_elf_payloads import Capture, audit


def fixture(root):
    capture = root / "capture"
    capture.mkdir()
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}
    ).encode()
    weight = np.array([1, 2, 3, 4], "<f4").tobytes()
    (capture / "weights.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + weight
    )
    (capture / "weights.safetensors.manifest.json").write_text("{}")
    np.savez(capture / "inputs.npz", in0=np.array([[42, 24]], "<i8"))
    np.savez(capture / "extra.npz")
    (capture / "model.mlir").write_text("module {}")
    sha = lambda data: hashlib.sha256(data).hexdigest()
    artifacts = {
        p.name: {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
        for p in capture.iterdir()
    }
    (capture / "capture_receipt.json").write_text(json.dumps({"artifacts": artifacts}))
    build = root / "build"
    catalog = build / "device_catalog"
    catalog.mkdir(parents=True)
    cgen = build / "cgen"
    cgen.mkdir()
    projection = build / "entry_weight_projection"
    projection.mkdir()
    (catalog / "runtime_quant.c").write_text("/* retained source */")
    (catalog / "mx_catalog.o").write_bytes(b"linked object")
    (catalog / "matrix_codes.bin").write_bytes(b"code")
    (catalog / "matrix_scales.bin").write_bytes(b"s")
    manifest = {
        "checkpoint_weights_sha256": sha(
            (capture / "weights.safetensors").read_bytes()
        ),
        "compilation": {
            "source_sha256": {
                "runtime_quant": sha((catalog / "runtime_quant.c").read_bytes())
            },
            "object_sha256": sha((catalog / "mx_catalog.o").read_bytes()),
        },
        "kernels": [
            {
                "symbol": "matrix",
                "prepacked_checkpoint_weight": True,
                "codes_bytes": 4,
                "codes_sha256": sha(b"code"),
                "scales_bytes": 1,
                "scales_sha256": sha(b"s"),
            }
        ],
    }
    (catalog / "mx_catalog.json").write_text(json.dumps(manifest))
    (projection / "entry_weight_projection.json").write_text(
        json.dumps(
            {
                "argument_table": {
                    "entries": [
                        {
                            "original_index": 0,
                            "kind": "immutable_stored_weight",
                            "shape": [2, 2],
                            "weight_name": "weight",
                        },
                        {"original_index": 1, "kind": "retained_capture"},
                    ]
                },
                "retained_original_indices": [0, 1],
                "removed_original_indices": [],
            }
        )
    )
    (cgen / "model_gen.h").write_text("{MERLIN_WEIGHT, 0L, 2, {2,2}, 4}")
    (cgen / "model_io.h").write_text("static long merlin_in_1[] = {42,24};")
    (cgen / "weights.bin").write_bytes(weight)
    payloads = {
        "merlin_in_1": np.array([42, 24], "<i8").tobytes(),
        "matrix_fixed_codes": b"code",
        "matrix_fixed_scales": b"s",
        "_binary_weights_bin_start": weight,
    }
    offsets = write_payload_elf(build / "model.elf", payloads)
    return capture, build, offsets


def test_audit_reopens_actual_linked_data_and_retained_checkpoint_bytes(tmp_path):
    capture, build, _ = fixture(tmp_path)
    report = audit(build, Capture(capture))
    assert (
        report["status"] == "linked_payloads_match"
        and report["packed_matrices_checked"] == 1
    )
    assert len(report["host_weight_blob"]["checkpoint_parameters"]) == 1
    assert len(report["linked_payloads"]) == 3


@pytest.mark.parametrize(
    "symbol",
    [
        "merlin_in_1",
        "matrix_fixed_codes",
        "matrix_fixed_scales",
        "_binary_weights_bin_start",
    ],
)
def test_changed_elf_payload_cannot_hide_behind_unchanged_header_or_blob(
    tmp_path, symbol
):
    capture, build, offsets = fixture(tmp_path)
    elf = build / "model.elf"
    data = bytearray(elf.read_bytes())
    data[offsets[symbol]] ^= 1
    elf.write_bytes(data)
    with pytest.raises(ValueError):
        audit(build, Capture(capture))


def test_changed_host_blob_and_elf_still_must_match_original_checkpoint(tmp_path):
    capture, build, offsets = fixture(tmp_path)
    blob = build / "cgen/weights.bin"
    data = bytearray(blob.read_bytes())
    data[0] ^= 1
    blob.write_bytes(data)
    elf = build / "model.elf"
    data = bytearray(elf.read_bytes())
    data[offsets["_binary_weights_bin_start"]] ^= 1
    elf.write_bytes(data)
    with pytest.raises(ValueError, match="differ from checkpoint"):
        audit(build, Capture(capture))


def test_changed_compiler_source_or_capture_is_rejected(tmp_path):
    capture, build, _ = fixture(tmp_path)
    (build / "device_catalog/runtime_quant.c").write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        audit(build, Capture(capture))
    (capture / "model.mlir").write_text("changed")
    with pytest.raises(ValueError, match="capture artifact changed"):
        Capture(capture)
