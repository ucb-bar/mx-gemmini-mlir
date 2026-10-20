"""Audit a compiled TinyLlama ELF against its captured inputs and linked weight files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path

import numpy as np

from mx_gemmini_support.elf_payload import RV64PayloadImage


def file_sha(path, offset=0, length=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        remaining = length
        while remaining is None or remaining:
            data = stream.read(
                1 << 20 if remaining is None else min(remaining, 1 << 20)
            )
            if not data:
                if remaining:
                    raise ValueError(
                        "source payload is shorter than its declared extent"
                    )
                break
            digest.update(data)
            if remaining is not None:
                remaining -= len(data)
    return digest.hexdigest()


class Capture:
    def __init__(self, path):
        self.path = Path(path)
        self.weights = self.path / "weights.safetensors"
        receipt = json.loads((self.path / "capture_receipt.json").read_text())
        self.weights_sha256 = file_sha(self.weights)
        for name in (
            "weights.safetensors",
            "weights.safetensors.manifest.json",
            "inputs.npz",
            "extra.npz",
            "model.mlir",
        ):
            record = receipt["artifacts"][name]
            actual = (
                self.weights_sha256
                if name == "weights.safetensors"
                else file_sha(self.path / name)
            )
            if (
                actual != record["sha256"]
                or (self.path / name).stat().st_size != record["bytes"]
            ):
                raise ValueError("capture artifact changed: " + name)
        with self.weights.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            self.weight_header = json.loads(stream.read(length))
            self.weight_base = 8 + length
        self.raw_hashes = {}
        self.runtime = Counter()
        for name in ("extra.npz", "inputs.npz"):
            with np.load(self.path / name, allow_pickle=False) as arrays:
                for key in arrays.files:
                    array = arrays[key]
                    self.runtime[
                        (
                            array.dtype.str,
                            array.size,
                            hashlib.sha256(array.tobytes()).hexdigest(),
                        )
                    ] += 1

    def weight(self, name, shape):
        record = self.weight_header[name]
        if record["dtype"] != "F32" or record["shape"] != shape:
            raise ValueError(
                "retained host weight is not the captured F32 tensor: " + name
            )
        start, end = record["data_offsets"]
        size = math.prod(shape) * 4
        if end - start != size:
            raise ValueError("checkpoint weight extent differs from its shape")
        if name not in self.raw_hashes:
            self.raw_hashes[name] = file_sha(
                self.weights, self.weight_base + start, size
            )
        return size, self.raw_hashes[name]


def audit(build, capture):
    build = Path(build)
    catalog_path = build / "device_catalog/mx_catalog.json"
    catalog = json.loads(catalog_path.read_text())
    if catalog["checkpoint_weights_sha256"] != capture.weights_sha256:
        raise ValueError("compiled catalog belongs to a different checkpoint")
    directory = catalog_path.parent
    sources = catalog["compilation"]["source_sha256"]
    specials = {
        "runtime_quant_header": "runtime_quant.h",
        "runtime_quant_lut": "runtime_quant_lut.h",
        "runtime_group_prepare": "runtime_group_prepare.h",
        "checkpoint_weights": "checkpoint_weights.S",
    }
    for name, digest in sources.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("invalid compiled catalog source name")
        path = directory / specials.get(name, name + ".c")
        if file_sha(path) != digest:
            raise ValueError("compiled catalog source changed: " + name)
    if file_sha(directory / "mx_catalog.o") != catalog["compilation"]["object_sha256"]:
        raise ValueError("linked catalog object differs from its compilation record")
    projection = json.loads(
        (build / "entry_weight_projection/entry_weight_projection.json").read_text()
    )
    original = {r["original_index"]: r for r in projection["argument_table"]["entries"]}
    retained = projection["retained_original_indices"]
    if retained != sorted(set(retained)) or set(retained) & set(
        projection["removed_original_indices"]
    ):
        raise ValueError("entry argument projection is ambiguous")
    weights = [
        original[i]
        for i in retained
        if original[i]["kind"] == "immutable_stored_weight"
    ]
    header = (build / "cgen/model_gen.h").read_text()
    descriptors = re.findall(
        r"\{MERLIN_WEIGHT, (\d+)L, (\d+), \{([^}]*)\}, (\d+)\}", header
    )
    if len(descriptors) != len(weights):
        raise ValueError(
            "linked host weight descriptor count differs from the projected entry"
        )
    embedded = re.findall(
        r"static (float|long) (merlin_in_\d+)\[\] = \{([^}]+)\};",
        (build / "cgen/model_io.h").read_text(),
    )
    found = Counter()
    linked = []
    with RV64PayloadImage(build / "model.elf") as image:
        for scalar, symbol, text in embedded:
            values = text.split(",")
            dtype = "<i8" if scalar == "long" else "<f4"
            values = [
                int(v)
                if scalar == "long"
                else float(v.replace("INFINITY", "inf").replace("NAN", "nan"))
                for v in values
            ]
            array = np.array(values, dtype=dtype)
            digest = hashlib.sha256(array.tobytes()).hexdigest()
            if image.digest(symbol, array.nbytes) != digest:
                raise ValueError(
                    "actual ELF runtime input differs from its generated header: "
                    + symbol
                )
            found[(array.dtype.str, array.size, digest)] += 1
            linked.append({"symbol": symbol, "bytes": array.nbytes, "sha256": digest})
        if found != capture.runtime:
            raise ValueError(
                "actual ELF runtime inputs differ from the captured mask, frequencies or token IDs"
            )
        for kernel in catalog["kernels"]:
            if not kernel["prepacked_checkpoint_weight"]:
                continue
            for kind in ("codes", "scales"):
                symbol = kernel["symbol"] + "_fixed_" + kind
                size, digest = kernel[kind + "_bytes"], kernel[kind + "_sha256"]
                path = directory / (kernel["symbol"] + "_" + kind + ".bin")
                if (
                    path.stat().st_size != size
                    or file_sha(path) != digest
                    or image.digest(symbol, size) != digest
                ):
                    raise ValueError(
                        "actual ELF packed checkpoint payload differs: " + symbol
                    )
                linked.append({"symbol": symbol, "bytes": size, "sha256": digest})
        blob = build / "cgen/weights.bin"
        blob_size = blob.stat().st_size
        blob_sha = file_sha(blob)
        if image.digest("_binary_weights_bin_start", blob_size) != blob_sha:
            raise ValueError(
                "actual ELF host weight blob differs from the generated data file"
            )
        host = []
        for record, (offset, rank, shape, width) in zip(
            weights, descriptors, strict=True
        ):
            shape = [int(x) for x in shape.split(",")]
            if int(rank) != len(shape) or shape != record["shape"] or int(width) != 4:
                raise ValueError(
                    "host weight descriptor layout differs from the projected source"
                )
            size, digest = capture.weight(record["weight_name"], shape)
            if (
                int(offset) + size > blob_size
                or file_sha(blob, int(offset), size) != digest
            ):
                raise ValueError(
                    "retained host weight bytes differ from checkpoint: "
                    + record["weight_name"]
                )
            host.append(
                {
                    "checkpoint_parameter": record["weight_name"],
                    "offset": int(offset),
                    "bytes": size,
                    "sha256": digest,
                }
            )
    return {
        "schema": "mx_gemmini.linked_payload_audit.v1",
        "status": "linked_payloads_match",
        "scope": "linked data identity and compiler inputs; numerical and timing qualification are separate",
        "elf_sha256": file_sha(build / "model.elf"),
        "catalog_sha256": file_sha(catalog_path),
        "capture_receipt_sha256": file_sha(capture.path / "capture_receipt.json"),
        "checkpoint_sha256": capture.weights_sha256,
        "compiled_source_files_checked": len(sources),
        "packed_matrices_checked": sum(
            k["prepacked_checkpoint_weight"] for k in catalog["kernels"]
        ),
        "host_weight_blob": {
            "bytes": blob_size,
            "sha256": blob_sha,
            "checkpoint_parameters": host,
        },
        "linked_payloads": linked,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--build", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be fresh")
    capture = Capture(args.capture)
    reports = []
    for build in args.build:
        report = audit(build, capture)
        reports.append({"build": str(build.resolve()), **report})
        print(
            json.dumps(
                {
                    "build": str(build),
                    "status": report["status"],
                    "packed_matrices": report["packed_matrices_checked"],
                }
            ),
            flush=True,
        )
    args.out.write_text(json.dumps({"audits": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
