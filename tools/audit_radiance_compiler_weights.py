"""Compare actual source/compiler FP8 checkpoint payloads after undoing layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mx_gemmini_support.elf_payload import RV64PayloadImage
from tools.audit_merlin_elf_payloads import file_sha


def unpack_weight(codes, scales, *, k, n, tile_k):
    """Undo the source's [KS,NT,TK,64] and [NT,KS,TK/32,64] layouts."""
    if min(k, n, tile_k) <= 0 or k % tile_k or tile_k % 32 or n % 64:
        raise ValueError("source weight extents do not cover complete FP8 tiles")
    if len(codes) != k * n or len(scales) != k * n // 32:
        raise ValueError("source weight payload extent differs from its descriptor")
    ks, nt = k // tile_k, n // 64
    logical_codes = (
        np.frombuffer(codes, np.uint8)
        .reshape(ks, nt, tile_k, 64)
        .transpose(0, 2, 1, 3)
        .reshape(k, n)
        .copy()
    )
    logical_scales = (
        np.frombuffer(scales, np.uint8)
        .reshape(nt, ks, tile_k // 32, 64)
        .transpose(1, 2, 0, 3)
        .reshape(k // 32, n)
        .copy()
    )
    return logical_codes, logical_scales


def phase_parameters(layer, operation):
    if operation == 14 and layer == 255:
        return ["lm.lm_head.weight"]
    if not 0 <= layer < 22:
        raise ValueError("source learned matrix has an unknown layer")
    parts = {
        2: ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        8: ["self_attn.o_proj"],
        10: ["mlp.gate_proj", "mlp.up_proj"],
        12: ["mlp.down_proj"],
    }.get(operation)
    if parts is None:
        raise ValueError("source learned matrix has an unknown phase")
    return [f"lm.model.layers.{layer}.{part}.weight" for part in parts]


def audit(port, compiler):
    port, compiler = Path(port), Path(compiler)
    receipt = json.loads((port / "receipt.json").read_text())
    catalog_raw = (compiler / "device_catalog/mx_catalog.json").read_bytes()
    catalog = json.loads(catalog_raw)
    if (
        receipt["source_revision"] != "5866983682583486b38ae64a4d1bfa5e7b2a002e"
        or receipt["mode"] != "recorded_tinyllama_forward"
        or catalog["checkpoint_weights_sha256"]
        != receipt["capture_files_sha256"]["weights.safetensors"]
        or catalog["profile_sha256"] != receipt["profile_sha256"]
    ):
        raise ValueError(
            "source/compiler checkpoint, profile or source version differs"
        )
    kernels = [row for row in catalog["kernels"] if row["prepacked_checkpoint_weight"]]
    by_name = {row["checkpoint_parameter_name"]: row for row in kernels}
    if len(kernels) != 155 or len(by_name) != len(kernels):
        raise ValueError(
            "TinyLlama checkpoint matrix coverage is incomplete or ambiguous"
        )
    source_elf = port / "session.elf"
    source_sha = file_sha(source_elf)
    if source_sha != receipt["elf_sha256"]:
        raise ValueError("source ELF differs from its executed image")
    program_raw = (port / "phases.bin").read_bytes()
    program = np.frombuffer(program_raw, "<u4").reshape(-1, 32)
    if len(program) != receipt["phase_count"] or len(program) != 266:
        raise ValueError("source phase program has an incomplete extent")
    records = []
    covered = set()
    with (
        RV64PayloadImage(source_elf) as source,
        RV64PayloadImage(compiler / "model.elf") as target,
    ):
        regions = {}
        for label, region in receipt["regions"].items():
            size = region["blob_bytes"]
            symbol = "rk_region" + label
            if source.digest(symbol, size) != region["blob_sha256"]:
                raise ValueError(
                    "actual source region differs from its generated payload"
                )
            start, _ = source.span(symbol, size)
            regions[int(label)] = (region["virtual_base"], size, start)
        start, size, offset = regions[3]
        if source.image[offset : offset + len(program_raw)] != program_raw:
            raise ValueError(
                "actual source phase descriptors differ from the retained program"
            )

        def read(address, length):
            for base, size, offset in regions.values():
                if base <= address and address + length <= base + size:
                    return source.image[
                        offset + address - base : offset + address - base + length
                    ]
            raise ValueError("source matrix payload exceeds its linked region")

        for index, raw in enumerate(program):
            d = [int(x) for x in raw]
            if d[0] != 1 or d[1] & 255 in (4, 6):
                continue
            names = phase_parameters(d[1] >> 8, d[1] & 255)
            parts = [by_name[name] for name in names]
            k, n, tile_k = d[17] * d[19], d[16] * 64, d[19]
            if d[14] != 1 or d[20] or sum(row["dimensions"]["n"] for row in parts) != n:
                raise ValueError(
                    "source matrix grouping or fusion differs from its phase"
                )
            codes, scales = unpack_weight(
                read(d[6], k * n), read(d[7], k * n // 32), k=k, n=n, tile_k=tile_k
            )
            column = 0
            for name, kernel in zip(names, parts, strict=True):
                dims = kernel["dimensions"]
                if (
                    name in covered
                    or kernel["batched"]
                    or kernel["precision"] != "mxfp8"
                    or dims["k"] != k
                    or dims["m"] != d[15] * d[18]
                    or kernel["padded_shape_mnk"] != [dims["m"], dims["n"], dims["k"]]
                ):
                    raise ValueError("source/compiler matrix identity or shape differs")
                compared = {}
                for kind, values in (("codes", codes), ("scales", scales)):
                    blob = values[:, column : column + dims["n"]].tobytes()
                    digest = hashlib.sha256(blob).hexdigest()
                    if (
                        len(blob) != kernel[kind + "_bytes"]
                        or digest != kernel[kind + "_sha256"]
                        or target.read(kernel["symbol"] + "_fixed_" + kind, len(blob))
                        != blob
                    ):
                        raise ValueError(
                            "source/compiler encoded checkpoint differs: "
                            + name
                            + " "
                            + kind
                        )
                    compared[kind] = {"bytes": len(blob), "canonical_sha256": digest}
                records.append(
                    {
                        "checkpoint_parameter": name,
                        "source_phase": index,
                        "compiler_kernel": kernel["symbol"],
                        "shape_kn": [k, dims["n"]],
                        "source_column_offset": column,
                        "source_tile_mnk": [d[18], 64, tile_k],
                        "compiler_tile_mnk": kernel["tile_mnk"],
                        "payloads": compared,
                    }
                )
                covered.add(name)
                column += dims["n"]
    if covered != set(by_name):
        raise ValueError("source/compiler learned matrix coverage differs")
    return {
        "schema": "mx_gemmini.source_compiler_weight_equivalence.v1",
        "status": "all_encoded_checkpoint_matrices_equal",
        "source_elf_sha256": source_sha,
        "compiler_elf_sha256": file_sha(compiler / "model.elf"),
        "source_receipt_sha256": file_sha(port / "receipt.json"),
        "compiler_catalog_sha256": hashlib.sha256(catalog_raw).hexdigest(),
        "checkpoint_sha256": catalog["checkpoint_weights_sha256"],
        "matrices_checked": len(records),
        "scope": "actual learned FP8 codes and block32 scales after layout conversion; runtime activations excluded",
        "matrices": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-build", type=Path, required=True)
    parser.add_argument("--compiler-build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be fresh")
    result = audit(args.port_build, args.compiler_build)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "matrices_checked",
                    "source_elf_sha256",
                    "compiler_elf_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
