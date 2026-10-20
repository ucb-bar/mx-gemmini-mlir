#!/usr/bin/env python3
"""Reopen retained target observations and regenerate independent numerical oracles."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unpack(destination):
    manifest = json.loads((HERE / "artifacts.json").read_text())
    archive = HERE / manifest["archive"]["path"]
    assert archive.stat().st_size == manifest["archive"]["bytes"]
    assert sha(archive.read_bytes()) == manifest["archive"]["sha256"]
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:xz") as tar:
        names = tar.getnames()
        assert len(names) == len(set(names)) and set(names) == set(manifest["members"])
        for member in tar:
            assert member.isfile() and not Path(member.name).is_absolute()
            assert ".." not in Path(member.name).parts
            data = tar.extractfile(member).read()
            expected = manifest["members"][member.name]
            assert len(data) == expected["bytes"] and sha(data) == expected["sha256"]
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    profile_bytes = (destination / "common/profile.json").read_bytes()
    assert sha(profile_bytes) == manifest["tools_sha256"]["profile"]
    canonical = (
        json.dumps(json.loads(profile_bytes), sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    assert sha(canonical) == manifest["profile_semantic_sha256"]
    assert (
        sha(canonical)
        == "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"
    )
    for name in manifest["probes"]:
        folder = destination / name
        status = json.loads((folder / "status.json").read_text())
        admission = json.loads((folder / "admission.json").read_text())
        report = json.loads((folder / "admission_report.json").read_text())
        assert status["exit_code"] == 0 and admission["status"] == "accepted"
        assert (
            status["elf_sha256"]
            == admission["elf_sha256"]
            == sha((folder / "pair.elf").read_bytes())
        )
        assert (
            sha((folder / "admission_report.json").read_bytes())
            == admission["report_sha256"]
        )
        assert report["elf_sha256"] == admission["elf_sha256"]
        assert (
            report["scope"]
            == "structural: digest and declared ISA; no instruction policy"
        )
    return manifest


ORACLE = r"""
#include "vpu_ref.h"
#include <math.h>
void softmax_reference(const float *input,unsigned rows,unsigned width,float *output) {
 for(unsigned row=0;row<rows;row++) {
  float maximum=-INFINITY;
  for(unsigned col=0;col<width;col++)if(input[row*width+col]>maximum)maximum=input[row*width+col];
  float sum=0.0f;
  for(unsigned col=0;col<width;col++) {
   float value=vpu_bf16_to_f(vpu_exp(vpu_f_to_bf16(input[row*width+col]-maximum)));
   output[row*width+col]=value;sum+=value;
  }
  for(unsigned col=0;col<width;col++)output[row*width+col]/=sum;
 }
}
void rms_reference(const float *input,const float *gamma,float *output) {
 for(unsigned row=0;row<160;row++) {
  uint16_t square[256][8],sum[1][8];
  for(unsigned col=0;col<2048;col++) {
   uint16_t x=vpu_f_to_bf16(input[row*2048+col]);square[col/8][col%8]=vpu_mul(x,x);
  }
  vpu_ref_exec(VPU_RSUM,sum,square,0,256,256,0,0);
  uint16_t mean=vpu_mul(sum[0][0],vpu_rcp(vpu_f_to_bf16(2048.0f)));
  uint16_t inverse=vpu_rsqrt(vpu_add(mean,vpu_f_to_bf16(1e-5f)));
  for(unsigned col=0;col<2048;col++)output[row*2048+col]=vpu_bf16_to_f(
   vpu_mul(vpu_mul(vpu_f_to_bf16(input[row*2048+col]),inverse),vpu_f_to_bf16(gamma[col])));
 }
}
void sigmoid_reference(const float *input,unsigned count,float *old,float *hybrid) {
 for(unsigned i=0;i<count;i++) {
  uint16_t a=vpu_f_to_bf16(input[i]);
  old[i]=vpu_bf16_to_f(vpu_rcp(vpu_add(vpu_exp(vpu_mul(a,0xbf80)),0x3f80)));
  hybrid[i]=1.0f/(1.0f+vpu_bf16_to_f(vpu_exp(vpu_f_to_bf16(-input[i]))));
 }
}
"""


def oracle(destination):
    header = destination / "common/vpu_ref.h"
    assert (
        sha(header.read_bytes())
        == "459726e52e7d19c51bada75b2e3ded16f64371799d5bc443750eb22e8616b17b"
    )
    source = destination / "reference.c"
    source.write_text(ORACLE)
    library = destination / "reference.so"
    subprocess.run(
        [
            "cc",
            "-O2",
            "-shared",
            "-fPIC",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-DMX_QUANT_REFERENCE",
            "-I",
            str(destination / "common"),
            str(source),
            str(destination / "fp8_pack/old_quant.c"),
            "-lm",
            "-o",
            str(library),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(library))
    pointer, unsigned = ctypes.c_void_p, ctypes.c_uint32
    dll.mx_pack_activation_f32.argtypes = [
        pointer,
        unsigned,
        unsigned,
        unsigned,
        pointer,
        pointer,
    ]
    dll.softmax_reference.argtypes = [pointer, unsigned, unsigned, pointer]
    dll.rms_reference.argtypes = [pointer] * 3
    dll.sigmoid_reference.argtypes = [pointer, unsigned, pointer, pointer]
    return dll


def packed(dll, values, rows, width):
    values = np.ascontiguousarray(values, dtype="<f4")
    codes = np.empty(rows * width, np.uint8)
    scales = np.empty(rows * width // 32, np.uint8)
    assert (
        dll.mx_pack_activation_f32(
            values.ctypes.data, rows, width, 8, codes.ctypes.data, scales.ctypes.data
        )
        == 0
    )
    return codes.tobytes() + scales.tobytes()


def blocks(log, prefix, lengths, dtype="<u4"):
    result = {key: np.empty(count, dtype) for key, count in lengths.items()}
    positions = dict.fromkeys(lengths, 0)
    key_length = len(next(iter(lengths)))
    for line in log.splitlines():
        if not line.startswith(prefix + " "):
            continue
        fields = line.split()
        key = tuple(int(x) for x in fields[1 : 1 + key_length])
        assert key in result
        offset, count = map(int, fields[1 + key_length : 3 + key_length])
        assert offset == positions[key] and count > 0 and offset + count <= lengths[key]
        values = fields[3 + key_length :]
        assert len(values) == count
        result[key][offset : offset + count] = np.fromiter(
            (int(x, 16) for x in values), dtype, count
        )
        positions[key] += count
    assert positions == lengths
    return result


def records(log, prefix, checked=None):
    found = []
    for line in log.splitlines():
        if not line.startswith(prefix + " "):
            continue
        fields = dict(re.findall(r"([a-z_]+)=(\d+)", line))
        row = {key: int(value) for key, value in fields.items()}
        assert row["bad"] == 0 and row["cycles"] > 0
        if checked is not None:
            assert row["checked"] == checked
        found.append(row)
    assert found and prefix + "_DONE" in log
    return found


def paired(rows, trials=2):
    assert [(r["trial"], r["role"]) for r in rows] == [
        (trial, trial % 2 ^ order) for trial in range(trials) for order in range(2)
    ]
    old = [r["cycles"] for r in rows if r["role"] == 0]
    new = [r["cycles"] for r in rows if r["role"] == 1]
    return {
        "control_cycles": old,
        "candidate_cycles": new,
        "mean_control_over_candidate": float(np.mean(old) / np.mean(new)),
    }


def verify(destination):
    manifest = unpack(destination)
    dll = oracle(destination)
    summary = {"scope": manifest["scope"], "probes": {}}
    results = summary["probes"]
    for name in manifest["probes"]:
        p = destination / name
        log = (p / "spike.log").read_text()
        if name == "fp8_pack":
            values = np.fromfile(p / "input.bin", "<f4")
            expected = packed(dll, values, 160, 5632)
            assert (
                expected
                == (p / "codes.bin").read_bytes() + (p / "scales.bin").read_bytes()
            )
            elf = (p / "pair.elf").read_bytes()
            assert (p / "input.bin").read_bytes() in elf
            assert (p / "codes.bin").read_bytes() in elf and (
                p / "scales.bin"
            ).read_bytes() in elf
            results[name] = paired(records(log, "QUANT", len(expected)), trials=4)
            results[name]["observed_check"] = (
                "target checks every code/scale byte and guards; no raw output dump"
            )
        elif name == "rms_pack":
            x = np.fromfile(p / "input.bin", "<f4")
            gamma = np.fromfile(p / "gamma.bin", "<f4")
            norm = np.empty_like(x)
            dll.rms_reference(x.ctypes.data, gamma.ctypes.data, norm.ctypes.data)
            expected = packed(dll, norm, 160, 2048)
            assert expected == (p / "expected.bin").read_bytes()
            elf = (p / "pair.elf").read_bytes()
            assert expected in elf and x.tobytes() in elf and gamma.tobytes() in elf
            results[name] = paired(records(log, "RMS", len(expected)))
            results[name]["observed_check"] = (
                "target checks every code/scale byte and guards; no raw output dump"
            )
        elif name == "hybrid_softmax":
            values = np.fromfile(p / "input.bin", "<f4")
            expected = np.empty_like(values)
            dll.softmax_reference(values.ctypes.data, 5120, 160, expected.ctypes.data)
            assert expected.tobytes() == (p / "expected.bin").read_bytes()
            observed = blocks(log, "SOFTMAX_WORDS", {(): expected.size})[()]
            assert np.array_equal(observed, expected.view(np.uint32))
            # This standalone run has one candidate and no separately timed control.
            row = re.findall(r"HYBRID cycles=(\d+) bad=0 checked=819200", log)
            assert len(row) == 1 and "HYBRID_DONE" in log
            results[name] = {"cycles": int(row[0]), "all_output_words": expected.size}
        elif name == "qkv_group":
            rows = records(log, "PROJECTION", 409600)
            columns = (2048, 256, 256)
            expected = []
            a_codes = np.repeat(
                np.array([0x38, 0x30, 0x28], np.uint8)[np.arange(160) % 3, None],
                2048,
                axis=1,
            )
            a_scales = np.tile((127 - np.arange(160) % 2).astype(np.uint8), (64, 1))
            assert (
                p / "input.bin"
            ).read_bytes() == a_codes.tobytes() + a_scales.tobytes()
            a = np.array([1, 0.5, 0.25], np.float32)[np.arange(160) % 3] * np.exp2(
                -(np.arange(160) % 2)
            )
            for part, width in enumerate(columns):
                codebook = ((0x38, 0x30, 0x28), (0x40, 0x38, 0x30), (0xB8, 0xB0, 0xA8))[
                    part
                ]
                weight = np.tile(
                    np.array(codebook, np.uint8)[np.arange(width) % 3], (2048, 1)
                )
                scale = np.tile((127 - np.arange(width) % 2).astype(np.uint8), (64, 1))
                assert (p / f"weight{part}.bin").read_bytes() == weight.tobytes()
                assert (p / f"scales{part}.bin").read_bytes() == scale.tobytes()
                b = np.array([1, 0.5, 0.25], np.float32)[
                    np.arange(width) % 3
                ] * np.exp2(-(np.arange(width) % 2))
                value = (2048 * a[:, None] * b[None, :] * (1, 2, -1)[part]).astype(
                    "<f4"
                )
                assert value.tobytes() == (p / f"expected{part}.bin").read_bytes()
                expected.append(value.ravel().view(np.uint32))
            sizes = {
                (trial, role, part): 160 * columns[part]
                for trial in range(2)
                for role in range(2)
                for part in range(3)
            }
            observed = blocks(log, "PROJECTION_WORDS", sizes)
            for key, value in observed.items():
                assert np.array_equal(value, expected[key[2]])
            results[name] = paired(rows)
            results[name]["all_matrix_words_per_call"] = 409600
        elif name.startswith("softmax_pack_"):
            rows = records(log, "SOFTMAX_PACK")
            assert all(
                r["packed"] == 844800 and r["matrix_words"] == 327680 for r in rows
            )
            scores = np.fromfile(p / "scores.bin", "<f4")
            probability = np.empty_like(scores)
            dll.softmax_reference(
                scores.ctypes.data, 5120, 160, probability.ctypes.data
            )
            expected = b"".join(
                packed(dll, probability[b * 25600 : (b + 1) * 25600], 160, 160)
                for b in range(32)
            )
            assert expected == (p / "expected.bin").read_bytes()
            assert probability.tobytes() == (p / "probability.bin").read_bytes()
            sizes = {(t, r): 844800 for t in range(2) for r in range(2)}
            observations = blocks(log, "PACKED_BYTES", sizes, "u1")
            for value in observations.values():
                assert value.tobytes() == expected
            matrices = blocks(log, "MATRIX_WORDS", dict.fromkeys(sizes, 327680))
            control = matrices[(0, 0)]
            for value in matrices.values():
                assert np.array_equal(value, control)
            results[name] = paired(rows)
            results[name]["observed_check"] = (
                "independent packed bytes; every matrix word equals actual unfused control"
            )
        elif name == "hybrid_sigmoid":
            values = np.fromfile(p / "input.bin", "<f4")
            old, new = np.empty_like(values), np.empty_like(values)
            dll.sigmoid_reference(
                values.ctypes.data, values.size, old.ctypes.data, new.ctypes.data
            )
            assert old.tobytes() == (p / "old_expected.bin").read_bytes()
            assert new.tobytes() == (p / "hybrid_expected.bin").read_bytes()
            sizes = {(t, r): values.size for t in range(2) for r in range(2)}
            observations = blocks(log, "SIGMOID_WORDS", sizes)
            for key, value in observations.items():
                assert np.array_equal(value, (new if key[1] else old).view(np.uint32))
            results[name] = paired(records(log, "SIGMOID", values.size))
            results[name]["arithmetic"] = (
                "candidate changes intermediate precision; not a bit-equivalence claim"
            )
        else:
            raise ValueError("unknown retained probe")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir", type=Path, required=True, help="new extraction/build directory"
    )
    parser.add_argument("--out", type=Path, help="optional JSON verification report")
    args = parser.parse_args()
    result = verify(args.workdir.resolve())
    expected = HERE / "verification.json"
    if expected.exists():
        assert result == json.loads(expected.read_text())
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
