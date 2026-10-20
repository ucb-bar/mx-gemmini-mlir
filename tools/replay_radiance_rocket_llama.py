"""Independently replay a completed smoke program and check every observed byte.

NumPy implements the source row arithmetic and layouts. Radiance's mesh
reference supplies matrix results; the pinned Rocket VPU reference supplies
BF16 exponential results. This checks the port's numerical execution, without
using its C kernels or its idealized NumPy model as the expected output.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools.qualify_radiance_rocket_llama import checked_sources, parse_console, sha

VPU_REFERENCE_SHA256 = (
    "459726e52e7d19c51bada75b2e3ded16f64371799d5bc443750eb22e8616b17b"
)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Replay:
    def __init__(self, build, source, rocc, out, *, accepted_modes=("random_smoke",)):
        self.build, self.out = build, out
        checked_sources(source)
        self.report = json.loads((build / "receipt.json").read_text())
        if (
            self.report["status"] != "completed_requires_matched_comparison"
            or self.report["mode"] not in accepted_modes
        ):
            raise ValueError("replay expects a completed source smoke program")
        self.generator = module(
            "replay_radiance_data", source / "kernels/nightly/llama_e2e/gen_data.py"
        )
        golden = out / "golden"
        golden.mkdir()
        for name in ("golden.py", "mx_golden.cpp", "mx_fp_math.h", "Makefile"):
            shutil.copyfile(source / "lib/golden" / name, golden / name)
        self.golden = module("replay_mesh_golden", golden / "golden.py")
        self.memory = {}
        for region, info in self.report["regions"].items():
            body = (build / f"data/llama.region{region}.bin").read_bytes()
            if sha(build / f"data/llama.region{region}.bin") != info["blob_sha256"]:
                raise ValueError("source model data changed")
            self.memory[info["virtual_base"]] = bytearray(body) + bytearray(
                info["bytes"] - len(body)
            )
        self.program = np.fromfile(build / "phases.bin", dtype="<u4").reshape(-1, 32)
        self.matrix_calls = 0
        self.exp_elements = 0
        self.round_quant_inputs = (
            "#define RK_ROUND_QUANT_INPUTS 1" in (build / "rocket_data.h").read_text()
        )
        vpu_header = rocc / "include/vpu_ref.h"
        self.vpu_header_sha256 = sha(vpu_header)
        if self.vpu_header_sha256 != VPU_REFERENCE_SHA256:
            raise ValueError(
                "Rocket VPU reference differs from the pinned target source"
            )
        shutil.copyfile(vpu_header, out / "vpu_ref.h")
        wrapper = out / "vpu_reference.c"
        wrapper.write_text(
            '#include "vpu_ref.h"\nvoid reference_exp(const uint16_t *a, uint16_t *b, uint32_t n) {\n'
            "  for (uint32_t i=0; i<n; i++) b[i]=vpu_exp(a[i]);\n}\n"
        )
        cc = shutil.which("cc")
        if cc is None:
            raise RuntimeError("host C compiler unavailable")
        self.host_cc_sha256 = sha(cc)
        command = [
            cc,
            "-shared",
            "-fPIC",
            "-O2",
            "-fno-fast-math",
            str(wrapper),
            "-o",
            str(out / "vpu_reference.so"),
        ]
        with (out / "compile_vpu.log").open("wb") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        self.reference_command = command
        self.library = ctypes.CDLL(str(out / "vpu_reference.so"))
        self.library.reference_exp.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_uint32,
        ]

    def view(self, address, count, dtype="u1"):
        address, count = int(address), int(count)
        dtype = np.dtype(dtype)
        for base, memory in self.memory.items():
            offset = address - base
            if 0 <= offset and offset + count * dtype.itemsize <= len(memory):
                return np.frombuffer(memory, dtype=dtype, count=count, offset=offset)
        raise ValueError("reference memory access exceeds source region")

    def write(self, address, array):
        raw = np.ascontiguousarray(array).view("u1").reshape(-1)
        self.view(address, len(raw))[:] = raw

    def f32(self, address, count):
        return self.generator.bf16_to_f32(self.view(address, count, "<u2"))

    def quant(self, values):
        values = np.asarray(values, dtype=np.float32)
        if not self.round_quant_inputs:
            return self.generator.mx_quant(values)
        values = self.generator.bf(values)
        groups = values.reshape(values.shape[:-1] + (values.shape[-1] // 32, 32))
        exponent = np.maximum(
            self.generator.block_exp(np.max(np.abs(groups), axis=-1)), -23
        )
        scaled = np.ldexp(groups, -exponent[..., None].astype(np.int32)).astype(
            np.float32
        )
        return self.generator.fp8_encode(scaled).reshape(values.shape), (
            exponent + 127
        ).astype(np.uint8)

    def exp(self, values):
        words = np.ascontiguousarray(self.generator.bf16_bits(values))
        result = np.empty_like(words)
        self.library.reference_exp(
            words.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            words.size,
        )
        self.exp_elements += words.size
        return self.generator.bf16_to_f32(result)

    @staticmethod
    def lane_sum(pairs):
        rows, width = pairs.shape
        partial = np.zeros((rows, 16), np.float32)
        for offset in range(0, width, 16):
            partial += pairs[:, offset : offset + 16]
        total = np.zeros(rows, np.float32)
        for lane in range(16):
            total += partial[:, lane]
        return total

    def a(self, codes, scales, rows, k, tm, tk):
        mt, ks = rows // tm, k // tk
        values = (
            self.view(codes, rows * k)
            .reshape(mt, ks, tm, tk)
            .transpose(0, 2, 1, 3)
            .reshape(rows, k)
        )
        sf = (
            self.view(scales, rows * k // 32)
            .reshape(mt, ks, tk // 32, tm)
            .transpose(0, 3, 1, 2)
            .reshape(rows, k // 32)
        )
        return values, sf

    def b(self, codes, scales, k, n, tk):
        ks, nt = k // tk, n // 64
        values = (
            self.view(codes, k * n)
            .reshape(ks, nt, tk, 64)
            .transpose(0, 2, 1, 3)
            .reshape(k, n)
        )
        sf = (
            self.view(scales, k * n // 32)
            .reshape(nt, ks, tk // 32, 64)
            .transpose(1, 2, 0, 3)
            .reshape(k // 32, n)
        )
        return values, sf

    def store_a(self, codes, scales, values, tm, tk):
        packed, sf = self.generator.pack_a(*self.quant(values), tm, tk)
        self.write(codes, packed)
        self.write(scales, sf)

    def gemm(self, d):
        tm, tk, mt, nt, ks = d[18], d[19], d[15], d[16], d[17]
        rows, n, k = mt * tm, nt * 64, ks * tk
        for batch in range(d[14]):
            a, asc = self.a(d[2] + batch * d[4], d[3] + batch * d[5], rows, k, tm, tk)
            b, bsc = self.b(d[6] + batch * d[8], d[7] + batch * d[9], k, n, tk)
            result = self.golden.mx_matmul(
                a,
                b,
                asc.T,
                bsc,
                rows,
                n,
                k,
                tmpdir=str(self.out / "matrix_inputs" / str(self.matrix_calls)),
            )
            self.matrix_calls += 1
            for m in range(mt):
                for row in range(tm):
                    self.write(
                        d[10] + batch * d[11] + m * d[13] + row * d[12],
                        result[m * tm + row],
                    )

    def prenorm(self, d):
        width, rows, tm, tk = d[8:12]
        h = self.generator.bf(
            self.f32(d[2], rows * width).reshape(rows, width)
            + self.f32(d[3], rows * width).reshape(rows, width)
        )
        self.write(d[4], self.generator.bf16_bits(h))
        lo, hi = h[:, 0::2], h[:, 1::2]
        pairs = lo * lo + hi * hi
        total = self.lane_sum(pairs)
        eps = np.asarray(d[12], dtype="<u4").view("<f4").item()
        inv = np.float32(1) / np.sqrt(total / np.float32(width) + np.float32(eps))
        gamma = self.f32(d[5], width)
        self.store_a(d[6], d[7], (h * inv[:, None]) * gamma, tm, tk)

    def ewq(self, d, silu):
        stride, width, rows, tm, tk = d[3], d[5], d[6], d[9], d[10]
        source = self.f32(d[2], rows * stride).reshape(rows, stride)
        values = source[:, :width]
        if silu:
            values = (values / (np.float32(1) + self.exp(-values))) * source[
                :, width : 2 * width
            ]
        self.store_a(d[7], d[8], values, tm, tk)

    def rope(self, d):
        nqkv, hq, hkv, hd = d[3:7]
        p0, rows, keys, tm = d[21:25]
        group = hq // hkv
        x = self.f32(d[2], rows * nqkv).reshape(rows, nqkv)
        cos = self.view(d[7], rows * hd // 2, "<f4").reshape(rows, hd // 2)
        sin = self.view(d[8], rows * hd // 2, "<f4").reshape(rows, hd // 2)
        q = self.generator.rope(x[:, : hq * hd].reshape(rows, hq, hd), cos, sin)
        kn = self.generator.rope(
            x[:, hq * hd : (hq + hkv) * hd].reshape(rows, hkv, hd), cos, sin
        )
        vn = x[:, (hq + hkv) * hd :].reshape(rows, hkv, hd)
        for h in range(hkv):
            stacked = (
                q[:, h * group : (h + 1) * group]
                .transpose(1, 0, 2)
                .reshape(group * rows, hd)
            )
            self.store_a(d[9] + h * d[11], d[10] + h * d[12], stacked, tm, hd)
            kc, ksc = self.b(d[13] + h * d[15], d[14] + h * d[16], hd, keys, hd)
            kc, ksc = kc.copy(), ksc.copy()
            new_k, new_ks = self.quant(kn[:, h])
            kc[:, p0 : p0 + rows], ksc[:, p0 : p0 + rows] = new_k.T, new_ks.T
            kb, ks = self.generator.pack_b(kc, ksc, hd)
            self.write(d[13] + h * d[15], kb)
            self.write(d[14] + h * d[16], ks)
            vc, vsc = self.b(d[17] + h * d[19], d[18] + h * d[20], keys, hd, 256)
            vc, vsc = vc.copy(), vsc.copy()
            new_v, new_vs = self.quant(vn[:, h].T)
            vc[p0 : p0 + rows], vsc[p0 // 32 : (p0 + rows) // 32] = new_v.T, new_vs.T
            vb, vs = self.generator.pack_b(vc, vsc, 256)
            self.write(d[17] + h * d[19], vb)
            self.write(d[18] + h * d[20], vs)

    def softmax(self, d):
        keys, rows, batch, p0, chunk, tm, tk = (
            d[4],
            d[5],
            d[6],
            d[12],
            d[13],
            d[14],
            d[15],
        )
        scale = np.asarray(d[11], dtype="<u4").view("<f4").item()
        valid = np.arange(keys)[None, :] <= (p0 + np.arange(rows) % chunk)[:, None]
        for b in range(batch):
            source = self.f32(d[2] + b * d[3], rows * keys).reshape(rows, keys)
            z = source * np.float32(scale)
            maximum = np.where(valid, z, -np.inf).max(axis=1)
            e = np.where(valid, self.exp(z - maximum[:, None]), np.float32(0))
            total = self.lane_sum(e[:, 0::2] + e[:, 1::2])
            probabilities = e * (np.float32(1) / total)[:, None]
            self.store_a(d[7] + b * d[9], d[8] + b * d[10], probabilities, tm, tk)

    def execute(self):
        for phase, words in enumerate(self.program):
            d = [int(v) for v in words]
            if d[0] == 1:
                self.gemm(d)
            elif d[0] == 2:
                self.prenorm(d)
            elif d[0] == 3:
                self.rope(d)
            elif d[0] == 4:
                self.softmax(d)
            elif d[0] in (5, 7):
                self.ewq(d, d[0] == 7)
            else:
                raise ValueError(f"reference has no phase {d[0]}")
            print(f"reference phase {phase + 1}/{len(self.program)}", flush=True)


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "radiance-root", "rocc-root", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("--out-dir must be fresh")
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    shutil.copyfile(Path(__file__), out / "producer.py")
    engine = Replay(args.build, args.radiance_root, args.rocc_root, out)
    checks = json.loads((args.build / "checks.json").read_text())
    _, _, observed = parse_console(
        (args.build / "spike.log").read_text(),
        checks,
        engine.report["phase_count"],
        engine.report["phase_timing"],
    )
    engine.execute()
    results = []
    for i, check in enumerate(checks):
        expected = engine.view(check["address"], check["bytes"]).tobytes()
        actual = observed[i]
        (out / f"expected{i}.bin").write_bytes(expected)
        results.append(
            {
                "name": check["name"],
                "bytes": len(expected),
                "byte_mismatches": sum(
                    a != b for a, b in zip(actual, expected, strict=True)
                ),
                "expected_sha256": sha(out / f"expected{i}.bin"),
            }
        )
    report = {
        "schema": "mx_gemmini.radiance_rocket_reference_replay.v1",
        "scope": "complete smoke model; exact mesh and Rocket VPU arithmetic",
        "source_receipt_sha256": sha(args.build / "receipt.json"),
        "producer_sha256": sha(Path(__file__)),
        "phase_count": len(engine.program),
        "matrix_reference_calls": engine.matrix_calls,
        "vpu_exp_elements": engine.exp_elements,
        "vpu_reference_header_sha256": engine.vpu_header_sha256,
        "host_compiler_sha256": engine.host_cc_sha256,
        "host_compile_command": engine.reference_command,
        "results": results,
        "passed": all(row["byte_mismatches"] == 0 for row in results),
    }
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
