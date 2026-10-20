"""Check a recurrent Rocket smoke run with independent NumPy and hardware references.

Matrix results come from the pinned Radiance mesh reference and exponentials
from the pinned Rocket VPU reference. No target C kernel or compiler host graph
is used to produce the expected logits or cache values.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from tools.qualify_radiance_rocket_llama import parse_console, sha
from tools.replay_radiance_rocket_llama import Replay


class CachedReplay(Replay):
    def __init__(self, build, source, rocc, out):
        super().__init__(build, source, rocc, out, accepted_modes=("cached_smoke",))
        d = self.report["defines"]
        self.defs = d
        self.raw_cache = np.zeros(
            (2, d["LL_L"], d["LL_HKV"], self.report["cache_capacity"], d["LL_HD"]),
            dtype="<f4",
        )
        self.cos = np.fromfile(build / "cos.bin", "<f4").reshape(-1, d["LL_HD"] // 2)
        self.sin = np.fromfile(build / "sin.bin", "<f4").reshape(self.cos.shape)
        self.embeddings = np.fromfile(build / "embedding.bin", "<f4").reshape(
            d["LL_V"], d["LL_D"]
        )
        self.ids = np.fromfile(build / "token_ids.bin", "<i8")
        for filename in (
            "phases.bin",
            "cos.bin",
            "sin.bin",
            "embedding.bin",
            "token_ids.bin",
            "driver.c",
            "radiance_rocket_llama.c",
        ):
            if sha(build / filename) != self.report["files_sha256"][filename]:
                raise ValueError("retained recurrent input changed: " + filename)

    def store_a(self, codes, scales, values, tm, tk):
        rows = values.shape[0]
        if rows % tm:
            values = np.pad(values, ((0, (-rows) % tm), (0, 0)))
        super().store_a(codes, scales, values, tm, tk)

    def rope(self, d):
        nqkv, hq, hkv, hd = d[3:7]
        p0, rows, keys, tm = d[21:25]
        layer, group = d[1] >> 8, hq // hkv
        x = self.f32(d[2], rows * nqkv).reshape(rows, nqkv)
        cos, sin = self.cos[p0 : p0 + rows], self.sin[p0 : p0 + rows]
        q = self.generator.rope(x[:, : hq * hd].reshape(rows, hq, hd), cos, sin)
        kn = self.generator.rope(
            x[:, hq * hd : (hq + hkv) * hd].reshape(rows, hkv, hd), cos, sin
        )
        vn = x[:, (hq + hkv) * hd :].reshape(rows, hkv, hd)
        self.raw_cache[0, layer, :, p0 : p0 + rows] = kn.transpose(1, 0, 2)
        self.raw_cache[1, layer, :, p0 : p0 + rows] = vn.transpose(1, 0, 2)
        for h in range(hkv):
            stacked = (
                q[:, h * group : (h + 1) * group]
                .transpose(1, 0, 2)
                .reshape(group * rows, hd)
            )
            self.store_a(d[9] + h * d[11], d[10] + h * d[12], stacked, tm, hd)
            # Reconstruct the entire packed cache from its unquantized tensor.
            # This deliberately differs from the target's incremental byte writes.
            k = np.zeros((keys, hd), np.float32)
            v = np.zeros((keys, hd), np.float32)
            k[: self.raw_cache.shape[3]] = self.raw_cache[0, layer, h]
            v[: self.raw_cache.shape[3]] = self.raw_cache[1, layer, h]
            kc, ksc = self.quant(k)
            vc, vsc = self.quant(v.T)
            kb, ks = self.generator.pack_b(kc.T, ksc.T, hd)
            vb, vs = self.generator.pack_b(vc.T, vsc.T, 256)
            self.write(d[13] + h * d[15], kb)
            self.write(d[14] + h * d[16], ks)
            self.write(d[17] + h * d[19], vb)
            self.write(d[18] + h * d[20], vs)

    def execute_session(self):
        np_ = self.defs["LL_NP"]
        prefix = self.program[:np_]
        for words in prefix:
            d = list(map(int, words))
            if d[0] == 3:
                for head in range(self.defs["LL_HKV"]):
                    for address, stride, byte in [
                        (d[13], d[15], 0),
                        (d[14], d[16], 104),
                        (d[17], d[19], 0),
                        (d[18], d[20], 104),
                    ]:
                        self.view(address + head * stride, stride)[:] = byte
        logits = []
        for step in range(self.report["session_steps"]):
            program = self.program[step * np_ : (step + 1) * np_]
            if step:
                embedding = self.generator.bf16_bits(
                    self.embeddings[self.ids[self.defs["LL_M"] + step - 1]]
                )
                self.write(program[0, 2], embedding)
            for words in program:
                d = list(map(int, words))
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
                    raise ValueError("unexpected phase in recurrent replay")
            address = self.defs["LL_LOGITS_ADDR"]
            if step == 0:
                address += (self.defs["LL_M"] - 1) * self.defs["LL_V"] * 2
            logits.append(self.f32(address, self.defs["LL_V"]).copy())
            print(
                f"reference recurrent step {step + 1}/{self.report['session_steps']}",
                flush=True,
            )
        return np.stack(logits), self.raw_cache


def main():
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "radiance-root", "rocc-root", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("output must be fresh")
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    shutil.copyfile(Path(__file__), out / "producer.py")
    engine = CachedReplay(args.build, args.radiance_root, args.rocc_root, out)
    console = (args.build / "spike.log").read_text()
    checks = json.loads((args.build / "checks.json").read_text())
    cycles, _, observed = parse_console(
        console, checks, engine.report["phase_count"], False
    )
    resets = re.findall(r"^CACHED_RESET ([1-9][0-9]*)$", console, re.MULTILINE)
    steps = [
        tuple(map(int, row))
        for row in re.findall(
            r"^CACHED_STEP (\d+) (\d+) ([1-9][0-9]*)$", console, re.MULTILINE
        )
    ]
    if len(resets) != 1 or len(steps) != engine.report["session_steps"]:
        raise ValueError("incomplete recurrent timing sequence")
    if [(s, p) for s, p, _ in steps] != [
        (s, engine.defs["LL_M"] + s) for s in range(len(steps))
    ]:
        raise ValueError("recurrent steps or positions differ")
    if cycles != int(resets[0]) + sum(t for _, _, t in steps):
        raise ValueError("inclusive recurrent timing does not sum")
    expected = engine.execute_session()
    results = []
    for i, (check, array) in enumerate(zip(checks, expected, strict=True)):
        array.astype("<f4").tofile(out / f"expected{i}.bin")
        raw = array.astype("<f4").tobytes()
        actual = observed[i]
        results.append(
            {
                "name": check["name"],
                "bytes": len(raw),
                "byte_mismatches": sum(
                    a != b for a, b in zip(actual, raw, strict=True)
                ),
                "word_mismatches": int(
                    np.count_nonzero(
                        np.frombuffer(actual, "<u4") != np.frombuffer(raw, "<u4")
                    )
                ),
                "expected_sha256": sha(out / f"expected{i}.bin"),
            }
        )
    report = {
        "schema": "mx_gemmini.radiance_rocket_cached_replay.v1",
        "source_receipt_sha256": sha(args.build / "receipt.json"),
        "producer_sha256": sha(Path(__file__)),
        "vpu_reference_header_sha256": engine.vpu_header_sha256,
        "matrix_reference_calls": engine.matrix_calls,
        "session_steps": len(steps),
        "cycles": cycles,
        "results": results,
        "passed": all(row["word_mismatches"] == 0 for row in results),
        "full_model_comparison": False,
    }
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
