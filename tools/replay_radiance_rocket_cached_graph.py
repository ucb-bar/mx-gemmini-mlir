"""Check recurrent model semantics without interpreting the decoder descriptors.

Reads weights, norm parameters and prefix inputs from the original source
prefill's bindings. The Llama graph below operates on logical tensors. It uses
the pinned mesh and VPU arithmetic references, with independent cache updates
and attention-head assembly, to check the port's scheduler as well as kernels.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from tools.qualify_radiance_rocket_llama import parse_console, sha
from tools.replay_radiance_rocket_cached import CachedReplay


class LogicalGraph(CachedReplay):
    def __init__(self, build, source, rocc, out):
        super().__init__(build, source, rocc, out)
        d = self.defs
        self.width, self.ff, self.hq, self.hkv, self.hd = (
            d[k] for k in ("LL_D", "LL_F", "LL_HQ", "LL_HKV", "LL_HD")
        )
        self.weights, self.gamma = {}, {}
        # No descriptor from a single-token decode enters this model graph.
        self.prefix = self.program[: d["LL_NP"]]
        for words in self.prefix:
            p = list(map(int, words))
            layer, op = p[1] >> 8, p[1] & 255
            if p[0] == 2:
                self.gamma[layer, op] = self.f32(p[5], self.width).copy()
            if p[0] != 1 or op in (4, 6):
                continue
            k, n = {
                2: (self.width, (self.hq + 2 * self.hkv) * self.hd),
                8: (self.hq * self.hd, self.width),
                10: (self.width, 2 * self.ff),
                12: (self.ff, self.width),
                14: (self.width, d["LL_V"]),
            }[op]
            if p[17] * p[19] != k or p[16] * 64 != n or p[14] != 1 or p[20]:
                raise ValueError(
                    "source learned parameter binding differs from Llama geometry"
                )
            codes, scales = self.b(p[6], p[7], k, n, p[19])
            self.weights[layer, op] = codes.copy(), scales.copy()

    def matrix(self, values, rhs):
        rows, k = values.shape
        codes, scales = self.quant(values)
        if rows % 16:
            padded = np.pad(values, ((0, (-rows) % 16), (0, 0)))
            codes, scales = self.quant(padded)
        b, bs = rhs
        n = b.shape[1]
        if b.shape[0] != k:
            raise ValueError("logical matrix contraction extents differ")
        words = self.golden.mx_matmul(
            codes,
            b,
            scales.T,
            bs,
            codes.shape[0],
            n,
            k,
            tmpdir=str(self.out / "matrix_inputs" / str(self.matrix_calls)),
        )
        self.matrix_calls += 1
        return self.generator.bf16_to_f32(words[:rows])

    def norm(self, h, gamma):
        total = self.lane_sum(h[:, 0::2] ** 2 + h[:, 1::2] ** 2)
        inv = np.float32(1) / np.sqrt(total / np.float32(self.width) + np.float32(1e-5))
        return (h * inv[:, None]) * gamma

    def attention(self, qkv, layer, position):
        rows = qkv.shape[0]
        hq, hkv, hd = self.hq, self.hkv, self.hd
        group, keys = hq // hkv, self.defs["LL_SK"]
        cos, sin = (
            self.cos[position : position + rows],
            self.sin[position : position + rows],
        )
        q = self.generator.rope(qkv[:, : hq * hd].reshape(rows, hq, hd), cos, sin)
        k = self.generator.rope(
            qkv[:, hq * hd : (hq + hkv) * hd].reshape(rows, hkv, hd), cos, sin
        )
        v = qkv[:, (hq + hkv) * hd :].reshape(rows, hkv, hd)
        self.raw_cache[0, layer, :, position : position + rows] = k.transpose(1, 0, 2)
        self.raw_cache[1, layer, :, position : position + rows] = v.transpose(1, 0, 2)
        valid = (
            np.arange(keys)[None, :]
            <= (position + np.tile(np.arange(rows), group))[:, None]
        )
        outputs = []
        for head in range(hkv):
            live_k = np.zeros((keys, hd), np.float32)
            live_v = np.zeros((keys, hd), np.float32)
            capacity = self.raw_cache.shape[3]
            live_k[:capacity] = self.raw_cache[0, layer, head]
            live_v[:capacity] = self.raw_cache[1, layer, head]
            kc, ks = self.quant(live_k)
            vc, vs = self.quant(live_v.T)
            queries = (
                q[:, head * group : (head + 1) * group]
                .transpose(1, 0, 2)
                .reshape(group * rows, hd)
            )
            scores = self.matrix(queries, (kc.T, ks.T)) * np.float32(1 / np.sqrt(hd))
            maximum = np.where(valid, scores, -np.inf).max(axis=1)
            exp = np.where(valid, self.exp(scores - maximum[:, None]), np.float32(0))
            total = self.lane_sum(exp[:, 0::2] + exp[:, 1::2])
            probabilities = exp * (np.float32(1) / total)[:, None]
            result = self.matrix(probabilities, (vc.T, vs.T))
            outputs.append(result.reshape(group, rows, hd).transpose(1, 0, 2))
        return np.concatenate(outputs, axis=1).reshape(rows, hq * hd)

    def execute_graph(self):
        self.raw_cache.fill(0)
        logits = []
        for step in range(self.report["session_steps"]):
            position = 0 if step == 0 else self.defs["LL_M"] + step - 1
            if step == 0:
                h = (
                    self.f32(self.prefix[0, 2], self.defs["LL_M"] * self.width)
                    .reshape(-1, self.width)
                    .copy()
                )
            else:
                h = self.generator.bf(self.embeddings[self.ids[position]][None, :])
            for layer in range(self.defs["LL_L"]):
                qkv = self.matrix(
                    self.norm(h, self.gamma[layer, 1]), self.weights[layer, 2]
                )
                attention = self.attention(qkv, layer, position)
                projected = self.matrix(attention, self.weights[layer, 8])
                middle = self.generator.bf(h + projected)
                gu = self.matrix(
                    self.norm(middle, self.gamma[layer, 9]), self.weights[layer, 10]
                )
                gate, up = gu[:, : self.ff], gu[:, self.ff :]
                activated = (gate / (np.float32(1) + self.exp(-gate))) * up
                down = self.matrix(activated, self.weights[layer, 12])
                h = self.generator.bf(middle + down)
            out = self.matrix(self.norm(h, self.gamma[255, 13]), self.weights[255, 14])
            logits.append(out[-1].copy())
            print(
                f"logical Llama graph step {step + 1}/{self.report['session_steps']}",
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
    engine = LogicalGraph(args.build, args.radiance_root, args.rocc_root, out)
    checks = json.loads((args.build / "checks.json").read_text())
    _, _, observed = parse_console(
        (args.build / "spike.log").read_text(),
        checks,
        engine.report["phase_count"],
        False,
    )
    results = []
    for index, (check, expected) in enumerate(
        zip(checks, engine.execute_graph(), strict=True)
    ):
        expected.astype("<f4").tofile(out / f"expected{index}.bin")
        actual = np.frombuffer(observed[index], "<u4")
        wanted = expected.astype("<f4").view("<u4").reshape(-1)
        results.append(
            {
                "name": check["name"],
                "words": wanted.size,
                "word_mismatches": int(np.count_nonzero(actual != wanted)),
                "expected_sha256": sha(out / f"expected{index}.bin"),
            }
        )
    report = {
        "schema": "mx_gemmini.logical_cached_llama_replay.v1",
        "source_receipt_sha256": sha(args.build / "receipt.json"),
        "producer_sha256": sha(Path(__file__)),
        "session_steps": engine.report["session_steps"],
        "decoder_descriptors_interpreted": False,
        "matrix_reference_calls": engine.matrix_calls,
        "results": results,
        "passed": all(r["word_mismatches"] == 0 for r in results),
    }
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
