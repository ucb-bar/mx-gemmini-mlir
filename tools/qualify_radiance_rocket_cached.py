"""Run continuous cached Llama execution using the Radiance-derived Rocket port.

The descriptor scheduler, persistent cache and input stream are independent of
the compiler's generated host program. A small recurrent smoke run is available
before the complete matched 128-token prefill and 32 single-token decodes.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np

from tools import qualify_radiance_rocket_llama as forward


def decode_program(prefill, defines, position):
    """Pad physical rows to sixteen; execute one logical token per layer."""
    result = np.array(prefill, dtype="<u4", copy=True)
    hd, groups, keys = (
        defines["LL_HD"],
        defines["LL_HQ"] // defines["LL_HKV"],
        defines["LL_SK"],
    )
    if groups > 16 or hd not in (64, 128) or keys % 256:
        raise ValueError("unsupported cached attention geometry")
    for d in result:
        op = int(d[1]) & 255
        if d[0] == 1:
            d[15], d[18], d[13] = 1, 16, 16 * d[12]
            if op == 4:  # QK: eight logical query heads, sixteen physical rows.
                d[4], d[5], d[11] = 16 * hd, 16 * hd // 32, 16 * keys * 2
            elif op == 6:  # PV: merge the query-head rows into one hidden row.
                d[4], d[5] = 16 * keys, 16 * keys // 32
                d[11], d[12], d[13] = groups * hd * 2, hd * 2, 16 * hd * 2
        elif d[0] == 2:
            d[9], d[10] = 1, 16
        elif d[0] == 3:
            d[11], d[12] = 16 * hd, 16 * hd // 32
            d[21], d[22], d[24] = position, 1, 16
        elif d[0] == 4:
            d[3], d[5] = 16 * keys * 2, groups
            d[9], d[10] = 16 * keys, 16 * keys // 32
            d[12], d[13], d[14] = position, 1, 16
        elif d[0] in (5, 7):
            d[6], d[9] = 1, 16
        else:
            raise ValueError("unqualified source phase in cached scheduler")
    return result


def prepare(args, out):
    shutil.copyfile(Path(__file__), out / "cached_producer.py")
    original = copy.copy(args)
    original.recorded_forward = (
        args.recorded_session / "stages/prefill" if args.recorded_session else None
    )
    original.recorded_tokens = 128
    original.zero_prefix = True
    original.skip_source_emulation = True
    original.compiler_packing = True
    prepared = forward._prepare(original, out)
    defs = prepared["defines"]
    m, layers, heads, hd, vocab = (
        defs[k] for k in ("LL_M", "LL_L", "LL_HKV", "LL_HD", "LL_V")
    )
    steps = args.decode_steps + 1
    capacity = m + args.decode_steps
    if capacity > defs["LL_SK"]:
        raise ValueError("session exceeds the persistent cache capacity")
    prefix = np.fromfile(out / "phases.bin", dtype="<u4").reshape(-1, 32)
    programs = np.stack(
        [prefix]
        + [decode_program(prefix, defs, m + step) for step in range(args.decode_steps)]
    )
    programs.tofile(out / "session_programs.bin")
    programs.tofile(out / "phases.bin")
    source = forward._load_module(
        "cached_source_data",
        args.radiance_root / "kernels/nightly/llama_e2e/gen_data.py",
    )
    cs, sn = source.rope_tables(capacity, 0, hd, 10000.0)
    cs.astype("<f4").tofile(out / "cos.bin")
    sn.astype("<f4").tofile(out / "sin.bin")
    if args.recorded_session:
        session_identities = {}
        for name in (
            "session_contract.yaml",
            "stages/prefill/capture_receipt.json",
            "stages/decode/capture_receipt.json",
            "stages/decode/model.mlir",
            "stages/decode/session_inputs.npz",
            "stages/decode/session_quality_fp32.npz",
        ):
            session_identities[name] = forward.sha(args.recorded_session / name)
        with np.load(
            original.recorded_forward / "inputs.npz", allow_pickle=False
        ) as arrays:
            prefill_ids = arrays["in0"]
        with np.load(
            args.recorded_session / "stages/decode/session_inputs.npz",
            allow_pickle=False,
        ) as arrays:
            decode_ids = arrays["decode_tokens"]
        if prefill_ids.shape != (1, 128) or decode_ids.shape != (32, 1, 1):
            raise ValueError("recorded session must contain exactly 128+32 tokens")
        ids = np.concatenate([prefill_ids.reshape(-1), decode_ids.reshape(-1)])
        if (
            ids.dtype != np.int64
            or ids.size != capacity
            or np.any(ids < 0)
            or np.any(ids >= vocab)
        ):
            raise ValueError("invalid recorded session input stream")
        prepared["session_capture_sha256"] = session_identities
    else:
        rng = np.random.default_rng(1701)
        ids = rng.integers(0, vocab, capacity, dtype=np.int64)
        rng.normal(0, 1, (vocab, defs["LL_D"])).astype("<f4").tofile(
            out / "embedding.bin"
        )
    ids.astype("<i8").tofile(out / "token_ids.bin")
    assets = {
        "rk_session_programs": "session_programs.bin",
        "rk_session_cos": "cos.bin",
        "rk_session_sin": "sin.bin",
    }
    if not args.recorded_session:
        assets |= {"rk_embeddings": "embedding.bin", "rk_token_ids": "token_ids.bin"}
    with (out / "data.S").open("a") as assembly:
        assembly.write(".section .rodata\n")
        for symbol, filename in assets.items():
            assembly.write(
                f'.balign 64\n.global {symbol}\n{symbol}:\n.incbin "{filename}"\n'
            )
    checks = [
        {
            "name": "session_logits",
            "address": 0,
            "bytes": steps * vocab * 4,
            "kind": "f32",
        },
        {
            "name": "final_kv_cache",
            "address": 1,
            "bytes": 2 * layers * heads * capacity * hd * 4,
            "kind": "f32",
        },
    ]
    header = '#include "llama_data.h"\n#define RK_ROUND_QUANT_INPUTS 1\n#define RK_PROFILE_PHASES 0\n'
    header += f"#define RK_STEPS {steps}u\n#define RK_CAPACITY {capacity}u\n"
    header += f"#define RK_SESSION_RECORDED {int(bool(args.recorded_session))}\n"
    header += "struct RkCheck { uint32_t address,bytes; };\n#define RK_NCHECKS 2\n"
    header += "static const struct RkCheck RK_CHECKS[] = {\n"
    header += "".join(f"{{{r['address']}u,{r['bytes']}u}},\n" for r in checks) + "};\n"
    for region, info in prepared["regions"].items():
        header += f"#define RK_REGION{region}_BYTES {info['bytes']}u\n"
    (out / "rocket_data.h").write_text(header)
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    fixture = Path(__file__).parent / "fixtures/radiance_rocket_llama.c"
    shutil.copyfile(fixture, out / fixture.name)
    prepared.update(
        mode="recorded_cached_tinyllama" if args.recorded_session else "cached_smoke",
        phase_count=len(programs) * len(prefix),
        checks=checks,
        session_steps=steps,
        cache_capacity=capacity,
        base_port_sha256=forward.sha(fixture),
        cached_producer_sha256=forward.sha(Path(__file__)),
        input_stream_sha256=forward.sha(out / "token_ids.bin"),
        timing_scope="reset_plus_prefill_plus_recurrent_decode_including_input_and_output_preparation",
        independent_host_graph=True,
        full_model_speedup_qualified=False,
        adaptations=[
            "Radiance-derived native Rocket matrix, prenorm, softmax and SiLU kernels",
            "independent descriptor scheduler; one-token decodes with sixteen physical matrix rows",
            "persistent FP32 K/V state; V groups requantized from original values after each append",
            "cache reset, input embedding lookup and final F32 logit widening timed",
            "RoPE source tables at absolute token positions; no teacher-forced floating KV state",
            "BF16 inputs before FP8/E8M0 block-32 packing; minimum exponent -23",
        ],
    )
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "radiance-root",
        "startup-dir",
        "riscv-root",
        "profile",
        "extlib",
        "out-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--recorded-session", type=Path)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--smoke-query-heads", type=int, choices=(4, 16), default=4)
    parser.add_argument("--timeout", type=int, default=86400)
    args = parser.parse_args()
    if args.out_dir.exists() or not 1 <= args.decode_steps <= 64:
        parser.error("output must be fresh and decode steps must be in [1,64]")
    if args.recorded_session and args.decode_steps != 32:
        parser.error("the recorded comparison requires all 32 decode steps")
    args.recorded_forward = None
    args.chunk, args.compiler_packing, args.phase_timing = 64, True, False
    forward.run_experiment(
        args,
        prepare=prepare,
        fixture=Path(__file__).parent / "fixtures/radiance_rocket_cached.c",
    )


if __name__ == "__main__":
    main()
