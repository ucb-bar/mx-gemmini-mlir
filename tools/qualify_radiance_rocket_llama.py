"""Build and audit a Rocket port of the pinned Radiance Llama phase program.

The smoke model checks the source's intermediate goldens. The recorded-forward
mode uses the published 160-token TinyLlama capture, including timed embedding
lookup, all 22 layers, and every output logit. Neither mode measures Muon.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from mx_gemmini_support.spike_timing import reject_deadlocked_timing
from mx_gemmini_support.target_profile import load_profile, profile_sha256

REVISION = "5866983682583486b38ae64a4d1bfa5e7b2a002e"
SOURCE_HASHES = {
    "kernels/nightly/llama_e2e/kernel.cpp": "727eba531bf077c92507c5c8ce398eb4ac8c5af630324cc6e69fd148f0d145ee",
    "kernels/nightly/llama_e2e/gen_data.py": "8be927ed38f01f8bb0fac216c2a5d131d8504f1658ac0505b41ec9c282afd0ae",
    "lib/include/nightly/mx.h": "1ec24ba974b76f6aa3ae17c5f8d58b400e93b9e4a80479def2cb4e94bca50be7",
    "kernels/nightly/common/nightly_data.py": "edd99c63760d6b70facdf624ed468b3729013e9962046b867a9b811d8b30b7d6",
    "lib/golden/golden.py": "da25bd520b0a1d694743a3224cc3f368dcc102902044c1c8a8da583cd4431747",
    "lib/golden/mx_golden.cpp": "538e83ffa93b33cfb5ad335a90a93318f12128a0202229e64a85300b0f2b3988",
    "lib/golden/mx_fp_math.h": "e91f2f83ff58c8d6a4c5c1161c9df4a63fc34e7052528a0b87f4dc27116e7b60",
    "lib/golden/Makefile": "b2b17a123be425f97705f74a37c6aae3978fd004fa33872a1be1ee5066395705",
}
PROFILE_SHA256 = "e9e33f134ddd3d0770fe3cec26f89923b2c63ca21c06657d0ae4139c95a0d28d"
EXTENSION_SHA256 = "5058ebfd1b1457d5897a87ad2208f67cb98e72c27ba111a771c981e5f6b5cdc5"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_sources(root):
    for name, digest in SOURCE_HASHES.items():
        if sha(root / name) != digest:
            raise ValueError(
                "Radiance source differs from the selected revision: " + name
            )
    return dict(SOURCE_HASHES)


def expected_accelerator_work(program):
    """Count all native matrix tiles and vector work in the source program."""
    loops = tiles = matrix_bytes = exp_elements = exp_calls = 0
    for words in program:
        d = [int(value) for value in words]
        if d[0] == 1:
            count = d[14] * d[15] * d[16] * d[17]
            loops += count
            tiles += count * (d[18] // 16) * 4 * (d[19] // 16)
            matrix_bytes += count * (d[18] + 64) * d[19]
        elif d[0] == 4:
            exp_calls += d[6] * d[5] * ((d[4] + 8191) // 8192)
            exp_elements += d[6] * d[5] * d[4]
        elif d[0] == 7:
            exp_calls += d[6] * ((d[5] + 8191) // 8192)
            exp_elements += d[6] * d[5]
    return {
        "loops": loops,
        "mesh_tiles": tiles,
        "mesh_busy_cycles": tiles * 16,
        "vpu_commands": exp_calls,
        "load_bytes": matrix_bytes + 2 * exp_elements,
        "scale_bytes": matrix_bytes // 32,
    }


def check_accelerator_work(program, console):
    """Refuse numerically complete runs with missing modeled device work."""
    from tools.report_stage_profile import parse_gemmini_perf

    reject_deadlocked_timing(console)
    actual = parse_gemmini_perf(console)
    expected = expected_accelerator_work(program)
    if actual["mode"] != "e4m3_vpu" or any(
        actual[key] != value for key, value in expected.items()
    ):
        raise ValueError(
            f"accelerator work is incomplete or differs: expected {expected}; observed {actual}"
        )
    return {"expected": expected, "observed": actual, "passed": True}


def parse_console(console, checks, phase_count, phase_timing):
    """Require terminal completion, intact guards and every emitted tensor byte."""
    reject_deadlocked_timing(console)
    lines = console.splitlines()
    completed = [line for line in lines if line.startswith("PORT_PHASES_COMPLETED ")]
    totals = [line for line in lines if line.startswith("PORT_TOTAL ")]
    if (
        lines.count("PORT_DONE") != 1
        or completed != [f"PORT_PHASES_COMPLETED {phase_count}"]
        or len(totals) != 1
    ):
        raise ValueError("port execution or timed phase sequence did not complete")
    match = re.fullmatch(r"PORT_TOTAL ([1-9][0-9]*) guards=0", totals[0])
    if match is None:
        raise ValueError("invalid cycles or corrupted arena guards")
    outputs, phases = {}, []
    for line in lines:
        if line.startswith("PORT_PHASE "):
            _, index, operation, cycles = line.split()
            index, operation, cycles = int(index), int(operation), int(cycles)
            if index != len(phases) or cycles <= 0:
                raise ValueError("incomplete or reordered phase timing")
            phases.append({"phase": index, "operation": operation, "cycles": cycles})
        elif line.startswith("PORT_BYTES "):
            _, index, offset, count, *words = line.split()
            index, offset, count = int(index), int(offset), int(count)
            if not 0 <= index < len(checks):
                raise ValueError("unknown output tensor")
            target = outputs.setdefault(index, bytearray())
            if offset != len(target) or not 1 <= count <= 32 or count != len(words):
                raise ValueError("incomplete, duplicated or reordered tensor output")
            if any(not re.fullmatch("[0-9a-f]{2}", word) for word in words):
                raise ValueError("invalid observed tensor byte")
            target.extend(int(word, 16) for word in words)
    if len(phases) != (phase_count if phase_timing else 0):
        raise ValueError("missing or unexpected phase timings")
    if set(outputs) != set(range(len(checks))) or any(
        len(outputs[i]) != check["bytes"] for i, check in enumerate(checks)
    ):
        raise ValueError("complete output tensors are required")
    return int(match.group(1)), phases, {i: bytes(v) for i, v in outputs.items()}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recorded_model(generator, capture, out):
    """Bind the real checkpoint and tokens; no model or dataset download occurs."""
    from safetensors import safe_open

    receipt = json.loads((capture / "capture_receipt.json").read_text())
    identities = {}
    for name in ("weights.safetensors", "inputs.npz", "golden.npy", "model.mlir"):
        path = capture / name
        digest = sha(path)
        expected = receipt["artifacts"][name]
        if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError("recorded capture identity differs: " + name)
        identities[name] = digest
    with np.load(capture / "inputs.npz", allow_pickle=False) as archive:
        if archive.files != ["in0"]:
            raise ValueError("recorded forward must have one token-ID input")
        ids = np.array(archive["in0"], dtype="<i8", copy=True)
    if ids.shape != (1, 160) or np.any(ids < 0) or np.any(ids >= 32000):
        raise ValueError("recorded forward requires 160 TinyLlama token IDs")
    ids.tofile(out / "token_ids.bin")
    with safe_open(capture / "weights.safetensors", framework="numpy") as tensors:

        def get(name):
            return tensors.get_tensor("lm." + name).astype(np.float32)

        embedding = get("model.embed_tokens.weight")
        embedding.astype("<f4").tofile(out / "embedding.bin")
        x0 = embedding[ids[0]]
        layers = []
        for layer in range(22):
            prefix = f"model.layers.{layer}."
            w = lambda name: get(prefix + name + ".weight")
            layers.append(
                {
                    "wqkv": np.concatenate(
                        [
                            w("self_attn.q_proj"),
                            w("self_attn.k_proj"),
                            w("self_attn.v_proj"),
                        ]
                    ),
                    "wo": w("self_attn.o_proj"),
                    "wgu": np.concatenate([w("mlp.gate_proj"), w("mlp.up_proj")]),
                    "wd": w("mlp.down_proj"),
                    "g_in": w("input_layernorm"),
                    "g_post": w("post_attention_layernorm"),
                    "kc": np.zeros((4, 0, 64), np.float32),
                    "vc": np.zeros((4, 0, 64), np.float32),
                }
            )
        model = {
            "layers": layers,
            "g_final": get("model.norm.weight"),
            "wlm": get("lm_head.weight"),
            "x0": x0,
            "hidden": None,
            "logits": None,
            "labels": None,
        }
    generator.load_hf = lambda cfg, m, p0, layers: model
    original_quant_weight = generator.quant_weight
    generator.quant_weight = lambda w: original_quant_weight(generator.bf(w))
    return identities


def _prepare(args, out):
    sys.dont_write_bytecode = True
    source = args.radiance_root.resolve()
    checked_sources(source)
    generator_path = source / "kernels/nightly/llama_e2e/gen_data.py"
    generator = _load_module("radiance_llama_generator", generator_path)
    memories = []
    original_mem = generator.Mem

    class ObservedMem(original_mem):
        def __init__(self, name):
            super().__init__(name)
            memories.append(self)

    generator.Mem = ObservedMem
    # The source builds its reference executable and writes temporary tensors
    # under HERE. Redirect these build products; keep source inputs untouched.
    golden_dir = out / "golden"
    golden_dir.mkdir()
    for name in ("golden.py", "mx_golden.cpp", "mx_fp_math.h", "Makefile"):
        shutil.copyfile(source / "lib/golden" / name, golden_dir / name)
    sys.modules["golden"] = _load_module("golden", golden_dir / "golden.py")
    generator.HERE = out / "generator"
    generator.HERE.mkdir()
    if args.compiler_packing and not args.recorded_forward:
        original_quant = generator.mx_quant
        original_exp = generator.block_exp
        generator.block_exp = lambda maximum: np.maximum(original_exp(maximum), -23)
        generator.mx_quant = lambda values: original_quant(generator.bf(values))
    identities = None
    if args.recorded_forward:
        # This target has a larger accumulator half than Muon. M=160 fits
        # without padding queries. Pad only the key axis to the source TK=256;
        # the causal mask excludes the 96 extra columns.
        original = generator_path.read_text()
        rewritten = original.replace("P0 = SK - M", "P0 = 0").replace(
            "assert M in (64, 128) and", "assert M in (64, 128, 160) and"
        )
        if rewritten == original or rewritten.count("P0 = 0") != 1:
            raise ValueError("source forward adaptation did not match")
        # Only main is replaced: keep the redirected memory/reference helpers.
        temporary = dict(generator.__dict__)
        exec(compile(rewritten, str(generator_path), "exec"), temporary)
        import types

        generator.main = types.FunctionType(
            temporary["main"].__code__, generator.__dict__
        )
        identities = _recorded_model(generator, args.recorded_forward.resolve(), out)
    argv = sys.argv
    sys.argv = [
        str(generator_path),
        "--model",
        "tinyllama" if identities else "smoke",
        "--M",
        "160" if identities else str(args.chunk),
        "--ctx",
        "256",
        "--attn",
        "gemm",
        "--out",
        str(out / "data"),
        "--header",
        str(out / "llama_data.h"),
    ]
    sys.argv += ["--no-emulate"] if identities else ["--debug-l0"]
    try:
        generator.main()
    finally:
        sys.argv = argv
    if len(memories) != 1:
        raise ValueError("source generator produced unexpected memory maps")
    memory = memories[0]
    header = (out / "llama_data.h").read_text()
    defines = dict(
        re.findall(r"^#define (LL_[A-Z0-9_]+) (0x[0-9a-f]+|[0-9]+)u$", header, re.M)
    )
    defines = {key: int(v, 0) for key, v in defines.items()}
    program_offset = defines["LL_PROG_ADDR"] - generator.REGION_BASE[3]
    program = np.frombuffer(
        memory.blobs[3], dtype="<u4", count=defines["LL_NP"] * 32, offset=program_offset
    ).reshape(-1, 32)
    # Validate capacities before generating or executing any target command.
    for d in program:
        if d[0] not in (1, 2, 3, 4, 5, 7):
            raise ValueError("source phase kind has no qualified Rocket port")
        if d[0] == 1:
            tm, tk = int(d[18]), int(d[19])
            if (
                d[20]
                or tm % 16
                or tk % 32
                or tm * 64 * 2 > 32768
                or (tm + 64) * tk > 131072
                or max(tm, 64) * (tk // 32) > 4096
            ):
                raise ValueError("source GEMM exceeds the qualified native port")
        elif d[0] == 2 and (d[13] or d[8] > 16384 or d[8] % 256):
            raise ValueError("unsupported source prenorm extent or fusion")
        elif d[0] == 3 and (d[25] or d[6] not in (64, 128) or d[22] % 32):
            raise ValueError("unsupported source RoPE/cache extent or fusion")
        elif d[0] == 4 and (d[4] > 16384 or d[4] % 256):
            raise ValueError("unsupported source softmax extent")
        elif d[0] in (5, 7) and (d[5] > 16384 or d[5] % 256):
            raise ValueError("unsupported source quantization extent")
    np.asarray(program).tofile(out / "phases.bin")
    checks = []
    if identities:
        checks.append(
            {"name": "logits", "address": 0, "bytes": 160 * 32000 * 4, "kind": "f32"}
        )
        for d in program:
            if d[0] == 2 and (int(d[1]) & 255) in (1, 13):
                checks.append(
                    {
                        "name": f"residual_{int(d[1])}",
                        "address": int(d[4]),
                        "bytes": 160 * 2048 * 2,
                        "kind": "bf16",
                    }
                )
    else:
        pattern = r'\{"([^"]+)", (0x[0-9a-f]+)u, (0x[0-9a-f]+)u, (\d+)u, (\d+), (\d+), (\d+)u\}'
        for name, dev, ref, n, scale, kind, tolerance in re.findall(pattern, header):
            address, ref, n, kind = int(dev, 16), int(ref, 16), int(n), int(kind)
            size = n * (1 if kind else 2)
            region = next(
                r
                for r in (1, 2, 3)
                if generator.REGION_BASE[r] <= ref < generator.REGION_END[r]
            )
            offset = ref - generator.REGION_BASE[region]
            expected = bytes(memory.blobs[region][offset : offset + size])
            if len(expected) != size:
                raise ValueError("source reference extends outside its data blob")
            checks.append(
                {
                    "name": name,
                    "address": address,
                    "bytes": size,
                    "golden": expected.hex(),
                    "kind": "bytes" if kind else "bf16",
                    "source_tolerance": int(tolerance),
                }
            )
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    config = (
        '#include "llama_data.h"\nstruct RkCheck { uint32_t address,bytes; };\n'
        f"#define RK_NCHECKS {len(checks)}\n#define RK_PROFILE_PHASES {int(args.phase_timing)}\n"
        "static const struct RkCheck RK_CHECKS[] = {\n"
    )
    config += "".join(f"{{{c['address']}u,{c['bytes']}u}},\n" for c in checks) + "};\n"
    assembly = ".section .data\n"
    regions = {}
    for r in (1, 2, 3):
        size = memory.arena[r] - generator.REGION_BASE[r]
        blob_size = max(len(memory.blobs[r]), 256)
        if size < blob_size:
            raise ValueError("arena truncates a source region")
        regions[r] = {
            "virtual_base": generator.REGION_BASE[r],
            "bytes": size,
            "blob_bytes": blob_size,
            "blob_sha256": sha(out / f"data/llama.region{r}.bin"),
        }
        config += f"#define RK_REGION{r}_BYTES {size}u\n"
        assembly += (
            f".balign 64\n.global rk_guard{r}a\nrk_guard{r}a:\n.fill 64,1,0xa5\n"
            f'.global rk_region{r}\nrk_region{r}:\n.incbin "data/llama.region{r}.bin"\n'
            f".zero {size - blob_size}\n.global rk_guard{r}b\nrk_guard{r}b:\n.fill 64,1,0xa5\n"
        )
    if identities or args.compiler_packing:
        config += "#define RK_ROUND_QUANT_INPUTS 1\n"
    if identities:
        config += (
            "#define RK_EMBEDDING_LOOKUP 1\n"
            f"#define RK_X0_ADDRESS {int(program[0, 2])}u\n"
        )
        assembly += (
            '.balign 64\n.global rk_embeddings\nrk_embeddings:\n.incbin "embedding.bin"\n'
            '.balign 64\n.global rk_token_ids\nrk_token_ids:\n.incbin "token_ids.bin"\n'
        )
    assembly += '.section .note.GNU-stack,"",@progbits\n'
    (out / "data.S").write_text(assembly)
    (out / "rocket_data.h").write_text(config)
    return {
        "regions": regions,
        "phase_count": len(program),
        "checks": checks,
        "capture_files_sha256": identities,
        "defines": defines,
    }


def run_experiment(args):
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    profile = load_profile(args.profile)
    if (
        profile_sha256(profile) != PROFILE_SHA256
        or sha(args.extlib) != EXTENSION_SHA256
    ):
        raise ValueError(
            "this experiment requires the recorded Rocket profile and extension"
        )
    report = {
        "schema": "mx_gemmini.radiance_rocket_llama_port.v1",
        "status": "preparing",
        "source_revision": REVISION,
        "source_files_sha256": checked_sources(args.radiance_root),
        "profile_sha256": PROFILE_SHA256,
        "extension_sha256": EXTENSION_SHA256,
        "mode": "recorded_tinyllama_forward"
        if args.recorded_forward
        else "random_smoke",
        "phase_timing": args.phase_timing,
        "compiler_packing": bool(args.compiler_packing or args.recorded_forward),
        "original_muon_timing": False,
        "full_model_speedup_qualified": False,
        "commands": [],
        "producer_sha256": sha(Path(__file__)),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "adaptations": [
            "single Rocket core replaces Muon SIMT launch and barriers",
            "native DRAM loops replace split scratchpad/accumulator commands",
            "source M/N/K tile partition and phase order retained",
            "Rocket FP32 row arithmetic; Rocket VPU BF16 exponential",
            "source virtual regions relocated to linked writable buffers",
        ],
    }
    receipt = out / "receipt.json"

    def save():
        receipt.write_text(json.dumps(report, indent=2) + "\n")

    def run(command, label, *, env=None, timeout=600):
        command = list(map(str, command))
        report["commands"].append({"label": label, "command": command})
        save()
        with (out / (label + ".log")).open("wb") as log:
            result = subprocess.run(
                command,
                cwd=out,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        report["commands"][-1]["exit_code"] = result.returncode
        save()
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {out / (label + '.log')}")

    save()
    try:
        shutil.copyfile(Path(__file__), out / "producer.py")
        fixture = Path(__file__).parent / "fixtures/radiance_rocket_llama.c"
        shutil.copyfile(fixture, out / "driver.c")
        report["port_source_sha256"] = sha(out / "driver.c")
        prepared = _prepare(args, out)
        report.update(
            {key: value for key, value in prepared.items() if key != "checks"}
        )
        if args.recorded_forward:
            report["adaptations"] += [
                "M=160 admitted by the larger Rocket accumulator; P0=0",
                "96 padded key columns excluded by the causal mask",
                "weights rounded through BF16 before source MX quantization",
                "live MX inputs rounded through BF16 with minimum scale 2^-23",
                "embedding lookup and final F32 widening inside the timed boundary",
            ]
        elif args.compiler_packing:
            report["adaptations"] += [
                "source smoke inputs use BF16 before MX quantization, with minimum scale 2^-23"
            ]
        startup = args.startup_dir.resolve(strict=True)
        riscv = args.riscv_root.resolve(strict=True)
        cc = riscv / "bin/riscv64-unknown-elf-gcc"
        readelf = riscv / "bin/riscv64-unknown-elf-readelf"
        os.environ["MERLIN_READELF"] = str(readelf)
        report["tools_sha256"] = {
            "cc": sha(cc),
            "readelf": sha(readelf),
            "spike": sha(riscv / "bin/spike"),
        }
        flags = [
            "-DPREALLOCATE=1",
            "-DMULTITHREAD=1",
            "-DBAREMETAL=1",
            "-DSPIKE_SIM",
            "-mcmodel=medany",
            "-O2",
            "-ffreestanding",
            "-fno-fast-math",
            "-fno-math-errno",
            "-ffp-contract=off",
            "-fno-common",
            "-fno-builtin",
            "-fno-tree-loop-distribute-patterns",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-std=gnu11",
            "-I",
            str(out),
            "-I",
            str(startup),
            "-I",
            str(startup.parent.parent),
            "-I",
            str(startup.parent.parent / "env"),
        ]
        sources = [
            out / "driver.c",
            out / "data.S",
            *sorted(startup.glob("*.c")),
            *sorted(startup.glob("*.S")),
        ]
        report["startup_sha256"] = {
            str(p): sha(p) for p in sources if p.parent == startup
        }
        objects = []
        report["status"] = "building"
        for i, source in enumerate(sources):
            obj = out / f"part{i}.o"
            run([cc, *flags, "-c", source, "-o", obj], f"compile{i}")
            objects.append(obj)
        elf = out / "session.elf"
        run(
            [
                cc,
                "-nostdlib",
                "-nostartfiles",
                "-static",
                "-T",
                startup / "test.ld",
                *objects,
                "-lm",
                "-lgcc",
                "-o",
                elf,
            ],
            "link",
        )
        from merlin.llvmlower.device_build import (
            DeviceRouting,
            admit_linked_device_elf,
            freeze_device_elf_admission,
        )
        from merlin.model_check.admission import service

        route = DeviceRouting(
            device="mx_gemmini",
            package_dir=Path(__file__).resolve().parents[1],
            operand_dtype="f32",
            accum_dtype="bf16",
            linked_elf_admission=service("mx_gemmini"),
        )
        report["elf_sha256"] = sha(elf)
        report["elf_bytes"] = elf.stat().st_size
        report["admission"] = admit_linked_device_elf(
            route,
            freeze_device_elf_admission(route),
            elf=elf,
            linked_sha256=report["elf_sha256"],
            directory=out / "admission",
        )
        environment = os.environ | {
            "GEMMINI_MODE": "both",
            "GEMMINI_PERF_CONFIG": "e4m3_vpu",
            "GEMMINI_PERF_SET": "mem.host_tracking=0",
            "GEMMINI_PERF_OUT": str(out / "perf.txt"),
        }
        report["environment"] = {
            key: environment[key]
            for key in ("GEMMINI_MODE", "GEMMINI_PERF_CONFIG", "GEMMINI_PERF_SET")
        }
        report["status"] = "running"
        run(
            [
                riscv / "bin/spike",
                "--isa=rv64gc_zicntr",
                "--extlib=" + str(args.extlib.resolve()),
                "--extension=gemmini",
                "-m0x80000000:0x1c0000000",
                elf,
            ],
            "spike",
            env=environment,
            timeout=args.timeout,
        )
        console = (out / "spike.log").read_text()
        cycles, phases, outputs = parse_console(
            console, prepared["checks"], prepared["phase_count"], args.phase_timing
        )
        report["accelerator_work"] = check_accelerator_work(
            np.fromfile(out / "phases.bin", dtype="<u4").reshape(-1, 32),
            (out / "perf.txt").read_text(),
        )
        report["cycles"], report["phases"] = cycles, phases
        report["results"] = []
        for i, check in enumerate(prepared["checks"]):
            observed = outputs[i]
            (out / f"observed{i}.bin").write_bytes(observed)
            row = {
                "name": check["name"],
                "bytes": len(observed),
                "observed_sha256": hashlib.sha256(observed).hexdigest(),
            }
            if "golden" in check:
                expected = bytes.fromhex(check["golden"])
                row["byte_mismatches"] = sum(
                    a != b for a, b in zip(observed, expected, strict=True)
                )
                if check["kind"] == "bf16":
                    to_f32 = lambda raw: (
                        np.frombuffer(raw, "<u2").astype(np.uint32) << 16
                    ).view("<f4")
                    x, y = to_f32(observed), to_f32(expected)
                    row.update(
                        {
                            "word_mismatches": int(np.count_nonzero(x != y)),
                            "max_abs": float(np.max(np.abs(x - y))),
                            "rmse": float(np.sqrt(np.mean((x - y) ** 2))),
                            "all_finite": bool(np.isfinite(x).all()),
                        }
                    )
            report["results"].append(row)
        if args.recorded_forward:
            logits = np.frombuffer(outputs[0], "<f4").reshape(160, 32000)
            if not np.isfinite(logits).all():
                raise ValueError("TinyLlama port produced nonfinite logits")
            np.save(out / "logits.npy", logits, allow_pickle=False)
            reference = np.load(
                args.recorded_forward / "golden.npy", allow_pickle=False
            ).reshape(logits.shape)
            from tools.compare_mx_model_quality import _kl

            kl = _kl(reference.astype(np.float64), logits.astype(np.float64))
            report["floating_reference_diagnostics"] = {
                "mean_kl_all_160": float(kl.mean()),
                "worst_kl_all_160": float(kl.max()),
                "mean_kl_late_33": float(kl[-33:].mean()),
                "worst_kl_late_33": float(kl[-33:].max()),
                "top1_all_160": int(
                    np.count_nonzero(logits.argmax(-1) == reference.argmax(-1))
                ),
                "top1_late_33": int(
                    np.count_nonzero(
                        logits[-33:].argmax(-1) == reference[-33:].argmax(-1)
                    )
                ),
            }
        else:
            if not args.compiler_packing and any(
                row.get("byte_mismatches") for row in report["results"][:11]
            ):
                raise ValueError(
                    "pre-exponential stages differ from the source hardware golden"
                )
            report["pre_exponential_source_checks"] = (
                "quantization_variant_requires_reference_replay"
                if args.compiler_packing
                else "all_bytes_exact"
            )
        report["status"] = "completed_requires_matched_comparison"
        report["files_sha256"] = {
            p.name: sha(p) for p in out.iterdir() if p.is_file() and p != receipt
        }
        save()
        print(
            json.dumps(
                {"receipt": str(receipt), "status": report["status"], "cycles": cycles},
                indent=2,
            )
        )
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        save()
        raise


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
    parser.add_argument(
        "--recorded-forward", type=Path, help="published TinyLlama capture directory"
    )
    parser.add_argument("--chunk", type=int, choices=(64, 128), default=64)
    parser.add_argument("--phase-timing", action="store_true")
    parser.add_argument(
        "--compiler-packing",
        action="store_true",
        help="check the small source model with BF16-input MX packing",
    )
    parser.add_argument("--timeout", type=int, default=36000)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error("--out-dir must be fresh")
    run_experiment(args)


if __name__ == "__main__":
    main()
