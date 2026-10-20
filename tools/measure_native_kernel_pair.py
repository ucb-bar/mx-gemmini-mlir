"""Time pinned native MX kernels against compiler issuers on the same Rocket guest.

Radiance fixtures supply packed operands and independent BF16 goldens. The
native implementation is from npu-exploration, not the Radiance Muon kernel.
This control neither measures a complete model nor ports Muon execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from mx_gemmini_support.scheduled_native_issuer import emit_scheduled_native_issuer
from mx_gemmini_support.source_gemm import read_source_gemm
from mx_gemmini_support.source_payload import read_source_payload
from mx_gemmini_support.spike_timing import reject_deadlocked_timing
from mx_gemmini_support.target_profile import load_profile, profile_sha256

NATIVE_REVISION = "0344bcba70b899039ccd0ac0801ca5605d46d3eb"
NATIVE_HEADER_SHA256 = (
    "98bc0384dbf1917af9427dd30a51770063e56152933259d5fb32e5e04a5505e8"
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_output(console, golden, trials):
    """Require terminal completion, all timed calls, full words and guard checks."""
    reject_deadlocked_timing(console)
    lines = console.splitlines()
    if lines.count("PAIR_DONE") != 1:
        raise ValueError("kernel pair did not finish")
    observations, outputs = [], {}
    for line in lines:
        if line.startswith("PAIR "):
            row = dict(part.split("=", 1) for part in line.split()[1:])
            if set(row) != {
                "trial",
                "native",
                "compiler",
                "mismatch",
                "oracle",
                "guard",
            }:
                raise ValueError("malformed paired timing record")
            row = {key: int(value) for key, value in row.items()}
            if (
                row["trial"] != len(observations)
                or min(row["native"], row["compiler"]) <= 0
            ):
                raise ValueError("missing, duplicated or invalid paired timings")
            if any(row[key] for key in ("mismatch", "oracle", "guard")):
                raise ValueError("kernel output or guard mismatch")
            observations.append(row)
        elif line.startswith("PAIR_WORDS "):
            _, role, trial, offset, count, *words = line.split()
            trial, offset, count = int(trial), int(offset), int(count)
            if role not in {"native", "compiler"} or not 0 <= trial < trials:
                raise ValueError("unknown observed kernel output")
            target = outputs.setdefault((role, trial), [])
            if offset != len(target) or not 1 <= count <= 32 or len(words) != count:
                raise ValueError("partial, duplicated or reordered kernel words")
            values = [int(word, 16) for word in words]
            if any(not 0 <= value <= 0xFFFF for value in values):
                raise ValueError("invalid BF16 output word")
            target.extend(values)
    if len(observations) != trials or len(outputs) != 2 * trials:
        raise ValueError("missing paired calls or complete outputs")
    expected = np.frombuffer(golden, dtype="<u2")
    for values in outputs.values():
        if not np.array_equal(np.asarray(values, dtype="<u2"), expected):
            raise ValueError("complete observed words differ from the source golden")
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "driver",
        "native-header",
        "rocc-root",
        "startup-dir",
        "riscv-root",
        "extlib",
        "profile",
        "out-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--tile-policy", choices=("legacy", "movement"), default="movement"
    )
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.out_dir.exists() or not 2 <= args.trials <= 8:
        parser.error("fresh output and 2..8 paired trials are required")
    if sha(args.native_header) != NATIVE_HEADER_SHA256:
        parser.error("native header differs from the pinned independent source")
    kernel = read_source_gemm(args.driver.resolve(strict=True))
    if kernel.datatype not in {"FP4", "FP8"} or kernel.quant_output:
        parser.error("this paired control supports FP4/FP8 BF16-output kernels")
    m, n, k = kernel.shape
    profile = load_profile(args.profile)
    schedule = emit_scheduled_native_issuer(
        profile,
        kernel.shape,
        precision=kernel.datatype,
        symbol="compiler_issue",
        tile_policy=args.tile_policy,
    )
    resources = read_source_payload(kernel)
    out = args.out_dir.resolve()
    out.mkdir(parents=True)
    payload, offsets = bytearray(), {}
    for name in (
        "activation",
        "activation_scales",
        "weight",
        "weight_scales",
        "golden_bf16",
    ):
        payload.extend(bytes((-len(payload)) % 64))
        offsets[name] = len(payload)
        payload.extend(resources[name].data)
    (out / "payload.bin").write_bytes(payload)
    (out / "payload.S").write_text(
        '.section .rodata\n.balign 64\n.global payload\npayload:\n.incbin "payload.bin"\n.section .note.GNU-stack,"",@progbits\n'
    )
    config = {
        "PAIR_WORDS": m * n,
        "PAIR_TRIALS": args.trials,
        "OFFSET_A": offsets["activation"],
        "OFFSET_AS": offsets["activation_scales"],
        "OFFSET_B": offsets["weight"],
        "OFFSET_BS": offsets["weight_scales"],
        "OFFSET_GOLD": offsets["golden_bf16"],
    }
    (out / "pair_config.h").write_text(
        "".join(f"#define {key} {value}\n" for key, value in config.items())
    )
    (out / "compiler.c").write_text(schedule.issuer.source)
    shutil.copyfile(args.native_header, out / "mx_native.h")
    fixture = Path(__file__).parent / "fixtures/native_kernel_pair_driver.c"
    shutil.copyfile(fixture, out / "driver.c")
    if kernel.datatype == "FP8":
        call = f"mxn_matmul_ex(a,{k},b,{n},(uint16_t *)c,{n},asc,{m},bsc,{n},{m},{k},{n},0,0)"
    else:
        call = f"mxn_matmul_fp4(a,{k},b,{n // 2},(uint16_t *)c,{n},asc,{m},bsc,{n},{m},{k},{n})"
    (out / "native.c").write_text(
        '#include "mx_native.h"\n'
        "void native_issue(const void *a,const void *asc,const void *b,const void *bsc,const void *c,const void *scratch) {\n"
        f"  (void)scratch; if ({call}) __builtin_trap();\n  gemmini_fence();\n}}\n"
    )
    startup = args.startup_dir.resolve(strict=True)
    cc = args.riscv_root.resolve() / "bin/riscv64-unknown-elf-gcc"
    readelf = args.riscv_root.resolve() / "bin/riscv64-unknown-elf-readelf"
    if not readelf.is_file():
        parser.error("selected RISC-V toolchain lacks readelf for image admission")
    os.environ["MERLIN_READELF"] = str(readelf)
    flags = [
        "-DPREALLOCATE=1",
        "-DMULTITHREAD=1",
        "-DBAREMETAL=1",
        "-DSPIKE_SIM",
        "-mcmodel=medany",
        "-O2",
        "-ffreestanding",
        "-fno-fast-math",
        "-fno-common",
        "-fno-builtin",
        "-fno-tree-loop-distribute-patterns",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-std=gnu11",
        "-I",
        str(out),
        "-I",
        str(args.rocc_root.resolve()),
        "-I",
        str(startup),
        "-I",
        str(startup.parent.parent),
        "-I",
        str(startup.parent.parent / "env"),
    ]
    commands, objects = [], []

    def run(command, label, *, env=None, timeout=300):
        command = list(map(str, command))
        commands.append(command)
        with (out / (label + ".log")).open("wb") as log:
            result = subprocess.run(
                command,
                cwd=out,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {out / (label + '.log')}")

    sources = [
        out / "native.c",
        out / "compiler.c",
        out / "payload.S",
        *sorted(startup.glob("*.c")),
        *sorted(startup.glob("*.S")),
    ]
    for i, source in enumerate(sources):
        obj = out / f"part{i}.o"
        run([cc, *flags, "-c", source, "-o", obj], f"compile{i}")
        objects.append(obj)
    report = {
        "schema": "mx_gemmini.independent_native_kernel_pair.v1",
        "scope": "independent_npu_exploration_native_MX_kernel_vs_compiler_encoded_issuer",
        "not_a_radiance_muon_runtime_or_full_model_comparison": True,
        "source_revision": NATIVE_REVISION,
        "native_header_sha256": sha(args.native_header),
        "fixture_driver_sha256": sha(args.driver),
        "fixture_header_sha256": sha(kernel.data_header),
        "shape_mnk": kernel.shape,
        "precision": kernel.datatype,
        "compiler_tile_mnk": schedule.chunks,
        "tile_policy": args.tile_policy,
        "profile_sha256": profile_sha256(profile),
        "extension_sha256": sha(args.extlib),
        "compiler_sha256": sha(cc),
        "readelf_sha256": sha(readelf),
        "offsets": offsets,
        "commands": commands,
        "runs": [],
        "timing_boundary": "complete encoded call plus completion fence; packing and validation excluded",
        "startup_sources_sha256": {
            str(p): sha(p) for p in sources if p.parent == startup
        },
        "status": "incomplete",
    }
    for first in (0, 1):
        (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        obj, elf = out / f"driver{first}.o", out / f"pair{first}.elf"
        run(
            [
                cc,
                *flags,
                f"-DPAIR_COMPILER_FIRST={first}",
                "-c",
                out / "driver.c",
                "-o",
                obj,
            ],
            f"driver{first}",
        )
        run(
            [
                cc,
                "-nostdlib",
                "-nostartfiles",
                "-static",
                "-T",
                startup / "test.ld",
                *objects,
                obj,
                "-lm",
                "-lgcc",
                "-o",
                elf,
            ],
            f"link{first}",
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
        admission = admit_linked_device_elf(
            route,
            freeze_device_elf_admission(route),
            elf=elf,
            linked_sha256=sha(elf),
            directory=out / f"admission{first}",
        )
        environment = os.environ | {
            "GEMMINI_MODE": "both",
            "GEMMINI_PERF_CONFIG": "e4m3_vpu",
            "GEMMINI_PERF_SET": "mem.host_tracking=0",
            "GEMMINI_PERF_OUT": str(out / f"perf{first}.txt"),
        }
        run(
            [
                args.riscv_root.resolve() / "bin/spike",
                "--isa=rv64gc_zicntr",
                "--extlib=" + str(args.extlib.resolve()),
                "--extension=gemmini",
                elf,
            ],
            f"spike{first}",
            env=environment,
            timeout=3600,
        )
        console = (out / f"spike{first}.log").read_text()
        checks = check_output(console, resources["golden_bf16"].data, args.trials)
        report["runs"].append(
            {
                "compiler_first": bool(first),
                "elf_sha256": sha(elf),
                "admission": admission,
                "checks": checks,
                "environment": {
                    key: environment[key]
                    for key in (
                        "GEMMINI_MODE",
                        "GEMMINI_PERF_CONFIG",
                        "GEMMINI_PERF_SET",
                    )
                },
            }
        )
    report["status"] = "complete_full_outputs_matched"
    report["files_sha256"] = {
        p.name: sha(p)
        for p in out.iterdir()
        if p.is_file() and p.name != "receipt.json"
    }
    (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "shape_mnk": kernel.shape,
                "precision": kernel.datatype,
                "checks": [row["checks"] for row in report["runs"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
