"""Compare typed MX commands with Nicolas's captured SmolVLA attention heads.

Both programs run in one Spike guest, alternate after accelerator flushes,
and check the complete BF16 output after every pass. The source uses its
pipelined schedule. The typed compiler uses a fenced serial graph by default,
or queued and cross-block graphs selected by checked Merlin schedule plans.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.nicolas_attention_shape import DenseAttentionShape
from mx_gemmini_support.nicolas_flash_mid import (
    flash_buffer_abi, lower_flash_mid, render_flash_mid,
)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.emit_resident_pair_object import _compile_object


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/smolvla_attention_cutpoint_20261014"
LATE_EVIDENCE = ROOT / "docs/evidence/smolvla_attention_session_20261014"
DRIVER = ROOT / "tools/fixtures/smol_attention_paired_driver.c"
DRIVER_PIPELINE = ROOT / "tools/fixtures/smol_attention_compiler_first_driver.c"
RAW_CORE_SHA256 = "004459aa0e95def61427d5a6e9d8f71be59fcbc281a93a35745c1f33591bf7c0"
PATCHED_CORE_SHA256 = "21dd12d942cc27b27a609d3e6b95503705c8fbd018232b0320a6da9e89ac88ee"
NPU_REVISION = "0344bcba70b899039ccd0ac0801ca5605d46d3eb"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], *, cwd: Path, log: Path,
        env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, check=False,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_bytes(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def paired_rows(log: str, expected_hash: str, *, compiler_first: bool
                ) -> tuple[list[int], list[int]]:
    rows = re.findall(
        r"^paired (pipeline|compiler) ([0-3]) cycles (\d+) hash ([0-9a-f]{16})$",
        log, re.M)
    if ([(kind, int(index)) for kind, index, _, _ in rows] !=
            [("pipeline", 0), ("compiler", 1),
             ("pipeline", 2), ("compiler", 3)] or
            any(digest != expected_hash for _, _, _, digest in rows) or
            "attn_flash PASSED" not in log or
            "check O hash vs Spike: match" not in log or
            "paired full BF16 output match 4096 values" not in log or
            (compiler_first and (
                f"compiler first fresh guest hash {expected_hash}" not in log or
                "compiler first full BF16 output match 4096 values" not in log))):
        raise ValueError("paired source and compiler hashes or pass order differ")
    return ([int(row[2]) for row in rows[::2]],
            [int(row[2]) for row in rows[1::2]])


def qualify_case(keys: int, *, args: argparse.Namespace, profile: dict,
                 raw_core: Path, scratch: Path,
                 schedule_plan: str | None, schedule: str) -> dict:
    case = scratch / f"keys{keys}"
    case.mkdir()
    front = EVIDENCE / f"attention{keys}.model2mlir.mlir"
    captured_call = 1 if keys == 113 else 0
    captured_step = 0
    if args.late_session:
        captured_step = 9
        captured_call = 15 if keys == 113 else 14
        folder = LATE_EVIDENCE / f"call{captured_call:02d}"
        cutpoint = folder / "cutpoint.npz"
        header = folder / "header.h"
        source_receipt = json.loads(
            (LATE_EVIDENCE / "late_replay_receipt.json").read_text())
        rows = [entry for entry in source_receipt["rows"]
                if entry["step"] == captured_step and
                entry["call"] == captured_call and entry["keys"] == keys]
        if len(rows) != 1:
            raise ValueError("late-session source call is missing or ambiguous")
        row = rows[0]
        expected_hash = row["source_output_hash"]
        source_cutpoint_sha = row["cutpoint_sha256"]
        source_header_sha = row["header_sha256"]
    else:
        cutpoint = EVIDENCE / ("cutpoint113.npz" if keys == 113 else "cutpoint.npz")
        header = EVIDENCE / ("smol_head113.h" if keys == 113 else "smol_head0.h")
        source_receipt = json.loads((EVIDENCE / ("source_113_all_heads.json"
                                                if keys == 113 else
                                                "source_all_heads.json")).read_text())
        row = source_receipt["rows"][0]
        expected_hash = row["output_hash"]
        source_cutpoint_sha = source_receipt["cutpoint_sha256"]
        source_header_sha = row["generated_header_sha256"]
    if (source_cutpoint_sha != sha(cutpoint) or row["head"] != 0 or
            source_header_sha != sha(header)):
        raise ValueError("captured head differs from the qualified source")
    patched = case / "attn_flash_smol.c"
    shutil.copyfile(raw_core, patched)
    run(["patch", "-s", str(patched), str(EVIDENCE / "attn_flash_smol.patch")],
        cwd=case, log=case / "patch.log")
    if sha(patched) != PATCHED_CORE_SHA256:
        raise ValueError("SmolVLA source mask patch differs")
    shutil.copyfile(header, case / "smol_head0.h")
    (case / "smol_expect.h").write_text(
        f"#define EXP_FLASH_HASH_O 0x{expected_hash}ULL\n")
    shape = DenseAttentionShape(64, 128 if keys == 113 else 192, 64)
    frontend = front.read_text()
    core = patched.read_text()
    mlir, binding = render_flash_mid(
        frontend, core, core, header.read_text(), profile,
        shape=shape, block_k=64, arbitrary_mask=True,
        schedule_plan=schedule_plan)
    commands = lower_flash_mid(
        mlir, frontend, core, core, header.read_text(), profile,
        shape=shape, block_k=64, arbitrary_mask=True,
        schedule_plan=schedule_plan)
    expected_commands = ((167 if keys == 113 else 232)
                         if schedule != "fenced" else
                         (178 if keys == 113 else 248))
    if len(commands) != expected_commands:
        raise ValueError("typed arbitrary-mask command roster differs")
    (case / "attention.mx.mlir").write_text(mlir)
    (case / "binding.json").write_text(json.dumps(binding, indent=2,
                                                  sort_keys=True) + "\n")
    run([str(args.mx_opt), str(case / "attention.mx.mlir"),
         "-o", str(case / "verified.mlir")], cwd=case,
        log=case / "mx_opt.log")
    buffers, _ = flash_buffer_abi(shape, 64, arbitrary_mask=True)
    (case / "mx_issue.c").write_text(
        emit_c(list(commands), transport="rocket_rocc", buffers=buffers))
    obj, data_bytes = _compile_object(case, args.riscv_root)
    if data_bytes:
        raise ValueError("typed MX issuer embeds runtime data")
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    bench = args.rocc_root / "riscv-tests/benchmarks/common"
    elf = case / "paired.elf"
    command = [str(cc), "-DPREALLOCATE=1", "-DMULTITHREAD=1",
               "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math",
               "-fno-common", "-fno-builtin-printf",
               "-fno-tree-loop-distribute-patterns", "-march=rv64gc",
               "-Wa,-march=rv64gc"]
    for folder in (case, args.rocc_root,
                   args.rocc_root / "riscv-tests", args.riscv_env, bench):
        command.append(f"-I{folder}")
    driver = DRIVER_PIPELINE if schedule == "pipelined" else DRIVER
    command += ["-DID_STRING=", "-DPRINT_TILE=0", "-nostdlib",
                "-nostartfiles", "-static", "-T", str(bench / "test.ld"),
                "-DBAREMETAL=1", "-DSPIKE_SIM", str(driver), str(obj),
                str(bench / "crt.S"), str(bench / "syscalls.c"),
                "-lm", "-lgcc", "-o", str(elf)]
    run(command, cwd=case, log=case / "link.log")
    spike_env = os.environ.copy()
    spike_env.pop("GEMMINI_PERF_SET", None)
    spike_env.pop("GEMMINI_PERF_CONFIG", None)
    spike_env["GEMMINI_MODE"] = "both"
    if args.perf_config:
        spike_env["GEMMINI_PERF_CONFIG"] = args.perf_config
    if args.perf_set:
        spike_env["GEMMINI_PERF_SET"] = args.perf_set
    run([str(args.riscv_root / "bin/spike"), "--isa=rv64gc_zicntr",
         f"--extlib={args.spike_extension}", "--extension=gemmini",
         "-m0x80000000:0x40000000", str(elf)], cwd=case,
        log=case / "spike.log", env=spike_env)
    source_cycles, compiler_cycles = paired_rows(
        (case / "spike.log").read_text(), expected_hash,
        compiler_first=schedule == "pipelined")
    source_mean = sum(source_cycles) / len(source_cycles)
    compiler_mean = sum(compiler_cycles) / len(compiler_cycles)
    result = {
        "logical_keys": keys,
        "captured_step": captured_step,
        "captured_call": captured_call,
        "physical_shape": [shape.sq, shape.sk, shape.d],
        "cutpoint_sha256": sha(cutpoint),
        "header_sha256": sha(header),
        "model2mlir_frontend_sha256": sha(front),
        "typed_mlir_sha256": sha(case / "attention.mx.mlir"),
        "issuer_source_sha256": sha(case / "mx_issue.c"),
        "issuer_object_sha256": sha(obj),
        "elf_sha256": sha(elf),
        "expected_output_hash": expected_hash,
        "source_pipelined_cycles": source_cycles,
        "compiler_over_source_fraction": compiler_mean / source_mean - 1,
        "command_count": sum(isinstance(command, Command)
                             for command in commands),
        "fence_count": len(commands) - sum(isinstance(command, Command)
                                            for command in commands),
        "same_output": True,
    }
    result[f"compiler_{schedule}_cycles"] = compiler_cycles
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("npu_root", "profile", "rtl_root", "riscv_root", "rocc_root",
                 "riscv_env", "spike_extension", "mx_opt", "scratch_dir",
                 "archive_dir"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--schedule-plan", type=Path)
    parser.add_argument("--late-session", action="store_true",
                        help="Compare final flow step calls 14 and 15")
    parser.add_argument("--perf-config",
                        help="Spike performance config for both paired kernels")
    parser.add_argument("--perf-set",
                        help="Spike performance override for both paired kernels")
    args = parser.parse_args()
    for name in ("npu_root", "profile", "rtl_root", "riscv_root", "rocc_root",
                 "riscv_env", "spike_extension", "mx_opt", "scratch_dir",
                 "archive_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.scratch_dir.exists() or args.archive_dir.exists():
        parser.error("scratch and archive destinations must be new")
    revision = subprocess.check_output(
        ["git", "-C", str(args.npu_root), "rev-parse", "HEAD"],
        text=True).strip()
    raw_core = args.npu_root / "baremetal/mxgemmini/src/attn_flash.c"
    if revision != NPU_REVISION or sha(raw_core) != RAW_CORE_SHA256:
        raise ValueError("Nicolas source revision differs")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    schedule_plan = args.schedule_plan.read_text() if args.schedule_plan else None
    schedule = "fenced"
    if schedule_plan is not None:
        from mx_gemmini_support.merlin_schedule_policy import selected_policy
        schedule = selected_policy(schedule_plan, symbol="mx_issue")
        if schedule not in {"queued", "pipelined"}:
            raise ValueError("overlapped comparison needs a checked Merlin schedule plan")
    args.scratch_dir.mkdir(parents=True)
    rows = [qualify_case(keys, args=args, profile=profile,
                         raw_core=raw_core, scratch=args.scratch_dir,
                         schedule_plan=schedule_plan, schedule=schedule)
            for keys in (113, 163)]
    args.archive_dir.mkdir(parents=True)
    artifacts = {}
    for keys in (113, 163):
        case = args.scratch_dir / f"keys{keys}"
        target = args.archive_dir / f"keys{keys}"
        target.mkdir()
        for name in ("attention.mx.mlir", "binding.json", "mx_issue.c",
                     "mx_issue.o"):
            shutil.copyfile(case / name, target / name)
            artifacts[f"keys{keys}/{name}"] = sha(target / name)
        compressed = gzip.compress((case / "spike.log").read_bytes(), mtime=0)
        (target / "spike.log.gz").write_bytes(compressed)
        artifacts[f"keys{keys}/spike.log.gz"] = sha(target / "spike.log.gz")
    receipt = {
        "schema": (f"mx_gemmini.smolvla_masked_attention_paired_{schedule}.v1"
                   if schedule_plan is not None else
                   "mx_gemmini.smolvla_masked_attention_paired.v1"),
        "scope": ("final flow step calls 14 and 15; head 0 only"
                  if args.late_session else
                  "first 15-head call of each key shape; head 0 only"),
        "is_stitched_stage_or_full_model": False,
        "profile_sha256": profile_sha256(profile),
        "source_revision": revision,
        "original_source_sha256": sha(raw_core),
        "source_patch_sha256": sha(EVIDENCE / "attn_flash_smol.patch"),
        "patched_source_sha256": PATCHED_CORE_SHA256,
        "driver_sha256": sha(DRIVER_PIPELINE if schedule == "pipelined" else DRIVER),
        "spike_extension_sha256": sha(args.spike_extension),
        "spike_mode": ("GEMMINI_MODE=both; " +
                       (f"GEMMINI_PERF_CONFIG={args.perf_config}; "
                        if args.perf_config else "") +
                       (f"GEMMINI_PERF_SET={args.perf_set}"
                        if args.perf_set else "no performance override")),
        "timing_boundary": "same guest, source and compiler alternate after flushes",
        "compiler_schedule": schedule,
        "artifact_sha256": artifacts,
        "rows": rows,
    }
    if args.schedule_plan is not None:
        receipt["schedule_plan_sha256"] = sha(args.schedule_plan)
    (args.archive_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
