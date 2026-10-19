"""Compile typed flash QK0 and compare every BF16 score to the reference implementation’s source on Spike."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.nicolas_flash_qk0 import (BUFFERS, BUFFER_BYTES,
                                                  lower_flash_qk0,
                                                  render_flash_qk0)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _source_closure
from tools.emit_resident_pair_object import _compile_object


ROOT = Path(__file__).resolve().parents[1]
FLASH_SOURCE = ROOT / "docs/evidence/nicolas_attn_flash_mid_source_baseline_a042643_0344bcb"
DATA_SOURCE = ROOT / "docs/evidence/nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb"
DRIVER = ROOT / "tools/fixtures/nicolas_flash_qk0_driver.c"
EXPECTED_HASH = "52bca547381dac69"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, cwd: Path, log: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def _score_words(log: Path) -> str:
    data = log.read_text()
    match = re.search(r"QK0 score hash ([0-9a-f]{16})\nQK0 BF16 words ([0-9a-f]+)\n", data)
    if match is None or match.group(1) != EXPECTED_HASH or len(match.group(2)) != 2048 * 4:
        raise ValueError(f"flash QK0 source or compiler score differs; see {log}")
    return match.group(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("npu-root", "profile", "rtl-root", "riscv-root",
                 "spike-extension", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("npu_root", "profile", "rtl_root", "riscv_root",
                 "spike_extension", "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    baseline = json.loads((FLASH_SOURCE / "receipt.json").read_text())
    data_baseline = json.loads((DATA_SOURCE / "receipt.json").read_text())
    npu = args.npu_root / "baremetal/mxgemmini"
    wrapper = npu / "src/attn_flash_mid.c"
    core = npu / "src/attn_flash.c"
    header_archive = DATA_SOURCE / "attn_vpu_mid.h.gz"
    frontend_path = DATA_SOURCE / "attention.model2mlir.mlir"
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if (_git_revision(args.npu_root) != baseline["npu_revision"] or
            _git_revision(args.rtl_root) != baseline["rtl_revision"] or
            profile_sha256(profile) != baseline["profile_sha256"] or
            _sha(wrapper) != baseline["source_wrapper_sha256"] or
            _sha(core) != baseline["source_core_sha256"] or
            _sha(header_archive) != baseline["source_header_archive_sha256"] or
            _sha(frontend_path) != baseline["frontend_mlir_sha256"]):
        raise ValueError("pinned flash source, frontend, RTL, or profile differs")
    header = gzip.decompress(header_archive.read_bytes())
    if hashlib.sha256(header).hexdigest() != baseline["source_header_sha256"]:
        raise ValueError("pinned flash data header differs")
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if (_sha(cc) != baseline["riscv_gcc_sha256"] or
            _sha(spike) != baseline["spike_binary_sha256"] or
            _sha(args.spike_extension) != baseline["spike_extension_sha256"]):
        raise ValueError("pinned RV64 toolchain or Spike extension differs")
    frontend = frontend_path.read_text()
    mlir, binding = render_flash_qk0(
        frontend, core.read_text(), wrapper.read_text(), header.decode(), profile)
    commands = lower_flash_qk0(
        mlir, frontend, core.read_text(), wrapper.read_text(), header.decode(), profile)
    if (sum(isinstance(command, Command) for command in commands),
            len(commands) - sum(isinstance(command, Command) for command in commands)) != (35, 1):
        raise ValueError("typed flash QK0 command inventory differs")
    out = args.out_dir
    out.mkdir(parents=True)
    (out / "attn_vpu_mid.h").write_bytes(header)
    shutil.copyfile(FLASH_SOURCE / "attn_flash_mid_expect.h",
                    out / "attn_flash_mid_expect.h")
    shutil.copyfile(DRIVER, out / "driver.c")
    (out / "flash_qk0.mlir").write_text(mlir)
    (out / "binding.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    _run([str(args.mx_opt), str(out / "flash_qk0.mlir"), "-o", "/dev/null"],
         cwd=out, log=out / "native_verify.log")
    issuer = out / "mx_issue.c"
    issuer.write_text(emit_c(list(commands), transport="rocket_rocc", buffers=BUFFERS))
    (out / "mx_issue.h").write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n"
        "void mx_issue(" + ", ".join(f"const void *{name}" for name in BUFFERS) +
        ");\n#endif\n")
    physical = out / "physical_program.json"
    physical.write_text(json.dumps({
        "schema": "mx_gemmini.flash_qk0_physical.v1",
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": _sha(out / "flash_qk0.mlir"),
        "commands": [asdict(command) if isinstance(command, Command)
                     else {"kind": "fence"} for command in commands],
    }, indent=2, sort_keys=True) + "\n")
    obj, data_bytes = _compile_object(out, args.riscv_root)
    if data_bytes != 0:
        raise ValueError("typed flash QK0 object unexpectedly embeds data")

    software = args.rtl_root / "software/gemmini-rocc-tests"
    bench = software / "riscv-tests/benchmarks/common"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc"]
    for path in (out, npu, npu / "src", software / "riscv-tests",
                 software / "riscv-tests/env", software, bench):
        flags.extend(("-I", str(path)))
    objects = []
    for index, source in enumerate([*sorted(bench.glob("*.c")),
                                    *sorted(bench.glob("*.S"))]):
        obj_common = out / f"common_{index}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(obj_common)],
             cwd=out, log=out / f"common_compile_{index}.log")
        objects.append(obj_common)
    logs = {}
    for kind in ("source", "generated"):
        driver_obj = out / f"driver_{kind}.o"
        _run([str(cc), *flags, *(["-DGENERATED"] if kind == "generated" else []),
              "-c", str(out / "driver.c"), "-o", str(driver_obj)],
             cwd=out, log=out / f"driver_{kind}_compile.log")
        elf = out / f"flash_qk0_{kind}.elf"
        _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
              str(bench / "test.ld"), str(driver_obj),
              *([str(obj)] if kind == "generated" else []),
              *(str(path) for path in objects), "-lm", "-lgcc", "-o", str(elf)],
             cwd=out, log=out / f"link_{kind}.log")
        for run in (1, 2):
            log = out / f"{kind}_spike_{run}.log"
            _run([str(spike), f"--extlib={args.spike_extension}",
                  "--extension=gemmini", str(elf)], cwd=out, log=log)
            logs[(kind, run)] = _score_words(log)
    if len(set(logs.values())) != 1:
        raise ValueError("compiler-generated QK0 scores differ from source BF16 words")
    object_manifest = {
        "schema": "mx_gemmini.flash_qk0_linkable_object.v1",
        "status": "rv64_rocc_typed_flash_qk0_object_spike_matched",
        "transport": "rocket_rocc",
        "buffer_abi": [{"name": name, "position": index,
                        "role": "write" if name in {"scale_sink", "S0_HW"} else "read",
                        "minimum_bytes": length, "alignment_bytes": 64,
                        "layout": "flat_bytes_source_flash_qk0"}
                       for index, (name, length) in enumerate(zip(BUFFERS, BUFFER_BYTES))],
        "command_count": 35, "fence_count": 1,
        "embedded_operand_bytes": 0, "embedded_golden_bytes": 0,
        "allocated_data_section_bytes": data_bytes,
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": _sha(out / "flash_qk0.mlir"),
        "physical_program_sha256": _sha(physical),
        "issuer_c_sha256": _sha(issuer), "object_sha256": _sha(obj),
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "rtl_revision": _git_revision(args.rtl_root),
        "riscv_gcc_sha256": _sha(cc), "defined_symbol": "mx_issue",
        "undefined_symbols": [],
    }
    (out / "object_manifest.json").write_text(
        json.dumps(object_manifest, indent=2, sort_keys=True) + "\n")
    receipt = {"schema": "mx_gemmini.flash_qk0_spike_parity.v1",
               "status": "typed_flash_qk0_full_bf16_source_parity",
               "source_baseline_sha256": _sha(FLASH_SOURCE / "receipt.json"),
               "data_baseline_sha256": _sha(DATA_SOURCE / "receipt.json"),
               "model2mlir_revision": data_baseline["model2mlir_revision"],
               "source_score_hash": EXPECTED_HASH,
               "compared_bf16_words": 2048, "source_runs": 2,
               "compiler_runs": 2, "bound_mlir_sha256": _sha(out / "flash_qk0.mlir"),
               "object_sha256": _sha(obj),
               "source_elf_sha256": _sha(out / "flash_qk0_source.elf"),
               "generated_elf_sha256": _sha(out / "flash_qk0_generated.elf"),
               "source_log_sha256": _sha(out / "source_spike_1.log"),
               "generated_log_sha256": _sha(out / "generated_spike_1.log"),
               "physical_program_sha256": _sha(physical),
               "object_manifest_sha256": _sha(out / "object_manifest.json"),
               "driver_sha256": _sha(DRIVER),
               "native_verifier_sha256": _sha(args.mx_opt),
               "spike_extension_sha256": _sha(args.spike_extension)}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("typed flash QK0: all 2,048 BF16 score words match source twice on Spike")


if __name__ == "__main__":
    main()
