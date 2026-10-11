"""Compile whole four-block flash attention and compare every BF16 output on Spike."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

from mx_gemmini_support.command_ir import Command, emit_c
from mx_gemmini_support.nicolas_flash_mid import (FA_SHAPE, SHAPE,
                                                  flash_buffer_abi,
                                                  lower_flash_mid,
                                                  render_flash_mid)
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _source_closure
from tools.emit_resident_pair_object import _compile_object
from tools.qualify_nicolas_flash_qk0 import (DATA_SOURCE, FLASH_SOURCE,
                                             _run, _sha)


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tools/fixtures/nicolas_flash_mid_serial_driver.c"
EXPECTED_HASH = "cd7aef4d68d1e0d9"
FA_DATA_SOURCE = ROOT / "docs/evidence/nicolas_attn_vpu_fa_source_baseline_a042643_0344bcb"
FA_DRIVER = ROOT / "tools/fixtures/nicolas_flash_fa_serial_driver.c"
FA_EXPECTED_HASH = "fba97c66241b47bf"


def _output_words(log: Path, expected_hash: str, count: int) -> str:
    data = log.read_text()
    match = re.search(r"O hash ([0-9a-f]{16})\nO BF16 words ([0-9a-f]+)\n", data)
    if match is None or match.group(1) != expected_hash or len(match.group(2)) != count * 4:
        raise ValueError(f"flash output differs from source golden; see {log}")
    return match.group(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("mid", "fa"), default="mid")
    for name in ("npu-root", "profile", "rtl-root", "riscv-root",
                 "spike-extension", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("npu_root", "profile", "rtl_root", "riscv_root",
                 "spike_extension", "mx_opt", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    fa = args.variant == "fa"
    baseline = json.loads((FLASH_SOURCE / "receipt.json").read_text())
    data_source = FA_DATA_SOURCE if fa else DATA_SOURCE
    data_baseline = json.loads((data_source / "receipt.json").read_text())
    shape, block_k = (FA_SHAPE, 64) if fa else (SHAPE, 32)
    buffers, buffer_bytes = flash_buffer_abi(shape, block_k)
    driver = FA_DRIVER if fa else DRIVER
    expected_hash = FA_EXPECTED_HASH if fa else EXPECTED_HASH
    slug = "flash_fa" if fa else "flash_mid"
    npu = args.npu_root / "baremetal/mxgemmini"
    core = npu / "src/attn_flash.c"
    wrapper = core if fa else npu / "src/attn_flash_mid.c"
    header_name = "attn_vpu_fa.h" if fa else "attn_vpu_mid.h"
    expect_name = "attn_flash_expect.h" if fa else "attn_flash_mid_expect.h"
    header_archive = data_source / f"{header_name}.gz"
    frontend_path = data_source / "attention.model2mlir.mlir"
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if (_git_revision(args.npu_root) != baseline["npu_revision"] or
            _git_revision(args.rtl_root) != baseline["rtl_revision"] or
            profile_sha256(profile) != data_baseline["profile_sha256"] or
            (not fa and _sha(wrapper) != baseline["source_wrapper_sha256"]) or
            _sha(core) != baseline["source_core_sha256"] or
            _sha(header_archive) != data_baseline["source_header_gzip_sha256"] or
            _sha(frontend_path) != data_baseline["frontend_mlir_sha256"]):
        raise ValueError("pinned flash source, frontend, RTL, or profile differs")
    header = gzip.decompress(header_archive.read_bytes())
    if hashlib.sha256(header).hexdigest() != data_baseline["source_header_sha256"]:
        raise ValueError("pinned flash data header differs")
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if (_sha(cc) != data_baseline["riscv_gcc_sha256"] or
            _sha(spike) != data_baseline["spike_binary_sha256"] or
            _sha(args.spike_extension) != data_baseline["spike_extension_sha256"]):
        raise ValueError("pinned RV64 toolchain or Spike extension differs")
    frontend = frontend_path.read_text()
    mlir, binding = render_flash_mid(
        frontend, core.read_text(), wrapper.read_text(), header.decode(), profile,
        shape=shape, block_k=block_k)
    commands = lower_flash_mid(
        mlir, frontend, core.read_text(), wrapper.read_text(), header.decode(), profile,
        shape=shape, block_k=block_k)
    command_count = sum(isinstance(command, Command) for command in commands)
    if (command_count, len(commands) - command_count) != (
            236 if fa else 156, 21):
        raise ValueError("typed flash mid command inventory differs")
    out = args.out_dir
    out.mkdir(parents=True)
    (out / header_name).write_bytes(header)
    if fa:
        (out / expect_name).write_text(
            f"#define EXP_FLASH_HASH_O 0x{expected_hash}ULL\n")
    else:
        shutil.copyfile(FLASH_SOURCE / expect_name, out / expect_name)
    shutil.copyfile(driver, out / "driver.c")
    (out / f"{slug}.mlir").write_text(mlir)
    (out / "binding.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
    _run([str(args.mx_opt), str(out / f"{slug}.mlir"), "-o", "/dev/null"],
         cwd=out, log=out / "native_verify.log")
    issuer = out / "mx_issue.c"
    issuer.write_text(emit_c(list(commands), transport="rocket_rocc", buffers=buffers))
    (out / "mx_issue.h").write_text(
        "#ifndef MX_ISSUE_H\n#define MX_ISSUE_H\n"
        "void mx_issue(" + ", ".join(f"const void *{name}" for name in buffers) +
        ");\n#endif\n")
    physical = out / "physical_program.json"
    physical.write_text(json.dumps({
        "schema": f"mx_gemmini.{slug}_serial_physical.v1",
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": _sha(out / f"{slug}.mlir"),
        "commands": [asdict(command) if isinstance(command, Command)
                     else {"kind": "fence"} for command in commands],
    }, indent=2, sort_keys=True) + "\n")
    obj, data_bytes = _compile_object(out, args.riscv_root)
    if data_bytes != 0:
        raise ValueError("typed flash mid object unexpectedly embeds data")

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
    outputs = {}
    for kind in ("source", "generated"):
        driver_obj = out / f"driver_{kind}.o"
        _run([str(cc), *flags, *(["-DGENERATED"] if kind == "generated" else []),
              "-c", str(out / "driver.c"), "-o", str(driver_obj)],
             cwd=out, log=out / f"driver_{kind}_compile.log")
        elf = out / f"{slug}_{kind}.elf"
        _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
              str(bench / "test.ld"), str(driver_obj),
              *([str(obj)] if kind == "generated" else []),
              *(str(path) for path in objects), "-lm", "-lgcc", "-o", str(elf)],
             cwd=out, log=out / f"link_{kind}.log")
        for run in (1, 2):
            log = out / f"{kind}_spike_{run}.log"
            _run([str(spike), f"--extlib={args.spike_extension}",
                  "--extension=gemmini", str(elf)], cwd=out, log=log)
            outputs[(kind, run)] = _output_words(log, expected_hash, shape.sq * shape.d)
    if len(set(outputs.values())) != 1:
        raise ValueError("compiler-generated flash output differs from source BF16 words")
    object_manifest = {
        "schema": f"mx_gemmini.{slug}_serial_linkable_object.v1",
        "status": f"rv64_rocc_typed_{slug}_serial_spike_matched",
        "transport": "rocket_rocc", "schedule": "fenced_serial",
        "buffer_abi": [{"name": name, "position": index,
                        "role": "write" if name in {"scale_sink", "p_scales_0",
                                                    "p_scales_1", "O_HW"} else "read",
                        "minimum_bytes": length, "alignment_bytes": 64,
                        "layout": f"flat_bytes_source_{slug}"}
                       for index, (name, length) in enumerate(zip(buffers, buffer_bytes))],
        "command_count": command_count, "fence_count": len(commands) - command_count,
        "embedded_operand_bytes": 0, "embedded_golden_bytes": 0,
        "allocated_data_section_bytes": data_bytes,
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": _sha(out / f"{slug}.mlir"),
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
    receipt = {"schema": f"mx_gemmini.{slug}_serial_spike_parity.v1",
               "status": f"typed_{slug}_full_bf16_source_parity",
               "schedule": "fenced_serial",
               "source_baseline_sha256": _sha(FLASH_SOURCE / "receipt.json"),
               "data_baseline_sha256": _sha(data_source / "receipt.json"),
               "model2mlir_revision": data_baseline["model2mlir_revision"],
               "source_output_hash": expected_hash,
               "compared_bf16_words": shape.sq * shape.d, "source_runs": 2,
               "compiler_runs": 2, "bound_mlir_sha256": _sha(out / f"{slug}.mlir"),
               "object_sha256": _sha(obj),
               "source_elf_sha256": _sha(out / f"{slug}_source.elf"),
               "generated_elf_sha256": _sha(out / f"{slug}_generated.elf"),
               "source_log_sha256": _sha(out / "source_spike_1.log"),
               "generated_log_sha256": _sha(out / "generated_spike_1.log"),
               "physical_program_sha256": _sha(physical),
               "object_manifest_sha256": _sha(out / "object_manifest.json"),
               "driver_sha256": _sha(driver),
               "native_verifier_sha256": _sha(args.mx_opt),
               "spike_extension_sha256": _sha(args.spike_extension)}
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(f"typed flash {args.variant}: all {shape.sq * shape.d:,} BF16 outputs "
          "match source twice on Spike")


if __name__ == "__main__":
    main()
