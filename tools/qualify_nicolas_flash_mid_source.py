"""Rebuild and replay the reference implementation’s unmodified four-block flash attention on Spike."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/nicolas_attn_flash_mid_source_baseline_a042643_0344bcb"
HEADER = ROOT / "docs/evidence/nicolas_attn_vpu_mid_source_baseline_a042643_0344bcb/attn_vpu_mid.h.gz"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, cwd: Path, log: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("npu-root", "profile", "rtl-root", "riscv-root",
                 "spike-extension", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("npu_root", "profile", "rtl_root", "riscv_root",
                 "spike_extension", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    baseline = json.loads((EVIDENCE / "receipt.json").read_text())
    npu = args.npu_root / "baremetal/mxgemmini"
    wrapper = npu / "src/attn_flash_mid.c"
    core = npu / "src/attn_flash.c"
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if (_git_revision(args.npu_root) != baseline["npu_revision"] or
            _git_revision(args.rtl_root) != baseline["rtl_revision"] or
            _sha(wrapper) != baseline["source_wrapper_sha256"] or
            _sha(core) != baseline["source_core_sha256"] or
            _sha(HEADER) != baseline["source_header_archive_sha256"] or
            _sha(EVIDENCE / "attn_flash_mid_expect.h") !=
            baseline["expect_header_sha256"] or
            profile_sha256(profile) != baseline["profile_sha256"]):
        raise ValueError("pinned flash source, header, RTL, or profile differs")
    header = gzip.decompress(HEADER.read_bytes())
    if hashlib.sha256(header).hexdigest() != baseline["source_header_sha256"]:
        raise ValueError("pinned flash source data differs")
    cc = args.riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = args.riscv_root / "bin/spike"
    if (_sha(cc) != baseline["riscv_gcc_sha256"] or
            _sha(spike) != baseline["spike_binary_sha256"] or
            _sha(args.spike_extension) != baseline["spike_extension_sha256"]):
        raise ValueError("pinned Spike toolchain or extension differs")
    out = args.out_dir
    out.mkdir(parents=True)
    (out / "attn_vpu_mid.h").write_bytes(header)
    shutil.copyfile(EVIDENCE / "attn_flash_mid_expect.h",
                    out / "attn_flash_mid_expect.h")
    software = args.rtl_root / "software/gemmini-rocc-tests"
    bench = software / "riscv-tests/benchmarks/common"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc",
             "-I", str(out), "-I", str(npu), "-I", str(software / "riscv-tests"),
             "-I", str(software / "riscv-tests/env"), "-I", str(software),
             "-I", str(bench)]
    sources = [wrapper, *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S"))]
    objects = []
    for index, source in enumerate(sources):
        obj = out / f"source_{index}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(obj)],
             cwd=out, log=out / f"compile_{index}.log")
        objects.append(obj)
    elf = out / "attn_flash_mid_source.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), *(str(obj) for obj in objects),
          "-lm", "-lgcc", "-o", str(elf)], cwd=out, log=out / "link.log")
    logs = []
    for run in (1, 2):
        log = out / f"spike_{run}.log"
        _run([str(spike), f"--extlib={args.spike_extension}",
              "--extension=gemmini", str(elf)], cwd=out, log=log)
        logs.append(log.read_text())
    expected = f'check O hash vs Spike: match ({baseline["source_output_hash"]})'
    if (logs[0] != logs[1] or expected not in logs[0] or
            "attn_flash PASSED" not in logs[0]):
        raise ValueError("flash source output differs from pinned Spike oracle")
    replay = {"schema": "mx_gemmini.flash_mid_source_replay.v1",
              "status": "original_flash_mid_source_reproduced",
              "source_baseline_sha256": _sha(EVIDENCE / "receipt.json"),
              "source_elf_sha256": _sha(elf),
              "source_spike_log_sha256": _sha(out / "spike_1.log"),
              "source_output_hash": baseline["source_output_hash"],
              "spike_reproduced_runs": 2}
    (out / "replay_receipt.json").write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n")
    print("unmodified four-block flash attention matched source hash on Spike")


if __name__ == "__main__":
    main()
