"""Build a source and object-backed flash driver and compare full output on Spike."""

from __future__ import annotations

from pathlib import Path

from tools.qualify_nicolas_flash_mid_typed import _output_words
from tools.qualify_nicolas_flash_qk0 import _run


def run_flash_spike_parity(*, out: Path, npu_root: Path, rtl_root: Path,
                           riscv_root: Path, spike_extension: Path,
                           object_path: Path, slug: str,
                           expected_hash: str, output_words: int) -> None:
    """Run both drivers twice; reject any BF16 mismatch across the four runs."""
    cc = riscv_root / "bin/riscv64-unknown-elf-gcc"
    spike = riscv_root / "bin/spike"
    software = rtl_root / "software/gemmini-rocc-tests"
    bench = software / "riscv-tests/benchmarks/common"
    flags = ["-DPREALLOCATE=1", "-DMULTITHREAD=1", "-DMX_ROCKET", "-DBAREMETAL=1",
             "-mcmodel=medany", "-std=gnu99", "-O2", "-ffast-math", "-fno-common",
             "-fno-builtin-printf", "-fno-tree-loop-distribute-patterns",
             "-march=rv64gc", "-Wa,-march=rv64gc"]
    for path in (out, npu_root / "baremetal/mxgemmini",
                 npu_root / "baremetal/mxgemmini/src",
                 software / "riscv-tests", software / "riscv-tests/env", software, bench):
        flags.extend(("-I", str(path)))
    common = []
    for index, source in enumerate([*sorted(bench.glob("*.c")),
                                    *sorted(bench.glob("*.S"))]):
        obj = out / f"common_{index}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(obj)],
             cwd=out, log=out / f"common_compile_{index}.log")
        common.append(obj)
    words = []
    for kind in ("source", "generated"):
        driver_obj = out / f"driver_{kind}.o"
        _run([str(cc), *flags, *(["-DGENERATED"] if kind == "generated" else []),
              "-c", str(out / "driver.c"), "-o", str(driver_obj)],
             cwd=out, log=out / f"driver_{kind}_compile.log")
        elf = out / f"{slug}_{kind}.elf"
        _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
              str(bench / "test.ld"), str(driver_obj),
              *([str(object_path)] if kind == "generated" else []),
              *(str(path) for path in common), "-lm", "-lgcc", "-o", str(elf)],
             cwd=out, log=out / f"link_{kind}.log")
        for run in (1, 2):
            log = out / f"{kind}_spike_{run}.log"
            _run([str(spike), f"--extlib={spike_extension}",
                  "--extension=gemmini", str(elf)], cwd=out, log=log)
            words.append(_output_words(log, expected_hash, output_words))
    if len(set(words)) != 1:
        raise ValueError("compiler flash output differs from source BF16 words")
