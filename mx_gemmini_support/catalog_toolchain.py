"""Owned RV64 catalog ABI and C compilation options."""

from pathlib import Path


def native_catalog_compile_flags():
    return ["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2",
            "-ffreestanding", "-fno-fast-math", "-fno-common",
            "-fno-tree-loop-distribute-patterns", "-std=gnu11",
            "-I", str(Path(__file__).resolve().parent)]
