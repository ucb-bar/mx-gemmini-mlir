"""Owned RV64 catalog ABI and C compilation options."""

from pathlib import Path


def native_catalog_compile_flags(*, far_constant_data=False):
    if type(far_constant_data) is not bool:
        raise ValueError("far constant data requires an explicit boolean")
    flags = [
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-ffreestanding",
        "-fno-fast-math",
        "-fno-common",
        "-fno-tree-loop-distribute-patterns",
        "-std=gnu11",
        "-I",
        str(Path(__file__).resolve().parent),
    ]
    if far_constant_data:
        flags.append("-fPIC")
    return flags
