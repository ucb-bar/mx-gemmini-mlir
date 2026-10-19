"""Pack a fixed f32 matrix with the same C quantizer as the live MX adapter.

Callers must prove that the matrix is a checkpoint parameter and bind its
packed bytes to that exact checkpoint. Dynamic RHS values must keep live
packing. This module owns only the numerical conversion and byte layout.
"""

from __future__ import annotations

import ctypes
import hashlib
import subprocess
from pathlib import Path

import numpy as np

_SOURCE = Path(__file__).with_name("runtime_quant.c")
_LUT = Path(__file__).with_name("runtime_quant_lut.h")


def pack_weight(values: np.ndarray, *, precision: str,
                padded_shape: tuple[int, int], workdir: Path,
                quantizer_source: Path | None = None) -> tuple[bytes, bytes]:
    """Return MX codes and E8M0 scales, including zero-padded edge tiles."""
    if precision not in {"FP4", "FP8"}:
        raise ValueError("MX offline weight packing needs FP4 or FP8")
    if (values.ndim != 2 or values.dtype != np.float32
            or len(padded_shape) != 2
            or any(type(dim) is not int or dim <= 0 for dim in padded_shape)):
        raise ValueError("MX offline weight needs a finite rank-two f32 matrix")
    k, n = padded_shape
    if (values.shape[0] > k or values.shape[1] > n or k % 32
            or n % (32 if precision == "FP4" else 16)
            or k * n > 0x7fffffff):
        raise ValueError("MX offline weight exceeds its padded hardware shape")
    if not np.isfinite(values).all():
        raise ValueError("MX offline weight has nonfinite elements")
    padded = np.zeros((k, n), dtype=np.float32)
    padded[:values.shape[0], :values.shape[1]] = values
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    source = _SOURCE if quantizer_source is None else Path(quantizer_source)
    lut = _LUT if quantizer_source is None else source.with_name("runtime_quant_lut.h")
    header = source.with_name("runtime_quant.h")
    digest = hashlib.sha256(source.read_bytes() + lut.read_bytes() + header.read_bytes()).hexdigest()[:16]
    library = workdir / f"runtime_quant_{digest}.so"
    if not library.exists():
        result = subprocess.run(
            ["cc", "-O2", "-fno-fast-math", "-fno-common", "-fPIC", "-shared",
             "-o", str(library), str(source)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"MX host quantizer build failed: {result.stderr[-500:]}")
    pack = ctypes.CDLL(str(library.resolve())).mx_pack_weight_f32
    pack.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32,
                     ctypes.c_uint32, ctypes.c_uint32,
                     ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8)]
    pack.restype = ctypes.c_int
    codes = (ctypes.c_uint8 * (k * n // (2 if precision == "FP4" else 1)))()
    scales = (ctypes.c_uint8 * (k * n // 32))()
    status = pack(padded.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), k, n,
                  4 if precision == "FP4" else 8, codes, scales)
    if status:
        raise ValueError("MX C quantizer refused the fixed weight")
    return bytes(codes), bytes(scales)
