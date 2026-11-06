"""Pack a fixed f32 matrix with the same C quantizer as the live MX adapter.

Callers must prove that the matrix is a checkpoint parameter and bind its
packed bytes to that exact checkpoint. Dynamic RHS values must keep live
packing. This module owns only the numerical conversion and byte layout.

``quantized_weight`` instead takes bytes an external quantizer already produced in
the same layout (codes [K][N], 4-bit as [K][N/2] with the even column in the low
nibble; E8M0 scales [K/32][N]), checks them against the checkpoint values, and pads
them exactly as ``pack_weight`` pads.
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


_FORMATS = {"FP8": "fp8_e4m3", "FP4": "fp4_e2m1"}
# E8M0 byte pack_weight writes for an all-zero block: the 2**-23 block-max floor
_ZERO_BLOCK_SCALE = 104
# relative error a correct quantization stays under; a transposed or wrong
# weight lands near 1.4, so this checks identity and orientation, not accuracy
_IDENTITY_ERROR = {"FP8": 0.125, "FP4": 0.5}
_FP4_VALUES = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def _decode(codes: np.ndarray, scales: np.ndarray, precision: str) -> np.ndarray:
    if precision == "FP4":
        nibbles = np.empty((codes.shape[0], codes.shape[1] * 2), dtype=np.uint8)
        nibbles[:, 0::2], nibbles[:, 1::2] = codes & 15, codes >> 4
        values = _FP4_VALUES[nibbles & 7] * np.where(nibbles & 8, -1, 1)
    else:
        sign = np.where(codes & 128, -1.0, 1.0)
        exponent, mantissa = (codes >> 3) & 15, (codes & 7).astype(np.float64)
        values = sign * np.where(exponent == 0, mantissa / 8 * 2.0 ** -6,
                                 (1 + mantissa / 8) * 2.0 ** (exponent.astype(np.float64) - 7))
    return (values * np.repeat(2.0 ** (scales.astype(np.float64) - 127), 32, axis=0)).astype(np.float32)


def quantized_weight(source, name: str, values: np.ndarray, *, precision: str,
                     padded_shape: tuple[int, int]) -> tuple[bytes, bytes]:
    """Return ``name``'s externally quantized codes and scales, padded like ``pack_weight``.

    ``source`` is an open safetensors file holding ``<name>.codes`` and
    ``<name>.scales`` with ``{name: format}`` metadata; ``values`` is the f32
    checkpoint RHS [K][N] the codes must encode.
    """
    if precision not in _FORMATS:
        raise ValueError("MX quantized weight needs FP4 or FP8")
    fmt = (source.metadata() or {}).get(name)
    if fmt != _FORMATS[precision]:
        raise ValueError(f"MX quantized weights give {name} as {fmt}; the policy selects "
                         f"{_FORMATS[precision]}")
    if f"{name}.codes" not in source.keys() or f"{name}.scales" not in source.keys():  # noqa: SIM118
        raise ValueError(f"MX quantized weights are missing {name}")
    codes, scales = source.get_tensor(f"{name}.codes"), source.get_tensor(f"{name}.scales")
    k, n = values.shape
    per_byte = 2 if precision == "FP4" else 1
    if (k % 32 or n % per_byte or codes.dtype != np.uint8 or scales.dtype != np.uint8
            or codes.shape != (k, n // per_byte) or scales.shape != (k // 32, n)):
        raise ValueError(f"MX quantized weight {name} has codes {codes.shape} and scales "
                         f"{scales.shape}; [K][N]={values.shape} needs ({k}, {n // per_byte}) "
                         f"and ({k // 32}, {n})")
    norm = float(np.linalg.norm(values))
    error = float(np.linalg.norm(_decode(codes, scales, precision) - values))
    if error > _IDENTITY_ERROR[precision] * max(norm, np.finfo(np.float32).tiny):
        raise ValueError(f"MX quantized weight {name} does not encode its checkpoint values "
                         f"(relative error {error / norm:.3f}); wrong weight or orientation")
    pk, pn = padded_shape
    if k > pk or n > pn:
        raise ValueError(f"MX quantized weight {name} exceeds its padded hardware shape")
    padded_codes = np.zeros((pk, pn // per_byte), dtype=np.uint8)
    padded_codes[:k, :n // per_byte] = codes
    padded_scales = np.full((pk // 32, pn), _ZERO_BLOCK_SCALE, dtype=np.uint8)
    padded_scales[:k // 32, :n] = scales
    return padded_codes.tobytes(), padded_scales.tobytes()
