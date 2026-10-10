"""Derive a target-mesh BF16 reference from Radiance's pinned host model.

Radiance's checked-in C_out_bf16 is computed for DIM16. The mesh precision
schedule changes for DIM8 and DIM32, so those bytes cannot be used as a
bit-exact oracle for another mesh. This module transforms only the dimension,
tile extent, and accumulator schedule in Radiance's independent host model.
It first recompiles DIM16 and requires exact agreement with the source header.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


_MODEL_CPP_SHA256 = "538e83ffa93b33cfb5ad335a90a93318f12128a0202229e64a85300b0f2b3988"
_MODEL_MATH_SHA256 = "e91f2f83ff58c8d6a4c5c1161c9df4a63fc34e7052528a0b87f4dc27116e7b60"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError(f"Radiance MX host reference changed at {before[:48]!r}")
    return text.replace(before, after, 1)


def _mesh_source(source: str, dim: int) -> str:
    source = _replace_once(source, "static const int DIM = 16;",
                           f"static const int DIM = {dim};")
    start = source.index("static const int ACC_E[16]")
    stop = source.index("static const int PROD_E", start)
    source = source[:start] + "static int ACC_E[DIM], ACC_M[DIM];\n" + source[stop:]
    source = _replace_once(source, "  using namespace mx;\n", """  using namespace mx;
  for (int kk = 0; kk < DIM; ++kk) {
    if (DIM == 8 || kk >= 15) { ACC_E[kk]=8; ACC_M[kk]=7; }
    else if (kk < 8) { ACC_E[kk]=4; ACC_M[kk]=4; }
    else if (kk < 10) { ACC_E[kk]=4; ACC_M[kk]=5; }
    else { ACC_E[kk]=4; ACC_M[kk]=6; }
  }
""")
    source = _replace_once(source, "const int TILE = sub ? 32 : DIM;",
                           "const int TILE = sub ? 2*DIM : DIM;")
    source = _replace_once(source, "float Ct[32][32] = {};",
                           "float Ct[64][64] = {};")
    source = _replace_once(source, "float A_col[32], B_row[32];",
                           "float A_col[64], B_row[64];")
    return source


def derive_mesh_reference(source_root: Path, resources: dict[str, bytes],
                          shape: tuple[int, int, int], precision: str,
                          dim: int) -> tuple[bytes, dict]:
    """Return a mesh reference and its byte-level provenance for FP8/FP4.

    The baseline DIM16 compile must reproduce the unmodified source golden.
    This catches source or arithmetic-model drift before deriving another mesh.
    """
    if dim not in {8, 32} or precision not in {"FP8", "FP4"}:
        raise ValueError("target mesh reference supports only DIM8/32 FP8/FP4 BF16")
    model_dir = source_root / "lib/golden"
    cpp_path, math_path = model_dir / "mx_golden.cpp", model_dir / "mx_fp_math.h"
    cpp, math = cpp_path.read_bytes(), math_path.read_bytes()
    if _sha(cpp) != _MODEL_CPP_SHA256 or _sha(math) != _MODEL_MATH_SHA256:
        raise ValueError("Radiance MX host reference differs from pinned source")
    m, n, k = shape
    if len(resources["golden_bf16"]) != m * n * 2:
        raise ValueError("source BF16 golden differs from requested shape")
    transformed = _mesh_source(cpp.decode(), dim)
    with TemporaryDirectory(prefix="mx-mesh-reference-") as directory:
        work = Path(directory)
        for name in ("activation", "weight", "activation_scales", "weight_scales"):
            (work / f"{name}.bin").write_bytes(resources[name])
        args = [str(m), str(n), str(k), *(
            str(work / f"{name}.bin") for name in (
                "activation", "weight", "activation_scales", "weight_scales"))]
        for selected, text in ((16, cpp.decode()), (dim, transformed)):
            source = work / f"mx_golden_dim{selected}.cpp"
            executable = work / f"mx_golden_dim{selected}"
            output = work / f"golden_dim{selected}.bin"
            source.write_text(text)
            subprocess.run(["g++", "-O2", "-std=c++17", "-I", str(model_dir),
                            "-o", str(executable), str(source)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run([str(executable), *args, str(output),
                            "0" if precision == "FP8" else "2"], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        baseline = (work / "golden_dim16.bin").read_bytes()
        if baseline != resources["golden_bf16"]:
            raise ValueError("DIM16 host model does not reproduce source BF16 golden")
        target = (work / f"golden_dim{dim}.bin").read_bytes()
    return target, {
        "schema": "mx_gemmini.radiance_target_mesh_reference.v1",
        "mesh_dim": dim, "source_golden_sha256": _sha(baseline),
        "model_cpp_sha256": _MODEL_CPP_SHA256,
        "model_math_sha256": _MODEL_MATH_SHA256,
        "transformed_cpp_sha256": _sha(transformed.encode()),
        "target_golden_sha256": _sha(target),
    }
