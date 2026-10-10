"""Derive a target-mesh BF16 reference from Radiance's pinned host model.

Radiance's checked-in C_out_bf16 is computed for DIM16. The mesh precision
schedule changes for DIM8 and DIM32, so those bytes cannot be used as a
bit-exact oracle for another mesh. This module transforms the dimension,
tile extent, and accumulator schedule in Radiance's independent host model.
The optional v2 policy also applies Nicolas's product floor to the target
run. It first recompiles DIM16 and requires exact agreement with the source
header before applying that target-only correction.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


_MODEL_CPP_SHA256 = "538e83ffa93b33cfb5ad335a90a93318f12128a0202229e64a85300b0f2b3988"
_MODEL_MATH_SHA256 = "e91f2f83ff58c8d6a4c5c1161c9df4a63fc34e7052528a0b87f4dc27116e7b60"
_MODEL_FLOOR_MATH_SHA256 = "e79560d4ba22f6dab76a2e70a1f7db9db574b7cbc16553c194c80a44fc9bf3d6"


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
                          dim: int, *, product_floor: bool = False) -> tuple[bytes, dict]:
    """Return a mesh reference and its byte-level provenance for FP8/FP4/FP6.

    The baseline DIM16 compile must reproduce the unmodified source golden.
    This catches source or arithmetic-model drift before deriving another mesh.
    """
    if dim not in {8, 32} or precision not in {"FP8", "FP4", "FP6"}:
        raise ValueError("target mesh reference supports only DIM8/32 FP8/FP4/FP6 BF16")
    model_dir = source_root / "lib/golden"
    cpp_path, math_path = model_dir / "mx_golden.cpp", model_dir / "mx_fp_math.h"
    cpp, math = cpp_path.read_bytes(), math_path.read_bytes()
    if _sha(cpp) != _MODEL_CPP_SHA256 or _sha(math) != _MODEL_MATH_SHA256:
        raise ValueError("Radiance MX host reference differs from pinned source")
    m, n, k = shape
    if len(resources["golden_bf16"]) != m * n * 2:
        raise ValueError("source BF16 golden differs from requested shape")
    transformed = _mesh_source(cpp.decode(), dim)
    floor_math = None
    if product_floor:
        floor_math = _replace_once(
            math.decode(), "  int E = e - 1;\n",
            "  int E = e - 1;\n"
            "  if (E < -16) return 0.0f;   // MxFPMul PROD_FLOOR\n")
        if _sha(floor_math.encode()) != _MODEL_FLOOR_MATH_SHA256:
            raise ValueError("RTL product-floor host reference changed")
    lut_inputs = {}
    if precision == "FP6":
        for name in ("activation_lut", "weight_lut"):
            packed = resources[name]
            if len(packed) != 64 * 12:
                raise ValueError(f"source FP6 {name} does not have 64 packed LUT lines")
            lut_inputs[name] = bytes(
                (int.from_bytes(packed[row * 12:(row + 1) * 12], "little")
                 >> (6 * code)) & 63
                for row in range(64) for code in range(16))
    with TemporaryDirectory(prefix="mx-mesh-reference-") as directory:
        work = Path(directory)
        for name in ("activation", "weight", "activation_scales", "weight_scales"):
            (work / f"{name}.bin").write_bytes(resources[name])
        for name, content in lut_inputs.items():
            (work / f"{name}.bin").write_bytes(content)
        args = [str(m), str(n), str(k), *(
            str(work / f"{name}.bin") for name in (
                "activation", "weight", "activation_scales", "weight_scales"))]
        for selected, text in ((16, cpp.decode()), (dim, transformed)):
            if selected == dim and floor_math is not None:
                (work / "mx_fp_math.h").write_text(floor_math)
            source = work / f"mx_golden_dim{selected}.cpp"
            executable = work / f"mx_golden_dim{selected}"
            output = work / f"golden_dim{selected}.bin"
            source.write_text(text)
            subprocess.run(["g++", "-O2", "-std=c++17", "-I", str(model_dir),
                            "-o", str(executable), str(source)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            command = [str(executable), *args, str(output),
                       {"FP8": "0", "FP6": "1", "FP4": "2"}[precision]]
            if precision == "FP6":
                command += [str(work / "activation_lut.bin"),
                            str(work / "weight_lut.bin"), "1"]
            subprocess.run(command, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        baseline = (work / "golden_dim16.bin").read_bytes()
        if baseline != resources["golden_bf16"]:
            raise ValueError("DIM16 host model does not reproduce source BF16 golden")
        target = (work / f"golden_dim{dim}.bin").read_bytes()
    policy = {
        "schema": ("mx_gemmini.radiance_target_mesh_reference.v2" if product_floor else
                   "mx_gemmini.radiance_target_mesh_reference.v1"),
        "mesh_dim": dim, "source_golden_sha256": _sha(baseline),
        "model_cpp_sha256": _MODEL_CPP_SHA256,
        "model_math_sha256": _MODEL_MATH_SHA256,
        "transformed_cpp_sha256": _sha(transformed.encode()),
        "target_golden_sha256": _sha(target),
    }
    if precision == "FP6":
        policy["lut_granularity_shift"] = 1
        for name, content in lut_inputs.items():
            policy[f"unpacked_{name}_sha256"] = _sha(content)
    if product_floor:
        policy["product_floor_exponent"] = -16
        policy["transformed_math_sha256"] = _MODEL_FLOOR_MATH_SHA256
    return target, policy
