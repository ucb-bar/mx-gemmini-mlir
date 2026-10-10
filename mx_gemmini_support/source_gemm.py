"""Read the committed Radiance MX GEMM drivers and derive their tile schedule.

The source kernels use one output tile, a double-buffered A/B scratchpad, and
one K loop. This module retains those constraints so the first matrix lowering
can compare its placement and command order against the actual source files.
It does not claim numerical or executable command equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


_DATATYPES = {"FP8": ("fp8_e4m3", "direct", 16, 1),
              "FP6": ("fp6_e3m2", "lut", 32, 2),
              "FP4": ("fp4_e2m1", "direct", 32, 2)}


def _match(pattern: str, source: str, label: str) -> str:
    found = re.search(pattern, source, re.MULTILINE | re.DOTALL)
    if found is None:
        raise ValueError(f"source MX GEMM lacks {label}")
    return found.group(1)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class SourceGemm:
    driver: Path
    data_header: Path
    shape: tuple[int, int, int]
    tile: tuple[int, int, int]
    datatype: str
    quant_output: bool
    acc_to_gmem: bool
    data_header_present: bool


def read_source_gemm(driver: Path) -> SourceGemm:
    """Extract literal source facts; reject expressions this audit cannot prove."""
    source = driver.read_text()
    if driver.name == "kernel.cpp" and driver.parent.name == "gemm_mxgemmini_ws_restream":
        return _read_restream_gemm(driver, source)
    if driver.name == "kernel.cpp" and driver.parent.name in {
            "gemm_mxgemmini_ws", "gemm_mxgemmini_ws_downproj_fp4"}:
        return _read_weight_stationary_gemm(driver, source)
    if driver.name == "kernel.cpp" and driver.parent.name in {
            "gemv_batched_fp8_m32", "gemv_batched_fp8_m64",
            "gemv_batched_fp8_m128", "gemv_batched_fp4_m128"}:
        return _read_batched_gemv(driver, source)
    header_name = _match(r'^#include "(mxgemm\.data\.[^"/]+\.h)"', source, "data header")
    header = driver.parent / header_name
    declared = re.fullmatch(r'mxgemm\.data\.(fp[468])\.m(\d+)n(\d+)k(\d+)\.h', header_name)
    if declared is None:
        raise ValueError("source MX GEMM data header name has unknown shape or precision")
    shape = tuple(int(value) for value in declared.groups()[1:])
    if header.is_file():
        header_text = header.read_text()
        header_shape = tuple(int(_match(rf'^#define MATMUL_{axis}\s+(\d+)\b', header_text,
                                        f"MATMUL_{axis}")) for axis in ("M", "N", "K"))
        if header_shape != shape:
            raise ValueError("source MX GEMM data header dimensions differ from its name")
    config = _match(r'constexpr GemmConfig C\s*\{(.*?)\};', source, "GemmConfig")
    tile = tuple(int(_match(rf'\.TILE_{axis}\s*=\s*(\d+)\b', config, f"TILE_{axis}"))
                 for axis in ("M", "N", "K"))
    datatype = _match(r'\.DATATYPE\s*=\s*GemmDatatype::(FP[468])\b', config, "DATATYPE")
    if datatype.lower() != declared.group(1):
        raise ValueError("source MX GEMM datatype differs from included data header")
    quant = _match(r'\.QUANT_OUTPUT\s*=\s*(true|false)\b', config, "QUANT_OUTPUT") == "true"
    acc = re.search(r'\.ACC_TO_GMEM\s*=\s*(true|false)\b', config)
    acc_to_gmem = acc is not None and acc.group(1) == "true"
    call = _match(r'mxgemm<C>\s*\((.*?)\);', source, "mxgemm<C> call")
    args = [arg.strip() for arg in call.split(",")[:3]]
    if len(args) != 3:
        raise ValueError("source MX GEMM has no three literal dimensions")
    symbols = {"C.TILE_M": tile[0], "C.TILE_N": tile[1], "C.TILE_K": tile[2],
               "MATMUL_M": shape[0], "MATMUL_N": shape[1], "MATMUL_K": shape[2]}
    evaluated = tuple(symbols.get(arg, int(arg) if arg.isdecimal() else None) for arg in args)
    if evaluated != shape:
        raise ValueError(f"driver call dimensions {evaluated} differ from data header {shape}")
    return SourceGemm(driver, header, shape, tile, datatype, quant, acc_to_gmem,
                      header.is_file())


def _read_weight_stationary_gemm(driver: Path, source: str) -> SourceGemm:
    """Read the two Radiance read-once MX kernels without inventing a driver."""
    if (source.count('#include "data"') != 1 or
            source.count('#include "mxgemm_lib.hpp"') != 1 or
            re.search(r'mxgemm<CFG>\s*\(MATMUL_M,\s*MATMUL_N,\s*MATMUL_K,',
                      source) is None):
        raise ValueError("weight-stationary MX source no longer calls the shared GEMM library")
    header = driver.parent / "data"
    if not header.is_file():
        raise ValueError("weight-stationary MX source data is missing; run its pinned generator")
    data = header.read_text(encoding="ascii")
    shape = tuple(int(_match(rf'^#define MATMUL_{axis}\s+(\d+)\b', data,
                             f"MATMUL_{axis}")) for axis in ("M", "N", "K"))
    config = _match(r'constexpr GemmConfig CFG\s*\{(.*?)\};', source, "GemmConfig CFG")
    tile = tuple(int(_match(rf'\.TILE_{axis}\s*=\s*(\d+)\b', config,
                            f"TILE_{axis}")) for axis in ("M", "N", "K"))
    datatype = _match(r'\.DATATYPE\s*=\s*GemmDatatype::(FP[468])\b',
                      config, "DATATYPE")
    quant = _match(r'\.QUANT_OUTPUT\s*=\s*(true|false)\b',
                   config, "QUANT_OUTPUT") == "true"
    expected = ((256, 64, 2048), (256, 64, 64), "FP8") if (
        driver.parent.name == "gemm_mxgemmini_ws") else (
        (256, 64, 5632), (256, 64, 64), "FP4")
    if (shape, tile, datatype) != expected or quant:
        raise ValueError("weight-stationary MX source shape, tile, or format changed")
    return SourceGemm(driver, header, shape, tile, datatype, False, False, True)


def _read_batched_gemv(driver: Path, source: str) -> SourceGemm:
    """Bind the four committed decode projections, including non-square tiles."""
    required = (
        '#include "data"', '#include "mxgemm_lib.hpp"',
        'mxgemm<GEMM_CFG>(a->M,a->N,a->K,a->C,tid,tpb,tbid);',
        'kernel_args={reinterpret_cast<uint8_t*>(reinterpret_cast<uint32_t>((uint16_t*)C_raw)),MATMUL_M,MATMUL_N,MATMUL_K};',
    )
    if any(source.count(marker) != 1 for marker in required):
        raise ValueError("batched GEMV source call or shape binding changed")
    header = driver.parent / "data"
    if not header.is_file():
        raise ValueError("batched GEMV source data is missing; run its pinned generator")
    data = header.read_text(encoding="ascii")
    shape = tuple(int(_match(rf'^#define MATMUL_{axis}\s+(\d+)\b', data,
                             f"MATMUL_{axis}")) for axis in ("M", "N", "K"))
    config = _match(r'constexpr GemmConfig GEMM_CFG\s*\{(.*?)\};', source,
                    "GemmConfig GEMM_CFG")
    tile = tuple(shape[("M", "N", "K").index(axis)] if value == f"MATMUL_{axis}"
                 else int(value) if value.isdecimal() else None
                 for axis in ("M", "N", "K")
                 for value in [_match(rf'\.TILE_{axis}\s*=\s*(MATMUL_{axis}|\d+)\b',
                                      config, f"TILE_{axis}")])
    datatype = _match(r'\.DATATYPE\s*=\s*GemmDatatype::(FP[48])\b',
                      config, "DATATYPE")
    quant = _match(r'\.QUANT_OUTPUT\s*=\s*(true|false)\b',
                   config, "QUANT_OUTPUT") == "true"
    expected_m = int(_match(r'^gemv_batched_fp[48]_m(\d+)$',
                            driver.parent.name, "batch size"))
    expected_precision = "FP4" if "_fp4_" in driver.parent.name else "FP8"
    if (shape != (expected_m, 128, 2048) or tile != (expected_m, 128, 128)
            or datatype != expected_precision or quant):
        raise ValueError("batched GEMV shape, tile, or format differs from source variant")
    return SourceGemm(driver, header, shape, tile, datatype, False, False, True)


def _read_restream_gemm(driver: Path, source: str) -> SourceGemm:
    """Read the committed four-M-block counterexample to read-once B traffic."""
    required = (
        '#include "data"', '#include "mxgemm_lib.hpp"',
        '#define WS_MB   64', '#define WS_MT   4',
        '.TILE_M=WS_MB', '.TILE_N=64', '.TILE_K=64',
        '.DATATYPE=GemmDatatype::FP8', '.QUANT_OUTPUT=false',
        'for (uint32_t m=0; m<WS_MT; m++)',
        'mxgemm_restream<CFG>(MATMUL_K, a->C, tid, tpb);',
        'A_in = &A_in_data[0][0] + (uint32_t)m*WS_MB*MATMUL_K;',
        'const uint8_t *A_sc = &A_scales_tiled[0][0] + (uint32_t)m*MATMUL_GK*WS_MB;',
        'const uint8_t *B_sc = &B_scales_col[0][0];',
    )
    if any(source.count(marker) != 1 for marker in required):
        raise ValueError("source re-stream MX schedule changed")
    header = driver.parent / "data"
    if not header.is_file():
        raise ValueError("committed re-stream MX source data is missing")
    data = header.read_text(encoding="ascii")
    shape = tuple(int(_match(rf'^#define MATMUL_{axis}\s+(\d+)\b', data,
                             f"MATMUL_{axis}")) for axis in ("M", "N", "K"))
    if shape != (256, 64, 2048):
        raise ValueError("source re-stream MX data shape changed")
    return SourceGemm(driver, header, shape, (64, 64, 64), "FP8", False, False, True)


def source_scratchpad_bytes(library: Path) -> int:
    source = library.read_text()
    verify_source_schedule(library, source=source)
    factor = _match(r'static_assert\(BANK_NUM \* BANK_ROWS \* DIM == \((\d+) \* 1024\)',
                    source, "scratchpad geometry assertion")
    return int(factor) * 1024


def verify_source_schedule(library: Path, *, source: str | None = None) -> None:
    """Fail when the handwritten K-loop no longer matches the modeled order."""
    source = library.read_text() if source is None else source
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.S)
    source = re.sub(r'//[^\n]*', '', source)
    for name, value in (("GEMMINI_DMA", "true"),
                        ("DISABLE_MOVE_IN_AFTER_FIRST_K", "false"),
                        ("DISABLE_SCALE_FACTOR_UPDATE", "false"),
                        ("SIMT_GMEM_MOVE_OUT", "true")):
        if re.search(rf'constexpr bool {name}\s*=\s*{value}\s*;', source) is None:
            raise ValueError(f"source MX GEMM scheduling flag changed: {name}")
    if 'const auto odd_k = (tile_k & 1);' not in source:
        raise ValueError("source MX GEMM double-buffer toggle changed")
    if 'matmul_tile_async<C>(tile_k, last_k && !C.ACC_TO_GMEM);' not in source:
        raise ValueError("source MX GEMM accumulator drain changed")
    start = source.find('void mxgemm_single_output_tile(')
    stop = source.find('static void\nmxgemm(', start)
    if start < 0 or stop < 0:
        raise ValueError("source MX GEMM K-loop body changed")
    body = source[start:stop]
    ordered = (
        'configure_mxgemmini<C>(', 'copy_gmem_to_smem_async<C>(',
        'load_scale_factors(', 'load_scale_factors(', 'load_lut<C>();',
        'mu_fence_smem();', 'gemmini_fence();',
        'for (; (tile_k * C.TILE_K) < dim_k; tile_k++)',
        'gemmini_mxquant_config_mvout(', 'copy_gmem_to_smem_async<C>(',
        'matmul_tile_async<C>(', 'load_scale_factors(', 'load_scale_factors(',
        'mu_fence_smem();', 'gemmini_fence();',
    )
    cursor = 0
    for marker in ordered:
        position = body.find(marker, cursor)
        if position < 0:
            raise ValueError(f"source MX GEMM K-loop order changed at {marker}")
        cursor = position + len(marker)


def _destination(rows: int, a: int, b: int, c: int) -> int:
    quarter = rows // 4
    odd_b_low = rows - quarter - b
    even_b_low = rows - b
    if a <= quarter and quarter - a >= c:
        return a
    if quarter + a <= odd_b_low and odd_b_low - (quarter + a) >= c:
        return quarter + a
    if rows - quarter <= even_b_low and even_b_low - (rows - quarter) >= c:
        return rows - quarter
    return 0


def plan_mx_gemm(*, shape: tuple[int, int, int], tile: tuple[int, int, int],
                 datatype: str, quant_output: bool, acc_to_gmem: bool,
                 scratchpad_bytes: int, profile: dict | None = None) -> dict:
    """Plan a physical MX tile from typed dimensions and target resources."""
    m, n, k = shape
    tm, tn, tk = tile
    named, projection, base_pe, values_per_byte = _DATATYPES[datatype]
    dim = profile["geometry"]["mesh_columns"] if profile is not None else 16
    pe = dim * values_per_byte
    if base_pe != 16 * values_per_byte or dim not in {8, 16, 32}:
        raise ValueError("source MX GEMM needs a supported square mesh")
    if (scratchpad_bytes <= 0 or scratchpad_bytes % (4 * dim) or
            tm <= 0 or tn <= 0 or tk < 32 or tm % pe or tn % pe or tk % 32 or
            k % tk or m < tm or n < tn):
        raise ValueError("source MX GEMM has unsupported tile or scratchpad geometry")
    if profile is not None:
        if profile.get("schema") != "mx_gemmini.target_profile.v2":
            raise ValueError("selected MX target profile must be source-bound v2")
        if profile["geometry"]["mesh_rows"] != dim:
            raise ValueError("source MX GEMM needs a square mesh")
        if profile["resources"]["scratchpad_bytes"] != scratchpad_bytes:
            raise ValueError("selected MX profile scratchpad differs from planned target")
        legal = any(cell["activation_format"] == cell["weight_format"] == named and
                    cell["activation_projection"] == cell["weight_projection"] == projection
                    for cell in profile["legal_compute"])
        if not legal:
            raise ValueError(f"selected MX profile has no {named}/{projection} compute mode")
        if quant_output and named not in profile["candidate_output_modes"]:
            raise ValueError("selected MX profile lacks the source output format")
    if m % tm or n % tn:
        raise ValueError("MX output dimensions must divide into complete tiles")
    a = tm * tk // values_per_byte // dim
    b = tk * tn // values_per_byte // dim
    out_size = 1 if quant_output else 2
    out_m = tm // values_per_byte if quant_output else tm
    c = out_m * tn * out_size // dim
    if any(size <= 0 or size * dim > scratchpad_bytes for size in (a, b, c)):
        raise ValueError("MX operand or output exceeds scratchpad")
    rows = scratchpad_bytes // dim
    dest = _destination(rows, a, b, c)
    if dest == 0 and not acc_to_gmem:
        raise ValueError("C does not fit beside double-buffered A/B tiles")
    quarter = rows // 4
    waves = []
    for index in range(k // tk):
        odd = index & 1
        waves.append({
            "index": index,
            "k_start": index * tk,
            "k_stop": (index + 1) * tk,
            "a_spad_start": quarter if odd else 0,
            "b_spad_end": rows - quarter if odd else rows,
            "a_scale_buffer": odd,
            "b_scale_buffer": odd,
            "prefetch_next": index + 1 < k // tk,
            "accumulate": index > 0,
            "move_acc_to_spad": index == k // tk - 1 and not acc_to_gmem,
        })
    plan = {
        "schema": "mx_gemmini.source_gemm_plan.v1",
        "shape": list(shape), "tile": list(tile),
        "format": named, "projection": projection,
        "quant_output": quant_output, "acc_to_gmem": acc_to_gmem,
        "scratchpad_bytes": scratchpad_bytes,
        "scratchpad_rows": rows,
        "a_rows": a, "b_rows": b, "c_rows": c,
        "c_spad_dest": dest,
        "a_scale_bytes_per_wave": tm * tk // 32,
        "b_scale_bytes_per_wave": tn * tk // 32,
        "lut_once": datatype == "FP6",
        "move_out": "acc_dma" if acc_to_gmem else "spad_simt",
        "prologue_order": ["configure", "prefetch_0", "upload_scales_0",
                           "upload_lut_once" if datatype == "FP6" else "skip_lut",
                           "fence_smem", "wait_idle"],
        "per_wave_order": ["configure_scale_buffers", "prefetch_next_if_any",
                           "compute", "upload_next_scales", "fence_smem", "wait_idle"],
        "waves": waves,
    }
    if m != tm or n != tn:
        plan["output_tiles"] = [
            {"index": i * (n // tn) + j,
             "m_start": i * tm, "n_start": j * tn}
            for i in range(m // tm) for j in range(n // tn)]
    return plan


def plan_source_gemm(kernel: SourceGemm, *, scratchpad_bytes: int,
                     profile: dict | None = None) -> dict:
    """Apply the physical planner to one committed Radiance source driver."""
    plan = plan_mx_gemm(shape=kernel.shape, tile=kernel.tile,
                        datatype=kernel.datatype, quant_output=kernel.quant_output,
                        acc_to_gmem=kernel.acc_to_gmem,
                        scratchpad_bytes=scratchpad_bytes, profile=profile)
    return {**plan, "driver": kernel.driver.name,
            "driver_sha256": _sha(kernel.driver),
            "data_header": kernel.data_header.name,
            "data_header_sha256": _sha(kernel.data_header) if kernel.data_header_present else None,
            "data_header_present": kernel.data_header_present}
