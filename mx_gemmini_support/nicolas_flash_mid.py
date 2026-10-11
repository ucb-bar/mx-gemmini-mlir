"""Serial functional schedule for Nicolas's four-block FP8 flash attention."""

from __future__ import annotations

import hashlib
import json
import re

from .command_ir import Command, Fence
from .nicolas_attention_shape import DenseAttentionShape
from .nicolas_attention_vector import _audit_frontend_edges
from .nicolas_flash_commands import SCHEMA, lower_flash_commands
from .target_profile import profile_sha256
from .verify_profile_ir import verify_ir


BUFFERS = ("Q_SCALES_FOLDED", "KT_SCALES", "V_SCALES", "Q_IN", "KT_IN", "V_IN",
           "scale_sink", "p_scales_0", "p_scales_1", "O_HW")
BUFFER_BYTES = (128, 256, 256, 4096, 8192, 8192, 2048, 64, 64, 8192)
SHAPE = DenseAttentionShape(64, 128, 64)
BK = 32
FA_SHAPE = DenseAttentionShape(64, 256, 128)
SQ, SK, D = SHAPE.sq, SHAPE.sk, SHAPE.d
NB = SK // BK
SP_Q, SP_V, SP_KT, SP_P, ST = 0, 256, 4096, 4608, 768
SP_O = 13056


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_defines(source: str) -> dict[str, str]:
    """Read preprocessor constants that select the source kernel's semantics."""
    result = {}
    for match in re.finditer(r"^[ \t]*#define[ \t]+([A-Za-z_]\w*)"
                             r"(?:[ \t]+([^\r\n]*))?$", source, re.MULTILINE):
        name = match.group(1)
        if name in result:
            raise ValueError(f"flash source policy repeats {name}")
        result[name] = (match.group(2) or "").split("//", 1)[0].split(
            "/*", 1)[0].strip()
    return result


def _check_source_policy(source_wrapper: str, source_core: str,
                         source_header: str, shape: DenseAttentionShape,
                         block_k: int, input_format: str) -> None:
    header = _source_defines(source_header)
    fp4 = input_format == "fp4_e2m1"
    expected = {"ATTN_SQ": str(shape.sq), "ATTN_SK": str(shape.sk),
                "ATTN_D": str(shape.d), "ATTN_CAUSAL": "0",
                "ATTN_HEADS": "1", "ATTN_SCALE_IN_Q": "0"}
    if fp4:
        expected["ATTN_FP4"] = "1"
    guards = {name for name in header if name.startswith("INCLUDE_ATTN_VPU_")}
    if (len(guards) != 1 or set(header) != set(expected) | guards or
            any(header.get(name) != value for name, value in expected.items())):
        raise ValueError("flash source header selects an unsupported policy")
    if source_wrapper != source_core:
        wrapper = _source_defines(source_wrapper)
        allowed = {"BK": str(block_k)}
        suffix = "_mid_fp4" if fp4 else "_mid"
        optional = {"ATTN_HEADER": f'"attn_vpu{suffix}.h"',
                    "ATTN_EXPECT": f'"attn_flash{suffix}_expect.h"'}
        if (wrapper.get("BK") != str(block_k) or
                set(wrapper) - set(allowed) - set(optional) or
                any(wrapper[name] != value for name, value in optional.items()
                    if name in wrapper) or
                source_wrapper.count('#include "attn_flash.c"') != 1):
            raise ValueError("flash source wrapper selects an unsupported policy")


def flash_buffer_abi(shape: DenseAttentionShape, block_k: int,
                     input_format: str = "fp8_e4m3"
                     ) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Runtime byte buffers for the selected resident flash source shape."""
    if (shape, block_k, input_format) not in (
            (SHAPE, BK, "fp8_e4m3"), (FA_SHAPE, 64, "fp8_e4m3"),
            (SHAPE, BK, "fp4_e2m1")):
        raise ValueError("flash compiler has no selected source shape")
    sq, sk, d = shape.sq, shape.sk, shape.d
    fp4 = input_format == "fp4_e2m1"
    qt = 2 if fp4 else 1
    names = (("Q_SCALES_FOLDED" if d == 64 else "Q_SCALES"),
             "KT_SCALES", "V_SCALES", "Q4_IN" if fp4 else "Q_IN",
             "KT4_IN" if fp4 else "KT_IN", "V4_IN" if fp4 else "V_IN",
             "scale_sink", "p_scales_0", "p_scales_1", "O_HW")
    lengths = (sq * d // 32, d * sk // 32, sk * d // 32,
               sq * d // qt, d * sk // qt, sk * d // qt, 2048,
               sq * block_k // 32, sq * block_k // 32, sq * d * 2)
    return names, lengths


def _score(j: int) -> int:
    return (2 + (j & 1)) * 4096


def _probability(j: int) -> int:
    return SP_P + (j & 1) * 128


def _output_j(j: int) -> int:
    return SP_O if j == 0 else _score(j) + 256


def _maximum(j: int) -> int:
    return ST + (j & 1) * SQ


def _alpha(j: int) -> int:
    return ST + (7 if j % 3 == 2 else 3 + j % 3) * SQ


def render_flash_mid(frontend: str, source_core: str, source_wrapper: str,
                     source_header: str, profile: dict, *,
                     shape: DenseAttentionShape = SHAPE, block_k: int = BK,
                     input_format: str = "fp8_e4m3"
                     ) -> tuple[str, dict]:
    """Render a source-bound, fenced QK/softmax/requant/PV program."""
    buffers, buffer_bytes = flash_buffer_abi(shape, block_k, input_format)
    if (DenseAttentionShape.from_header(source_header) != shape or
            (source_wrapper != source_core and
             (f'#define BK {block_k}' not in source_wrapper or
              '#include "attn_flash.c"' not in source_wrapper)) or
            (source_wrapper == source_core and block_k != 64) or
            'static uint64_t run_serial(void)' not in source_core or
            'gemmini_mx_load_scales_2d_gated' not in source_core or
            'gemmini_mxquant_config_mvout_managed' not in source_core):
        raise ValueError("pinned flash source or geometry differs")
    _check_source_policy(source_wrapper, source_core, source_header,
                         shape, block_k, input_format)
    _audit_frontend_edges(frontend, shape)
    shape.validate(profile)
    if (profile.get("source", {}).get("gemmini_revision") !=
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52"):
        raise ValueError("flash needs Nicolas's managed-scale MX RTL")
    SQ, SK, D, BK = shape.sq, shape.sk, shape.d, block_k
    fp4 = input_format == "fp4_e2m1"
    qt = 2 if fp4 else 1
    NB = SK // BK
    rows8 = lambda m, n: m * n // (16 * qt)
    rows16 = lambda m, n: m * n * 2 // 16
    SP_Q, SP_V, SP_KT = 0, rows8(SQ, D), 4096
    SP_P = SP_KT + NB * rows8(D, BK)
    ST = SP_V + NB * rows8(BK, D)
    SP_O = 3 * 4096 + rows16(SQ, BK) + rows16(SQ, D)
    if (SP_V + NB * rows8(BK, D) + 8 * SQ > 4096 or
            SP_P + 2 * rows8(SQ, BK) > 8192 or
            SP_O + rows16(SQ, D) > 16384):
        raise ValueError("flash source bank-split residency differs")
    _score = lambda j: (2 + (j & 1)) * 4096
    _probability = lambda j: SP_P + (j & 1) * rows8(SQ, BK)
    _output_j = lambda j: SP_O if j == 0 else _score(j) + rows16(SQ, BK)
    _maximum = lambda j: ST + (j & 1) * SQ
    _alpha = lambda j: ST + (7 if j % 3 == 2 else 3 + j % 3) * SQ
    folded = D == 64
    variant = "mid_fp4" if fp4 else "mid" if folded else "fa"
    policy = {"schema": f"mx_gemmini.flash_{variant}_serial_policy.v1",
              "shape_sq_sk_d_bk": [SQ, SK, D, BK],
              "block_order": ["QK", "softmax", "requant", "PV", "update"],
              "fenced": True, "fold_q_scale_shift": 3 if folded else 0,
              "spad_q_v_kt_p_st_o": [SP_Q, SP_V, SP_KT, SP_P, ST, SP_O],
              "score_rows": [_score(j) for j in range(NB)],
              "probability_rows": [_probability(j) for j in range(NB)],
              "output_rows": [_output_j(j) for j in range(NB)]}
    if fp4:
        policy["input_format"] = input_format
    binding = {"schema": f"mx_gemmini.flash_{variant}_serial_binding.v1",
               "profile_sha256": profile_sha256(profile),
               "source_sha256": _sha(source_core.encode()),
               "wrapper_sha256": _sha(source_wrapper.encode()),
               "header_sha256": _sha(source_header.encode()),
               "frontend_sha256": _sha(frontend.encode()),
               "policy": policy}
    binding["policy_sha256"] = _sha(json.dumps(policy, sort_keys=True,
                                                separators=(",", ":")).encode())
    binding["manifest_sha256"] = _sha(json.dumps(binding, sort_keys=True,
                                                  separators=(",", ":")).encode())
    digest = binding["profile_sha256"]
    source = binding["source_sha256"]
    contract = binding["frontend_sha256"]
    policy_digest = binding["policy_sha256"]
    manifest = binding["manifest_sha256"]
    source_attrs = f'profile_sha256 = "{digest}", source_sha256 = "{source}"'
    vector_attrs = (f'profile_sha256 = "{digest}", contract_sha256 = "{contract}", '
                    f'policy_sha256 = "{policy_digest}", manifest_sha256 = "{manifest}"')
    args = ", ".join(f"%{name}: memref<{length}xi8>"
                     for name, length in zip(buffers, buffer_bytes))

    def memory(op: str, buffer: str, site: str, attrs: str) -> str:
        length = buffer_bytes[buffers.index(buffer)]
        return (f'    "mx_gemmini.{op}"(%{buffer}) {{site_id = "{site}", '
                f'{attrs}, {source_attrs}}} : (memref<{length}xi8>) -> ()')

    def control(op: str, site: str, attrs: str) -> str:
        common = vector_attrs if op == "spad_requant" else source_attrs
        return (f'    "mx_gemmini.{op}"() {{site_id = "{site}", '
                f'{attrs}, {common}}} : () -> ()')

    def wait(site: str) -> str:
        return (f'    "mx_gemmini.wait"() {{site_id = "{site}", '
                f'{vector_attrs}}} : () -> ()')

    def vpu(site: str, kind: str, src1: int, src2: int, dst: int,
            rows: int, rlen: int = 1, broadcast: bool = False,
            immediate: int = 0) -> str:
        return (f'    "mx_gemmini.vpu_execute"() {{site_id = "{site}", '
                f'kind = "{kind}", src1_row = {src1} : i32, '
                f'src2_row = {src2} : i32, dst_row = {dst} : i32, '
                f'rows = {rows} : i32, reduction_length = {rlen} : i32, '
                f'broadcast = {str(broadcast).lower()}, '
                f'immediate_bf16 = {immediate} : i32, {vector_attrs}}} : () -> ()')

    def gated(kind: str, j: int, half: int) -> str:
        if kind == "qk":
            return memory("gated_scale_dma", "KT_SCALES", f"{kind}{j}:scales",
                          f"row_bytes = {BK} : i32, rows = {D // 32} : i32, "
                          f"pitch_bytes = {SK} : i32, destination_byte = {half * 4096} : i32, "
                          f"selector = 1 : i32, source_offset_bytes = {j * BK} : i32")
        return memory("gated_scale_dma", "V_SCALES", f"{kind}{j}:scales",
                      f"row_bytes = {D} : i32, rows = {BK // 32} : i32, "
                      f"pitch_bytes = {D} : i32, destination_byte = {half * 4096} : i32, "
                      f"selector = 1 : i32, source_offset_bytes = {j * BK * D // 32} : i32")

    def managed(kind: str, j: int, half: int) -> str:
        qk = kind == "qk"
        m, n, k = SQ, BK if qk else D, D if qk else BK
        a = SP_Q if qk else _probability(j)
        b_end = (SP_KT + (j + 1) * D * BK // (16 * qt) if qk else
                 SP_V + (j + 1) * BK * D // (16 * qt))
        c = _score(j) if qk else _output_j(j)
        return control("managed_matmul_spad", f"{kind}{j}:matmul",
                       f"m = {m} : i32, n = {n} : i32, k = {k} : i32, "
                       f"activation_row = {a} : i32, weight_end_row = {b_end} : i32, "
                       f"output_row = {c} : i32, weight_half = {half} : i32, "
                       f"activation_half = {1 if qk else 0} : i32, "
                       'scale_sink_buffer = "scale_sink"' +
                       (f', input_format = "{input_format}"' if fp4 else ""))

    lines = [
        "builtin.module attributes {",
        f'  mx.attention_program_schema = "{SCHEMA}",',
        f'  mx.profile_sha256 = "{digest}", mx.source_sha256 = "{source}",',
        f'  mx.source_mlir_sha256 = "{contract}",',
        f'  mx.source_header_sha256 = "{binding["header_sha256"]}",',
        f'  mx.contract_sha256 = "{contract}",',
        f'  mx.policy_sha256 = "{policy_digest}",',
        f'  prov.quantization_manifest_sha256 = "{manifest}"',
        "} {", f"  func.func @mx_issue({args}) {{",
    ]
    setup = control("flash_setup", "flash:setup",
                    f"store_stride_bytes = {2 * BK} : i32" +
                    (f', input_format = "{input_format}"' if fp4 else ""))
    if fp4:
        # The quad layout must be selected before the first activation scale DMA.
        lines.append(setup)
    lines.extend([
        memory("scale_dma", buffers[0],
               "scales:q_folded" if folded else "scales:q",
               f"row_bytes = {SQ} : i32, rows = {D // 32} : i32, pitch_bytes = {SQ} : i32, "
               "destination_byte = 4096 : i32, selector = 0 : i32"),
        wait("scales:q_fence"),
        memory("dma_matrix", buffers[3], "operand:q",
               f"matrix_rows = {SQ // qt} : i32, matrix_cols = {D} : i32, "
               "burst_cols = 64 : i32, spad_row = 0 : i32"),
    ])
    for j in range(NB):
        lines.append(memory("dma_matrix", buffers[4], f"operand:kt{j}",
                            f"matrix_rows = {D} : i32, matrix_cols = {BK // qt} : i32, "
                            f"burst_cols = {min(BK // qt, 64)} : i32, "
                            f"spad_row = {SP_KT + j * rows8(D, BK)} : i32, "
                            f"source_stride_bytes = {SK // qt} : i32, "
                            f"source_offset_bytes = {j * BK // qt} : i32"))
        lines.append(memory("dma_matrix", buffers[5], f"operand:v{j}",
                            f"matrix_rows = {BK} : i32, matrix_cols = {D // qt} : i32, "
                            f"burst_cols = {min(D // qt, 64)} : i32, "
                            f"spad_row = {SP_V + j * rows8(BK, D)} : i32, "
                            f"source_stride_bytes = {D // qt} : i32, "
                            f"source_offset_bytes = {j * BK * D // qt} : i32"))
    if not fp4:
        lines.append(setup)
    n_mm = 0
    for j in range(NB):
        s = _score(j)
        m = _maximum(j)
        lines.extend((gated("qk", j, n_mm & 1), managed("qk", j, n_mm & 1),
                      wait(f"qk{j}:fence")))
        n_mm += 1
        score_rows, score_rlen = SQ * BK // 8, BK // 8
        if not folded:
            lines.append(vpu(f"softmax{j}:scale", "muls", s, 0, s,
                             score_rows, immediate=shape.scale_immediate_bf16))
        if j == 0:
            lines.append(vpu("softmax0:rmax", "rmax", s, 0, m,
                             score_rows, score_rlen))
        else:
            old = _maximum(j - 1)
            alpha = _alpha(j)
            lines.extend((vpu(f"softmax{j}:rmax", "rmax", s, 0, ST + 2 * SQ,
                              score_rows, score_rlen),
                          vpu(f"softmax{j}:max", "max", old, ST + 2 * SQ, m, SQ),
                          vpu(f"softmax{j}:alpha_sub", "sub", old, m, alpha, SQ),
                          vpu(f"softmax{j}:alpha_exp", "exp", alpha, 0, alpha, SQ)))
        lines.extend((vpu(f"softmax{j}:sub", "sub", s, m, s,
                          score_rows, score_rlen, True),
                      vpu(f"softmax{j}:exp", "exp", s, 0, s, score_rows),
                      vpu(f"softmax{j}:rsum", "rsum", s, 0,
                          ST + (5 if j == 0 else 6) * SQ,
                          score_rows, score_rlen)))
        if j > 0:
            lines.extend((vpu(f"softmax{j}:sum_mul", "mul", ST + 5 * SQ,
                              _alpha(j), ST + 5 * SQ, SQ),
                          vpu(f"softmax{j}:sum_add", "add", ST + 5 * SQ,
                              ST + 6 * SQ, ST + 5 * SQ, SQ)))
        lines.append(wait(f"softmax{j}:fence"))
        lines.append(control("spad_requant", f"probability{j}:requant",
                             f"source_row = {s} : i32, "
                             f"destination_row = {_probability(j)} : i32, "
                             f"m = {SQ} : i32, n = {BK} : i32, "
                             f'output_format = "{input_format}", tiled = true, resident = true, '
                             f'scale_dram_address = 0 : i64, scale_buffer = "p_scales_{j & 1}"'))
        lines.append(wait(f"probability{j}:fence"))
        lines.extend((gated("pv", j, n_mm & 1), managed("pv", j, n_mm & 1),
                      wait(f"pv{j}:fence")))
        n_mm += 1
        if j > 0:
            lines.extend((vpu(f"output{j}:mul", "mul", SP_O, _alpha(j),
                              SP_O, SQ * D // 8, D // 8, True),
                          vpu(f"output{j}:add", "add", SP_O, _output_j(j),
                              SP_O, SQ * D // 8),
                          wait(f"output{j}:fence")))
    lines.extend((vpu("final:rcp", "rcp", ST + 5 * SQ, 0, ST + 5 * SQ, SQ),
                  vpu("final:normalize", "mul", SP_O, ST + 5 * SQ,
                      SP_O, SQ * D // 8, D // 8, True),
                  wait("final:fence"),
                  memory("spad_mvout_linear", "O_HW", "output:readout",
                         f"total_bytes = {SQ * D * 2} : i32, "
                         f"tile_cols = 16 : i32, spad_row = {SP_O} : i32"),
                  "    func.return", "  }", "}", ""))
    mlir = "\n".join(lines)
    report = verify_ir(mlir, profile)
    if (report["memory_phases"], report["flash_setups"], report["spad_matmuls"],
            report["vpu_commands"], report["spad_requants"]) != (
                19, 1, 8, 39 if folded else 43, 4):
        raise ValueError(f"flash {variant} typed stage inventory differs: {report}")
    return mlir, binding


def lower_flash_mid(mlir_text: str, frontend: str, source_core: str,
                    source_wrapper: str, source_header: str, profile: dict, *,
                    shape: DenseAttentionShape = SHAPE, block_k: int = BK,
                    input_format: str = "fp8_e4m3"
                    ) -> tuple[Command | Fence, ...]:
    """Require the exact source-bound serial graph before issuing commands."""
    expected, _ = render_flash_mid(frontend, source_core, source_wrapper,
                                   source_header, profile, shape=shape,
                                   block_k=block_k, input_format=input_format)
    if mlir_text != expected:
        raise ValueError("flash typed graph or source binding differs")
    buffers, buffer_bytes = flash_buffer_abi(shape, block_k, input_format)
    return lower_flash_commands(mlir_text, profile, buffers, buffer_bytes)
