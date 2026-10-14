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
STREAM_SHAPE = DenseAttentionShape(64, 2048, 64)
SMOL_SHAPES = (DenseAttentionShape(64, 128, 64),
               DenseAttentionShape(64, 192, 64))
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
                         block_k: int, input_format: str,
                         softmax_mode: str, causal: bool, arbitrary_mask: bool,
                         block_first: int, block_last: int,
                         v3: bool) -> None:
    header = _source_defines(source_header)
    fp4 = input_format == "fp4_e2m1"
    expected = {"ATTN_SQ": str(shape.sq), "ATTN_SK": str(shape.sk),
                "ATTN_D": str(shape.d), "ATTN_CAUSAL": "1" if causal else "0",
                "ATTN_HEADS": "1", "ATTN_SCALE_IN_Q": "0"}
    if fp4:
        expected["ATTN_FP4"] = "1"
    if arbitrary_mask:
        expected.update(ATTN_ARBITRARY_MASK="1", ATTN_VALID_Q="50")
    guards = {name for name in header if name.startswith("INCLUDE_ATTN_VPU_") or
              (arbitrary_mask and name == "SMOL_ATTENTION_HEAD0_H")}
    if (len(guards) != 1 or set(header) != set(expected) | guards or
            any(header.get(name) != value for name, value in expected.items())):
        raise ValueError("flash source header selects an unsupported policy")
    if source_wrapper != source_core:
        wrapper = _source_defines(source_wrapper)
        variable_blocks = block_first != block_k or block_last != block_k
        allowed = {"BK": str(block_k)}
        if variable_blocks:
            allowed.update(BK_FIRST=str(block_first), BK_LAST=str(block_last))
        if not v3:
            allowed["ATTN_V3"] = "0"
        suffix = ("_llama_causal" if causal else "_llama" if shape == STREAM_SHAPE
                  else "_mid_fp4" if fp4 else "_mid")
        optional = {"ATTN_HEADER": f'"attn_vpu{suffix}.h"',
                    "ATTN_EXPECT": (f'"attn_flash_llama_vb_expect.h"' if variable_blocks
                                    else f'"attn_flash{suffix}_expect.h"')}
        if softmax_mode == "expsum":
            optional["ATTN_EXPSUM"] = "1"
        if (wrapper.get("BK") != str(block_k) or
                any(wrapper.get(name) != value for name, value in allowed.items()) or
                (softmax_mode == "expsum" and wrapper.get("ATTN_EXPSUM") != "1") or
                set(wrapper) - set(allowed) - set(optional) or
                any(wrapper[name] != value for name, value in optional.items()
                    if name in wrapper) or
                source_wrapper.count('#include "attn_flash.c"') != 1):
            raise ValueError("flash source wrapper selects an unsupported policy")


def flash_buffer_abi(shape: DenseAttentionShape, block_k: int,
                     input_format: str = "fp8_e4m3", *, causal: bool = False,
                     arbitrary_mask: bool = False
                     ) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Runtime byte buffers for the selected flash source shape."""
    if (shape, block_k, input_format) not in (
            (SHAPE, BK, "fp8_e4m3"), (FA_SHAPE, 64, "fp8_e4m3"),
            (SHAPE, BK, "fp4_e2m1"), (STREAM_SHAPE, 256, "fp8_e4m3")):
        if not (arbitrary_mask and shape in SMOL_SHAPES and block_k == 64
                and input_format == "fp8_e4m3"):
            raise ValueError("flash compiler has no selected source shape")
    if arbitrary_mask and (shape not in SMOL_SHAPES or block_k != 64 or
                           input_format != "fp8_e4m3"):
        raise ValueError("arbitrary mask needs a selected SmolVLA FP8 shape")
    if causal and arbitrary_mask:
        raise ValueError("flash masks cannot be both causal and arbitrary")
    if causal and (shape, block_k, input_format) != (STREAM_SHAPE, 256, "fp8_e4m3"):
        raise ValueError("causal flash needs selected streamed source shape")
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
    if causal or arbitrary_mask:
        names += ("MASK_BF16",)
        lengths += (sq * (sk if arbitrary_mask else sq) * 2,)
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
                     input_format: str = "fp8_e4m3",
                     softmax_mode: str = "split", causal: bool = False,
                     arbitrary_mask: bool = False,
                     block_first: int | None = None, block_last: int | None = None,
                     v3: bool = True
                     ) -> tuple[str, dict]:
    """Render a source-bound, fenced QK/softmax/requant/PV program."""
    buffers, buffer_bytes = flash_buffer_abi(shape, block_k, input_format,
                                             causal=causal,
                                             arbitrary_mask=arbitrary_mask)
    block_first = block_k if block_first is None else block_first
    block_last = block_k if block_last is None else block_last
    variable_blocks = block_first != block_k or block_last != block_k
    if type(v3) is not bool:
        raise ValueError("flash V-buffer policy must be boolean")
    if (type(block_first) is not int or type(block_last) is not int or
            block_first <= 0 or block_last <= 0 or
            block_first > block_k or block_last > block_k or
            (shape.sk - block_first - block_last) % block_k or
            shape.sk < block_first + block_last or
            (variable_blocks and (shape, block_k, input_format, block_first,
                                  block_last, causal, softmax_mode) !=
             (STREAM_SHAPE, 256, "fp8_e4m3", 64, 192, False, "split")) or
            (not v3 and (shape, block_k, input_format, block_first,
                         block_last, causal, softmax_mode) !=
             (STREAM_SHAPE, 256, "fp8_e4m3", 256, 256, False, "split"))):
        raise ValueError("flash compiler has no selected key-block schedule")
    if softmax_mode not in {"split", "expsum"} or (
            softmax_mode == "expsum" and
            (shape, block_k, input_format) not in {
                (SHAPE, 32, "fp8_e4m3"), (SHAPE, 32, "fp4_e2m1")}):
        raise ValueError("flash compiler has no selected softmax mode")
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
                         shape, block_k, input_format, softmax_mode, causal,
                         arbitrary_mask,
                         block_first, block_last, v3)
    _audit_frontend_edges(frontend, shape, causal=causal or arbitrary_mask)
    if arbitrary_mask and (
            "gemmini_vpu_binary(VPU_ADD, S, S, SP_MASK + ROWS16(SQ, BOFF(j)), rows);"
            not in source_core or
            "MASK_BLOCK_BF16" not in source_header or
            softmax_mode != "split"):
        raise ValueError("arbitrary-mask source schedule differs")
    streamed = (shape, block_k, input_format) == (STREAM_SHAPE, 256, "fp8_e4m3")
    if streamed:
        if (profile.get("transport") != "rocket_rocc" or
                profile.get("geometry", {}).get("mesh_rows") != 16 or
                profile.get("geometry", {}).get("mesh_columns") != 16 or
                profile.get("resources", {}).get("scratchpad_banks") != 4 or
                profile.get("resources", {}).get("scratchpad_bytes") != 262144 or
                not profile.get("resources", {}).get("vpu") or
                not profile.get("resources", {}).get("spad_requant") or
                "#define K_BUFS   (ATTN_STREAM ? 2 : NB)" not in source_core or
                "#define V_BUFS   (ATTN_STREAM ? (ATTN_V3 ? 3 : 2) : NB)" not in source_core):
            raise ValueError("streamed flash needs Nicolas's four-bank Kx2/Vx3 source layout")
        if causal and (softmax_mode != "split" or
                       "#define SP_MASK  (S_BUF(NB) + ROWS16(SQ, BK) + ROWS16(SQ, D))" not in source_core or
                       "SP_MASK + r * (ATTN_CH / 8), ATTN_CH / 8" not in source_core):
            raise ValueError("causal flash source mask schedule differs")
    else:
        shape.validate(profile)
    if arbitrary_mask and profile.get("resources", {}).get("scratchpad_banks") != 4:
        raise ValueError("arbitrary mask needs four scratchpad banks")
    if (profile.get("source", {}).get("gemmini_revision") !=
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52"):
        raise ValueError("flash needs Nicolas's managed-scale MX RTL")
    SQ, SK, D, BK = shape.sq, shape.sk, shape.d, block_k
    fp4 = input_format == "fp4_e2m1"
    qt = 2 if fp4 else 1
    NB = 2 + (SK - block_first - block_last) // BK
    block_lengths = [block_first] + [BK] * (NB - 2) + [block_last]
    key_offsets = [0] + [block_first + (j - 1) * BK for j in range(1, NB)]
    rows8 = lambda m, n: m * n // (16 * qt)
    rows16 = lambda m, n: m * n * 2 // 16
    SP_Q, SP_V, SP_KT = 0, rows8(SQ, D), 4096
    k_buffers, v_buffers = (2, 3 if v3 else 2) if streamed else (NB, NB)
    SP_P = SP_KT + k_buffers * rows8(D, BK)
    ST = SP_V + v_buffers * rows8(BK, D)
    SP_O = 3 * 4096 + rows16(SQ, BK) + rows16(SQ, D)
    SP_MASK = (SP_O + rows16(SQ, D) if arbitrary_mask and NB & 1 else
               (2 + (NB & 1)) * 4096 + rows16(SQ, BK) + rows16(SQ, D))
    if (SP_V + v_buffers * rows8(BK, D) + 8 * SQ > 4096 or
            SP_P + 2 * rows8(SQ, BK) > 8192 or
            SP_O + rows16(SQ, D) > 16384 or
            (causal and SP_MASK + rows16(SQ, SQ) > 3 * 4096) or
            (arbitrary_mask and (
                SP_MASK + rows16(SQ, SK) > 4 * 4096 or
                SP_MASK // 4096 !=
                (SP_MASK + rows16(SQ, SK) - 1) // 4096))):
        raise ValueError("flash source bank-split residency differs")
    _score = lambda j: (2 + (j & 1)) * 4096
    _probability = lambda j: SP_P + (j & 1) * rows8(SQ, BK)
    _output_j = lambda j: SP_O if j == 0 else _score(j) + rows16(SQ, BK)
    _maximum = lambda j: ST + (j & 1) * SQ
    _alpha = lambda j: ST + (7 if j % 3 == 2 else 3 + j % 3) * SQ
    folded = D == 64
    variant = ("smolvla_arbitrary" if arbitrary_mask else
               "llama_causal_stream" if causal else
               "llama_vb_stream" if variable_blocks else
               "llama_wide_stream" if streamed and not v3 else
               "llama_stream" if streamed else
               "mid_fp4_expsum" if fp4 and softmax_mode == "expsum" else
               "mid_fp4" if fp4 else "mid_expsum" if softmax_mode == "expsum"
               else "mid" if folded else "fa")
    policy = {"schema": f"mx_gemmini.flash_{variant}_serial_policy.v1",
              "shape_sq_sk_d_bk": [SQ, SK, D, BK],
              "block_order": ["QK", "softmax", "requant", "PV", "update"],
              "fenced": True, "fold_q_scale_shift": 3 if folded else 0,
              "spad_q_v_kt_p_st_o": [SP_Q, SP_V, SP_KT, SP_P, ST, SP_O],
              "score_rows": [_score(j) for j in range(NB)],
              "probability_rows": [_probability(j) for j in range(NB)],
              "output_rows": [_output_j(j) for j in range(NB)]}
    if streamed:
        policy["kv_stream_buffers"] = [k_buffers, v_buffers]
    if not v3:
        policy["v3"] = False
    if variable_blocks:
        policy["block_first"] = block_first
        policy["block_last"] = block_last
        policy["block_lengths"] = block_lengths
        policy["key_offsets"] = key_offsets
    if causal:
        policy["causal"] = True
        policy["mask_spad_row"] = SP_MASK
        policy["mask_key_start"] = SK - SQ
    if arbitrary_mask:
        policy["arbitrary_mask"] = True
        policy["mask_spad_row"] = SP_MASK
        policy["valid_query_rows"] = 50
    if fp4:
        policy["input_format"] = input_format
    if softmax_mode == "expsum":
        policy["softmax_mode"] = softmax_mode
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
            immediate: int = 0, second_dst: int | None = None) -> str:
        return (f'    "mx_gemmini.vpu_execute"() {{site_id = "{site}", '
                f'kind = "{kind}", src1_row = {src1} : i32, '
                f'src2_row = {src2} : i32, dst_row = {dst} : i32, '
                f'rows = {rows} : i32, reduction_length = {rlen} : i32, '
                f'broadcast = {str(broadcast).lower()}, '
                f'immediate_bf16 = {immediate} : i32, '
                + (f'second_dst_row = {second_dst} : i32, '
                   if second_dst is not None else "") +
                f'{vector_attrs}}} : () -> ()')

    def gated(kind: str, j: int, half: int) -> str:
        if kind == "qk":
            return memory("gated_scale_dma", "KT_SCALES", f"{kind}{j}:scales",
                          f"row_bytes = {block_lengths[j]} : i32, rows = {D // 32} : i32, "
                          f"pitch_bytes = {SK} : i32, destination_byte = {half * 4096} : i32, "
                          f"selector = 1 : i32, source_offset_bytes = {key_offsets[j]} : i32")
        return memory("gated_scale_dma", "V_SCALES", f"{kind}{j}:scales",
                      f"row_bytes = {D} : i32, rows = {block_lengths[j] // 32} : i32, "
                      f"pitch_bytes = {D} : i32, destination_byte = {half * 4096} : i32, "
                      f"selector = 1 : i32, source_offset_bytes = {key_offsets[j] * D // 32} : i32")

    def managed(kind: str, j: int, half: int) -> str:
        qk = kind == "qk"
        m, n, k = SQ, block_lengths[j] if qk else D, D if qk else block_lengths[j]
        a = SP_Q if qk else _probability(j)
        b_end = (SP_KT + (j % k_buffers) * rows8(D, BK) + rows8(D, block_lengths[j])
                 if qk else SP_V + (j % v_buffers) * rows8(BK, D) +
                 rows8(block_lengths[j], D))
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
                    f"store_stride_bytes = {2 * block_first} : i32" +
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
    if causal or arbitrary_mask:
        lines.append(memory("dma_matrix", "MASK_BF16",
                            "operand:arbitrary_mask" if arbitrary_mask else
                            "operand:causal_mask",
                            f"matrix_rows = {SQ * (SK if arbitrary_mask else SQ) * 2 // 16} : i32, "
                            f"matrix_cols = 16 : i32, burst_cols = 16 : i32, "
                            f"spad_row = {SP_MASK} : i32"))
    def block_load(j: int) -> tuple[str, str]:
        return (memory("dma_matrix", buffers[4], f"operand:kt{j}",
                            f"matrix_rows = {D} : i32, matrix_cols = {block_lengths[j] // qt} : i32, "
                            f"burst_cols = {min(block_lengths[j] // qt, 64)} : i32, "
                            f"spad_row = {SP_KT + (j % k_buffers) * rows8(D, BK)} : i32, "
                            f"source_stride_bytes = {SK // qt} : i32, "
                            f"source_offset_bytes = {key_offsets[j] // qt} : i32"),
                memory("dma_matrix", buffers[5], f"operand:v{j}",
                            f"matrix_rows = {block_lengths[j]} : i32, matrix_cols = {D // qt} : i32, "
                            f"burst_cols = {min(D // qt, 64)} : i32, "
                            f"spad_row = {SP_V + (j % v_buffers) * rows8(BK, D)} : i32, "
                            f"source_stride_bytes = {D // qt} : i32, "
                            f"source_offset_bytes = {key_offsets[j] * D // qt} : i32"))
    if streamed:
        lines.append(wait("operand:q_fence"))
    else:
        for j in range(NB):
            lines.extend(block_load(j))
    if not fp4:
        lines.append(setup)
    n_mm = 0
    for j in range(NB):
        if streamed:
            lines.extend(block_load(j))
            lines.append(wait(f"operand:kv{j}:fence"))
        s = _score(j)
        m = _maximum(j)
        lines.extend((gated("qk", j, n_mm & 1), managed("qk", j, n_mm & 1),
                      wait(f"qk{j}:fence")))
        n_mm += 1
        score_rows, score_rlen = SQ * block_lengths[j] // 8, block_lengths[j] // 8
        if not folded:
            lines.append(vpu(f"softmax{j}:scale", "muls", s, 0, s,
                             score_rows, immediate=shape.scale_immediate_bf16))
        if causal and j == NB - 1:
            for r in range(SQ):
                row = s + r * (BK // 8) + (BK - SQ) // 8
                lines.append(vpu(f"softmax{j}:mask{r}", "add", row,
                                 SP_MASK + r * (SQ // 8), row, SQ // 8))
        if arbitrary_mask:
            lines.append(vpu(f"softmax{j}:mask", "add", s,
                             SP_MASK + rows16(SQ, key_offsets[j]), s,
                             score_rows))
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
        if softmax_mode == "expsum":
            lines.append(vpu(f"softmax{j}:expsum", "expsum", s, m, s,
                             score_rows, score_rlen, True,
                             second_dst=ST + (5 if j == 0 else 6) * SQ))
        else:
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
                             f"m = {SQ} : i32, n = {block_lengths[j]} : i32, "
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
                4 * NB + 3 + int(causal or arbitrary_mask), 1, 2 * NB,
                147 if causal else 12 * NB - 5 if arbitrary_mask else
                11 * NB - 5 if streamed else
                31 if softmax_mode == "expsum" else 39 if folded else 43, NB):
        raise ValueError(f"flash {variant} typed stage inventory differs: {report}")
    return mlir, binding


def lower_flash_mid(mlir_text: str, frontend: str, source_core: str,
                    source_wrapper: str, source_header: str, profile: dict, *,
                    shape: DenseAttentionShape = SHAPE, block_k: int = BK,
                    input_format: str = "fp8_e4m3",
                    softmax_mode: str = "split", causal: bool = False,
                    arbitrary_mask: bool = False,
                    block_first: int | None = None, block_last: int | None = None,
                    v3: bool = True
                    ) -> tuple[Command | Fence, ...]:
    """Require the exact source-bound serial graph before issuing commands."""
    expected, _ = render_flash_mid(frontend, source_core, source_wrapper,
                                   source_header, profile, shape=shape,
                                   block_k=block_k, input_format=input_format,
                                   softmax_mode=softmax_mode, causal=causal,
                                   arbitrary_mask=arbitrary_mask,
                                   block_first=block_first, block_last=block_last,
                                   v3=v3)
    if mlir_text != expected:
        raise ValueError("flash typed graph or source binding differs")
    buffers, buffer_bytes = flash_buffer_abi(shape, block_k, input_format,
                                             causal=causal,
                                             arbitrary_mask=arbitrary_mask)
    return lower_flash_commands(mlir_text, profile, buffers, buffer_bytes)
