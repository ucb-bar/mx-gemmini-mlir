"""Serial functional schedule for Nicolas's four-block FP8 flash attention."""

from __future__ import annotations

import hashlib
import json

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
SQ, SK, D = SHAPE.sq, SHAPE.sk, SHAPE.d
NB = SK // BK
SP_Q, SP_V, SP_KT, SP_P, ST = 0, 256, 4096, 4608, 768
SP_O = 13056


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
                     source_header: str, profile: dict) -> tuple[str, dict]:
    """Render a source-bound, fenced QK/softmax/requant/PV program."""
    if (DenseAttentionShape.from_header(source_header) != SHAPE or
            '#define BK 32' not in source_wrapper or
            '#include "attn_flash.c"' not in source_wrapper or
            'static uint64_t run_serial(void)' not in source_core or
            'gemmini_mx_load_scales_2d_gated' not in source_core or
            'gemmini_mxquant_config_mvout_managed' not in source_core):
        raise ValueError("pinned flash mid source or geometry differs")
    _audit_frontend_edges(frontend, SHAPE)
    SHAPE.validate(profile)
    if (profile.get("source", {}).get("gemmini_revision") !=
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52"):
        raise ValueError("flash mid needs Nicolas's managed-scale MX RTL")
    policy = {"schema": "mx_gemmini.flash_mid_serial_policy.v1",
              "shape_sq_sk_d_bk": [SQ, SK, D, BK],
              "block_order": ["QK", "softmax", "requant", "PV", "update"],
              "fenced": True, "fold_q_scale_shift": 3,
              "spad_q_v_kt_p_st_o": [SP_Q, SP_V, SP_KT, SP_P, ST, SP_O],
              "score_rows": [_score(j) for j in range(NB)],
              "probability_rows": [_probability(j) for j in range(NB)],
              "output_rows": [_output_j(j) for j in range(NB)]}
    binding = {"schema": "mx_gemmini.flash_mid_serial_binding.v1",
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
                     for name, length in zip(BUFFERS, BUFFER_BYTES))

    def memory(op: str, buffer: str, site: str, attrs: str) -> str:
        length = BUFFER_BYTES[BUFFERS.index(buffer)]
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
            rows: int, rlen: int = 1, broadcast: bool = False) -> str:
        return (f'    "mx_gemmini.vpu_execute"() {{site_id = "{site}", '
                f'kind = "{kind}", src1_row = {src1} : i32, '
                f'src2_row = {src2} : i32, dst_row = {dst} : i32, '
                f'rows = {rows} : i32, reduction_length = {rlen} : i32, '
                f'broadcast = {str(broadcast).lower()}, '
                f'immediate_bf16 = 0 : i32, {vector_attrs}}} : () -> ()')

    def gated(kind: str, j: int, half: int) -> str:
        if kind == "qk":
            return memory("gated_scale_dma", "KT_SCALES", f"{kind}{j}:scales",
                          f"row_bytes = {BK} : i32, rows = {D // 32} : i32, "
                          f"pitch_bytes = {SK} : i32, destination_byte = {half * 4096} : i32, "
                          f"selector = 1 : i32, source_offset_bytes = {j * BK} : i32")
        return memory("gated_scale_dma", "V_SCALES", f"{kind}{j}:scales",
                      f"row_bytes = {D} : i32, rows = {BK // 32} : i32, "
                      f"pitch_bytes = {D} : i32, destination_byte = {half * 4096} : i32, "
                      f"selector = 1 : i32, source_offset_bytes = {j * D} : i32")

    def managed(kind: str, j: int, half: int) -> str:
        qk = kind == "qk"
        m, n, k = SQ, BK if qk else D, D if qk else BK
        a = SP_Q if qk else _probability(j)
        b_end = (SP_KT + (j + 1) * D * BK // 16 if qk else
                 SP_V + (j + 1) * BK * D // 16)
        c = _score(j) if qk else _output_j(j)
        return control("managed_matmul_spad", f"{kind}{j}:matmul",
                       f"m = {m} : i32, n = {n} : i32, k = {k} : i32, "
                       f"activation_row = {a} : i32, weight_end_row = {b_end} : i32, "
                       f"output_row = {c} : i32, weight_half = {half} : i32, "
                       f"activation_half = {1 if qk else 0} : i32, "
                       'scale_sink_buffer = "scale_sink"')

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
        memory("scale_dma", "Q_SCALES_FOLDED", "scales:q_folded",
               "row_bytes = 64 : i32, rows = 2 : i32, pitch_bytes = 64 : i32, "
               "destination_byte = 4096 : i32, selector = 0 : i32"),
        wait("scales:q_fence"),
        memory("dma_matrix", "Q_IN", "operand:q",
               "matrix_rows = 64 : i32, matrix_cols = 64 : i32, "
               "burst_cols = 64 : i32, spad_row = 0 : i32"),
    ]
    for j in range(NB):
        lines.append(memory("dma_matrix", "KT_IN", f"operand:kt{j}",
                            f"matrix_rows = {D} : i32, matrix_cols = {BK} : i32, "
                            f"burst_cols = 32 : i32, spad_row = {SP_KT + j * 128} : i32, "
                            f"source_stride_bytes = {SK} : i32, "
                            f"source_offset_bytes = {j * BK} : i32"))
        lines.append(memory("dma_matrix", "V_IN", f"operand:v{j}",
                            f"matrix_rows = {BK} : i32, matrix_cols = {D} : i32, "
                            f"burst_cols = 64 : i32, spad_row = {SP_V + j * 128} : i32, "
                            f"source_stride_bytes = {D} : i32, "
                            f"source_offset_bytes = {j * BK * D} : i32"))
    lines.append(control("flash_setup", "flash:setup", "store_stride_bytes = 64 : i32"))
    n_mm = 0
    for j in range(NB):
        s = _score(j)
        m = _maximum(j)
        lines.extend((gated("qk", j, n_mm & 1), managed("qk", j, n_mm & 1),
                      wait(f"qk{j}:fence")))
        n_mm += 1
        if j == 0:
            lines.append(vpu("softmax0:rmax", "rmax", s, 0, m, 256, 4))
        else:
            old = _maximum(j - 1)
            alpha = _alpha(j)
            lines.extend((vpu(f"softmax{j}:rmax", "rmax", s, 0, ST + 2 * SQ, 256, 4),
                          vpu(f"softmax{j}:max", "max", old, ST + 2 * SQ, m, SQ),
                          vpu(f"softmax{j}:alpha_sub", "sub", old, m, alpha, SQ),
                          vpu(f"softmax{j}:alpha_exp", "exp", alpha, 0, alpha, SQ)))
        lines.extend((vpu(f"softmax{j}:sub", "sub", s, m, s, 256, 4, True),
                      vpu(f"softmax{j}:exp", "exp", s, 0, s, 256),
                      vpu(f"softmax{j}:rsum", "rsum", s, 0,
                          ST + (5 if j == 0 else 6) * SQ, 256, 4)))
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
                             'output_format = "fp8_e4m3", tiled = true, resident = true, '
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
            report["vpu_commands"], report["spad_requants"]) != (19, 1, 8, 39, 4):
        raise ValueError(f"flash mid typed stage inventory differs: {report}")
    return mlir, binding


def lower_flash_mid(mlir_text: str, frontend: str, source_core: str,
                    source_wrapper: str, source_header: str, profile: dict
                    ) -> tuple[Command | Fence, ...]:
    """Require the exact source-bound serial graph before issuing commands."""
    expected, _ = render_flash_mid(frontend, source_core, source_wrapper,
                                   source_header, profile)
    if mlir_text != expected:
        raise ValueError("flash mid typed graph or source binding differs")
    return lower_flash_commands(mlir_text, profile, BUFFERS, BUFFER_BYTES)
