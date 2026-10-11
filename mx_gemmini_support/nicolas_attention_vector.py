"""Bind Nicolas's dense attention softmax/requant seam to typed MX commands.

The surrounding QK and PV matrix commands are a separate lowering. This
module preserves the exact resident scratchpad layout and source VPU order
for the pinned 32x64x64 dense-attention kernel.
"""

from __future__ import annotations

import hashlib
import json

from .model2mlir_worklist import build_model2mlir_worklist
from .target_profile import profile_sha256
from .vector_lowering import lower_vector_commands
from .verify_profile_ir import _text_attr


SOURCE_CALLS = (
    "gemmini_vpu_scalar(VPU_MULS, SP_S, SP_S, sc, rows);",
    "gemmini_vpu_reduce(VPU_RMAX, SP_M, SP_S, rows, rlen);",
    "gemmini_vpu_bcast(VPU_SUB, SP_S, SP_S, SP_M, rows, rlen);",
    "gemmini_vpu_unary(VPU_EXP, SP_S, SP_S, rows);",
    "gemmini_vpu_reduce(VPU_RSUM, SP_L, SP_S, rows, rlen);",
    "gemmini_vpu_unary(VPU_RCP, SP_L, SP_L, SQ);",
    "gemmini_vpu_bcast(VPU_MUL, SP_S, SP_S, SP_L, rows, rlen);",
)
SOURCE_REQUANT = "gemmini_spad_requant(SP_P, SP_S, SQ, SK, 1, (uint64_t)p_scales, 1);"
SOURCE_LAYOUT = (
    "#define SP_Q      0",
    "#define SP_KT     (SP_Q + ROWS8(SQ, D))",
    "#define SP_KT_ARG (SP_KT + ROWS8(D, SK))",
    "#define SP_S      SP_KT_ARG",
    "#define SP_P      (SP_S + ROWS16(SQ, SK))",
    "#define SP_V      (SP_P + ROWS8(SQ, SK))",
    "#define SP_V_ARG  (SP_V + ROWS8(SK, D))",
    "#define SP_O      SP_V_ARG",
    "#define SP_M      0x3000",
    "#define SP_L      (SP_M + SQ)",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _audit_frontend_edges(frontend: str) -> None:
    """Require QK -> scale -> softmax -> PV, not just two matching shapes."""
    from xdsl.context import Context
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.math import Math
    from xdsl.dialects.scf import Scf
    from xdsl.dialects.tensor import Tensor
    from xdsl.parser import Parser

    context = Context(allow_unregistered=False)
    for dialect in (Builtin, Func, Tensor, Arith, Linalg, Math, Scf):
        context.load_dialect(dialect)
    module = Parser(context, frontend).parse_module()
    operations = list(module.walk())
    by_region = lambda region, name: [
        op for op in operations if op.name == name and
        _text_attr(op, "prov.region_id") == region]
    qk = by_region("matmul_0", "linalg.generic")
    scale = by_region("mul_0", "linalg.generic")
    pv = by_region("matmul_1", "linalg.generic")
    if (len(qk) != 1 or len(scale) != 1 or len(pv) != 1 or
            len(scale[0].operands) < 2 or len(pv[0].operands) < 1 or
            scale[0].operands[0] is not qk[0].results[0] or
            not any(op.name == "linalg.reduce" and op.operands[0] is
                    scale[0].results[0] for op in operations) or
            pv[0].operands[0].owner is None or
            _text_attr(pv[0].operands[0].owner, "prov.region_id") != "softmax_0" or
            _text_attr(pv[0].operands[0].owner, "prov.aten") !=
            "aten._softmax.default"):
        raise ValueError("dense attention QK-scale-softmax-PV SSA edges differ")
    scale_source = scale[0].operands[1].owner
    if (scale_source is None or scale_source.name != "tensor.splat" or
            not scale_source.operands or
            scale_source.operands[0].owner is None or
            scale_source.operands[0].owner.name != "arith.constant" or
            "1.250000e-01 : f32" not in str(scale_source.operands[0].owner)):
        raise ValueError("dense attention source scale differs from 1/8")


def render_dense_attention_vector(frontend: str, source: str,
                                  source_header: str, profile: dict,
                                  contract: bytes) -> tuple[str, dict]:
    """Emit the source-bound seven VPU ops and resident E4M3 requant op."""
    worklist = build_model2mlir_worklist(frontend, profile)
    sites = worklist["rank3_batch_matmuls"]
    if ([site["shape_bmnk"] for site in sites] != [[1, 32, 64, 64]] * 2 or
            [site["region_id"] for site in sites] != ["matmul_0", "matmul_1"] or
            frontend.count('prov.aten = "aten._softmax.default"') < 2 or
            'prov.region_id = "softmax_0"' not in frontend or
            any(token in frontend for token in ("torch.operator", "func.call"))):
        raise ValueError("dense attention frontend differs from QK-scale-softmax-PV capture")
    _audit_frontend_edges(frontend)
    if (any(source.count(line) != 1 for line in SOURCE_CALLS + (SOURCE_REQUANT,)) or
            any(source.count(line) != 1 for line in SOURCE_LAYOUT) or
            source.count("\n".join("  " + line for line in SOURCE_CALLS)) != 1 or
            any(source_header.count(line) != 1 for line in
                ("#define ATTN_SQ 32", "#define ATTN_SK 64",
                 "#define ATTN_D  64", "#define ATTN_SCALE_IN_Q 0"))):
        raise ValueError("dense attention source vector schedule or data shape differs")
    rows = profile["resources"]["scratchpad_bytes"] // 16
    if (profile["geometry"]["mesh_columns"] != 16 or
            profile["geometry"]["mesh_rows"] != 16 or
            not profile["resources"].get("vpu") or
            not profile["resources"].get("spad_requant") or
            rows < 0x3020 + 32):
        raise ValueError("dense attention needs the DIM16 MX+VPU resident profile")

    sq, sk, d = 32, 64, 64
    score = (sq * d + d * sk) // 16
    probability = score + sq * sk * 2 // 16
    v_base = probability + sq * sk // 16
    output = v_base + sk * d // 16
    if (score, probability, v_base, output) != (384, 640, 768, 1024):
        raise ValueError("dense attention scratchpad plan differs")
    vector_rows, row_length = sq * sk // 8, sk // 8
    steps = (
        ("muls", score, 0, score, vector_rows, 1, False, 0x3e00),
        ("rmax", score, 0, 0x3000, vector_rows, row_length, False, 0),
        ("sub", score, 0x3000, score, vector_rows, row_length, True, 0),
        ("exp", score, 0, score, vector_rows, 1, False, 0),
        ("rsum", score, 0, 0x3020, vector_rows, row_length, False, 0),
        ("rcp", 0x3020, 0, 0x3020, sq, 1, False, 0),
        ("mul", score, 0x3020, score, vector_rows, row_length, True, 0),
    )
    plan = {"shape_sq_sk_d": [sq, sk, d], "score_row": score,
            "probability_row": probability, "v_row": v_base,
            "output_row": output, "max_row": 0x3000,
            "sum_row": 0x3020, "steps": [list(step) for step in steps],
            "rounding": "BF16 after each VPU operation"}
    policy = _sha(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode())
    binding = {"schema": "mx_gemmini.nicolas_dense_attention_vector_binding.v1",
               "source_sha256": _sha(source.encode()),
               "source_header_sha256": _sha(source_header.encode()),
               "frontend_mlir_sha256": _sha(frontend.encode()),
               "profile_sha256": profile_sha256(profile),
               "contract_sha256": _sha(contract), "policy_sha256": policy,
               "plan": plan}
    digest = _sha(json.dumps(binding, sort_keys=True).encode())
    common = (f'profile_sha256 = "{binding["profile_sha256"]}", '
              f'contract_sha256 = "{binding["contract_sha256"]}", '
              f'policy_sha256 = "{policy}", manifest_sha256 = "{digest}"')
    lines = ["builtin.module attributes {",
             f'  mx.contract_sha256 = "{binding["contract_sha256"]}",',
             f'  mx.policy_sha256 = "{policy}",',
             f'  prov.quantization_manifest_sha256 = "{digest}",',
             f'  mx.source_mlir_sha256 = "{binding["frontend_mlir_sha256"]}",',
             f'  mx.profile_sha256 = "{binding["profile_sha256"]}"',
             "} {", "  func.func @attention_vector() {"]
    for index, (kind, src1, src2, dst, count, rlen, broadcast, immediate) in enumerate(steps):
        lines.append(
            f'    "mx_gemmini.vpu_execute"() {{site_id = "softmax:{index}", '
            f'kind = "{kind}", src1_row = {src1} : i32, src2_row = {src2} : i32, '
            f'dst_row = {dst} : i32, rows = {count} : i32, '
            f'reduction_length = {rlen} : i32, broadcast = {str(broadcast).lower()}, '
            f'immediate_bf16 = {immediate} : i32, {common}}} : () -> ()')
    lines.append(
        f'    "mx_gemmini.spad_requant"() {{site_id = "probability:requant", '
        f'source_row = {score} : i32, destination_row = {probability} : i32, '
        f'm = {sq} : i32, n = {sk} : i32, output_format = "fp8_e4m3", '
        f'tiled = true, resident = true, scale_dram_address = 0 : i64, '
        f'scale_buffer = "p_scales", {common}}} : () -> ()')
    lines.extend(("    func.return", "  }", "}", ""))
    mlir = "\n".join(lines)
    commands = lower_vector_commands(mlir, profile)
    if [command.funct for command in commands] != [33] * 7 + [34]:
        raise ValueError("dense attention vector commands differ")
    binding["manifest_sha256"] = digest
    binding["bound_mlir_sha256"] = _sha(mlir.encode())
    return mlir, binding
