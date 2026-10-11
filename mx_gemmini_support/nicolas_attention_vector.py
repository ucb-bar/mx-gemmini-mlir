"""Bind Nicolas's dense attention softmax/requant seam to typed MX commands.

The surrounding QK and PV matrix commands are a separate lowering. This
module preserves the exact resident scratchpad layout and source VPU order
for the pinned 32x64x64 dense-attention kernel.
"""

from __future__ import annotations

import hashlib
import json
import re

from .model2mlir_worklist import build_model2mlir_worklist
from .target_profile import profile_sha256
from .vector_lowering import lower_vector_commands
from .verify_profile_ir import (_bool_attr, _int_attr, _operation_name,
                                _text_attr, verify_ir)


VECTOR_STEPS = (
    ("muls", 384, 0, 384, 256, 1, False, 0x3e00),
    ("rmax", 384, 0, 0x3000, 256, 8, False, 0),
    ("sub", 384, 0x3000, 384, 256, 8, True, 0),
    ("exp", 384, 0, 384, 256, 1, False, 0),
    ("rsum", 384, 0, 0x3020, 256, 8, False, 0),
    ("rcp", 0x3020, 0, 0x3020, 32, 1, False, 0),
    ("mul", 384, 0x3020, 384, 256, 8, True, 0),
)


def _dense_plan() -> dict:
    return {"shape_sq_sk_d": [32, 64, 64], "score_row": 384,
            "probability_row": 640, "v_row": 768, "output_row": 1024,
            "max_row": 0x3000, "sum_row": 0x3020,
            "steps": [list(step) for step in VECTOR_STEPS],
            "rounding": "BF16 after each VPU operation"}


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
    steps = VECTOR_STEPS
    if vector_rows != 256 or row_length != 8:
        raise ValueError("dense attention VPU geometry differs")
    plan = _dense_plan()
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


def lower_dense_attention_vector(mlir_text: str, profile: dict,
                                 binding: dict) -> tuple:
    """Check the pinned source-derived schedule before issuing physical commands."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser

    report = verify_ir(mlir_text, profile)
    if (report["contracts"], report["resident_contracts"],
            report["vpu_commands"], report["spad_requants"]) != (0, 0, 7, 1):
        raise ValueError("dense attention middle needs seven VPU and one requant operation")
    if (binding.get("schema") != "mx_gemmini.nicolas_dense_attention_vector_binding.v1" or
            binding.get("plan") != _dense_plan() or
            binding.get("profile_sha256") != profile_sha256(profile) or
            binding.get("bound_mlir_sha256") != _sha(mlir_text.encode())):
        raise ValueError("dense attention vector binding differs from graph or profile")
    for name in ("source_sha256", "source_header_sha256", "frontend_mlir_sha256",
                 "contract_sha256", "policy_sha256", "manifest_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", binding.get(name, "")) is None:
            raise ValueError(f"dense attention {name} binding is malformed")
    policy = _sha(json.dumps(binding["plan"], sort_keys=True,
                             separators=(",", ":")).encode())
    unsigned = {key: value for key, value in binding.items()
                if key not in ("manifest_sha256", "bound_mlir_sha256")}
    manifest = _sha(json.dumps(unsigned, sort_keys=True).encode())
    if binding["policy_sha256"] != policy or binding["manifest_sha256"] != manifest:
        raise ValueError("dense attention policy or manifest digest differs")

    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    for attr, key in (("mx.profile_sha256", "profile_sha256"),
                      ("mx.contract_sha256", "contract_sha256"),
                      ("mx.policy_sha256", "policy_sha256"),
                      ("mx.source_mlir_sha256", "frontend_mlir_sha256"),
                      ("prov.quantization_manifest_sha256", "manifest_sha256")):
        if _text_attr(module, attr) != binding[key]:
            raise ValueError(f"dense attention module {attr} differs from binding")
    funcs = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(funcs) != 1 or len(funcs[0].body.blocks) != 1 or funcs[0].body.block.args:
        raise ValueError("dense attention vector needs one argument-free function")
    ops = list(funcs[0].body.block.ops)
    if ([_operation_name(op) for op in ops] !=
            ["mx_gemmini.vpu_execute"] * 7 +
            ["mx_gemmini.spad_requant", "func.return"] or
            not isinstance(ops[-1], ReturnOp) or ops[-1].operands or
            any(op.operands or op.results or op.regions for op in ops[:-1])):
        raise ValueError("dense attention vector operation order or SSA differs")
    for index, (op, expected) in enumerate(zip(ops[:7], VECTOR_STEPS)):
        kind, src1, src2, dst, count, rlen, broadcast, immediate = expected
        if (_text_attr(op, "site_id") != f"softmax:{index}" or
                _text_attr(op, "kind") != kind or
                tuple(_int_attr(op, field) for field in
                      ("src1_row", "src2_row", "dst_row", "rows",
                       "reduction_length", "immediate_bf16")) !=
                (src1, src2, dst, count, rlen, immediate) or
                _bool_attr(op, "broadcast") != broadcast):
            raise ValueError(f"dense attention VPU step {index} differs from source")
    req = ops[7]
    if (_text_attr(req, "site_id") != "probability:requant" or
            (_int_attr(req, "source_row"), _int_attr(req, "destination_row"),
             _int_attr(req, "m"), _int_attr(req, "n"),
             _int_attr(req, "scale_dram_address")) != (384, 640, 32, 64, 0) or
            _text_attr(req, "output_format") != "fp8_e4m3" or
            _text_attr(req, "scale_buffer") != "p_scales" or
            not _bool_attr(req, "tiled") or not _bool_attr(req, "resident")):
        raise ValueError("dense attention requant placement or scale binding differs")
    commands = lower_vector_commands(mlir_text, profile)
    if [command.funct for command in commands] != [33] * 7 + [34]:
        raise ValueError("dense attention command schedule differs")
    return commands
