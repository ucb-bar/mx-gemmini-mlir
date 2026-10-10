"""Bind Nicolas's BF16 VPU softmax to a model2MLIR softmax capture.

The frontend proves the selected tensor operation and shape. Nicolas's source
defines the BF16 scratchpad schedule and the bit-exact VPU oracle. These are
separate facts: model2MLIR uses f32 intermediates, while the VPU rounds after
each BF16 operation.
"""

from __future__ import annotations

import hashlib
import json
import re

from .target_profile import profile_sha256
from .vector_lowering import lower_vector_commands


ROWS = 64
RLEN = 4
SOURCE_CALLS = (
    "gemmini_vpu_reduce(VPU_RMAX, SP_MX, SP_S, ROWS, RL);",
    "gemmini_vpu_bcast(VPU_SUB, SP_X, SP_S, SP_MX, ROWS, RL);",
    "gemmini_vpu_unary(VPU_EXP, SP_X, SP_X, ROWS);",
    "gemmini_vpu_reduce(VPU_RSUM, SP_SUM, SP_X, ROWS, RL);",
    "gemmini_vpu_unary(VPU_RCP, SP_SUM, SP_SUM, M);",
    "gemmini_vpu_bcast(VPU_MUL, SP_P, SP_X, SP_SUM, ROWS, RL);",
)
STEPS = (
    ("rmax", 0x0000, 0, 0x1000, ROWS, RLEN, False),
    ("sub", 0x0000, 0x1000, 0x2000, ROWS, RLEN, True),
    ("exp", 0x2000, 0, 0x2000, ROWS, 1, False),
    ("rsum", 0x2000, 0, 0x3000, ROWS, RLEN, False),
    ("rcp", 0x3000, 0, 0x3000, 16, 1, False),
    ("mul", 0x2000, 0x3000, 0x0400, ROWS, RLEN, True),
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit_source(source: str) -> None:
    """Reject a changed source schedule before replacing its six VPU calls."""
    constants = (
        "#define M 16", "#define L 32", "#define RL (L / VPU_LANES)",
        "#define ROWS (M * RL)", "#define SP_S   0x0000",
        "#define SP_P   0x0400", "#define SP_MX  0x1000",
        "#define SP_X   0x2000", "#define SP_SUM 0x3000",
    )
    if any(source.count(line) != 1 for line in constants):
        raise ValueError("Nicolas softmax dimensions or scratchpad addresses changed")
    calls = tuple(line.strip() for line in
                  re.findall(r"^\s*gemmini_vpu_(?:reduce|bcast|unary)\([^\n]*;",
                             source, re.MULTILINE))
    if calls != SOURCE_CALLS:
        raise ValueError("Nicolas softmax VPU schedule differs from the selected lowering")
    if source.count("\n".join("  " + call for call in SOURCE_CALLS)) != 1:
        raise ValueError("Nicolas softmax VPU calls are not one replaceable block")
    if "vpu_ref_exec(VPU_MUL, P_ref, X, sum, ROWS, RL, 1, 0);" not in source:
        raise ValueError("Nicolas softmax bit-exact reference is absent")


def audit_frontend(frontend: str) -> None:
    """Require the one known model2MLIR BF16 softmax decomposition."""
    if (frontend.count("func.func @forward(%0: tensor<16x32xbf16>) -> tensor<16x32xbf16>") != 1 or
            'prov.region_id = "softmax_0"' not in frontend or
            'prov.aten = "aten._softmax.default"' not in frontend or
            frontend.count("linalg.reduce ins(") != 2 or
            any(frontend.count(f"{name} ") != 1 for name in
                ("arith.maximumf", "arith.subf", "math.exp", "arith.addf", "arith.divf")) or
            "func.call" in frontend or "torch.operator" in frontend):
        raise ValueError("model2MLIR did not capture the selected 16x32 BF16 softmax")


def render_softmax_bound(frontend: str, source: str, profile: dict,
                         contract: bytes) -> tuple[str, dict]:
    audit_frontend(frontend)
    audit_source(source)
    if (profile["geometry"]["mesh_columns"] != 16 or
            not profile["resources"].get("vpu")):
        raise ValueError("selected profile lacks Nicolas's DIM16 VPU")
    digest = profile_sha256(profile)
    plan = {"steps": [list(step) for step in STEPS],
            "rounding": "BF16 after each VPU operation",
            "logical_shape": [16, 32], "scratchpad_rows": ROWS}
    manifest = {
        "schema": "mx_gemmini.nicolas_vpu_softmax_binding.v1",
        "source_sha256": _sha(source.encode()),
        "frontend_mlir_sha256": _sha(frontend.encode()),
        "profile_sha256": digest,
        "contract_sha256": _sha(contract),
        "policy_sha256": _sha(json.dumps(plan, sort_keys=True).encode()),
        "plan": plan,
    }
    manifest_digest = _sha(json.dumps(manifest, sort_keys=True).encode())
    common = (f'contract_sha256 = "{manifest["contract_sha256"]}", '
              f'policy_sha256 = "{manifest["policy_sha256"]}", '
              f'manifest_sha256 = "{manifest_digest}", '
              f'profile_sha256 = "{digest}"')
    lines = [
        "builtin.module attributes {",
        f'  mx.contract_sha256 = "{manifest["contract_sha256"]}",',
        f'  mx.policy_sha256 = "{manifest["policy_sha256"]}",',
        f'  prov.quantization_manifest_sha256 = "{manifest_digest}",',
        f'  mx.source_mlir_sha256 = "{manifest["frontend_mlir_sha256"]}",',
        f'  mx.profile_sha256 = "{digest}"',
        "} {", "  func.func @softmax_vpu() {",
    ]
    for index, (kind, src1, src2, dst, rows, rlen, broadcast) in enumerate(STEPS):
        lines.append(
            f'    "mx_gemmini.vpu_execute"() {{site_id = "softmax:{index}", '
            f'kind = "{kind}", src1_row = {src1} : i32, src2_row = {src2} : i32, '
            f'dst_row = {dst} : i32, rows = {rows} : i32, '
            f'reduction_length = {rlen} : i32, '
            f'broadcast = {str(broadcast).lower()}, immediate_bf16 = 0 : i32, '
            f'{common}}} : () -> ()')
    lines.extend(("    func.return", "  }", "}", ""))
    mlir = "\n".join(lines)
    commands = lower_vector_commands(mlir, profile)
    if len(commands) != len(STEPS) or any(command.funct != 33 for command in commands):
        raise ValueError("typed softmax did not lower to six physical VPU commands")
    manifest["manifest_sha256"] = manifest_digest
    manifest["bound_mlir_sha256"] = _sha(mlir.encode())
    return mlir, manifest


def replace_source_commands(source: str) -> str:
    audit_source(source)
    block = "\n".join("  " + call for call in SOURCE_CALLS)
    replacement = "  mx_issue();"
    marker = "#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)"
    if source.count(marker) != 1:
        raise ValueError("Nicolas softmax source guard changed")
    return source.replace(marker, "void mx_issue(void);\n\n" + marker, 1).replace(
        block, replacement, 1)
