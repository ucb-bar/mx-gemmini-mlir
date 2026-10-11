"""First blocked-flash QK tile as a typed, source-bound MX command graph."""

from __future__ import annotations

import hashlib
import json

from .command_ir import Command, Fence
from .nicolas_attention_shape import DenseAttentionShape
from .nicolas_flash_commands import lower_flash_commands
from .target_profile import profile_sha256
from .verify_profile_ir import verify_ir


SCHEMA = "mx_gemmini.nicolas_flash_attention_program.v1"
BUFFERS = ("Q_SCALES_FOLDED", "KT_SCALES", "Q_IN", "KT_IN", "scale_sink", "S0_HW")
BUFFER_BYTES = (128, 256, 4096, 8192, 2048, 4096)
SHAPE = DenseAttentionShape(64, 128, 64)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_flash_qk0(frontend: str, source_core: str, source_wrapper: str,
                     source_header: str, profile: dict) -> tuple[str, dict]:
    """Bind the first QK32 block to the pinned source and model2MLIR capture."""
    from .nicolas_attention_vector import _audit_frontend_edges

    if (DenseAttentionShape.from_header(source_header) != SHAPE or
            '#define BK 32' not in source_wrapper or
            '#include "attn_flash.c"' not in source_wrapper or
            'gemmini_mx_load_scales_2d_gated' not in source_core or
            'gemmini_mxquant_config_mvout_managed' not in source_core or
            'prov.region_id = "matmul_0"' not in frontend):
        raise ValueError("pinned flash QK0 source or frontend differs")
    _audit_frontend_edges(frontend, SHAPE)
    SHAPE.validate(profile)
    if (profile.get("source", {}).get("gemmini_revision") !=
            "266c593f2cb51d7e3fe83fc0317072b585ac3c52"):
        raise ValueError("flash QK0 needs Nicolas's managed-scale MX RTL")
    policy = {"schema": "mx_gemmini.flash_qk0_policy.v1",
              "shape_sq_sk_d_bk": [64, 128, 64, 32],
              "fold_q_scale_shift": 3, "qk0_weight_half": 0,
              "qk0_activation_half": 1, "score_row": 8192}
    binding = {"schema": "mx_gemmini.flash_qk0_binding.v1",
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
    common = f'profile_sha256 = "{digest}", source_sha256 = "{source}"'
    wait_common = (f'profile_sha256 = "{digest}", contract_sha256 = "{contract}", '
                   f'policy_sha256 = "{policy_digest}", manifest_sha256 = "{manifest}"')
    args = ", ".join(f"%{name}: memref<{length}xi8>"
                     for name, length in zip(BUFFERS, BUFFER_BYTES))
    def memory(op: str, buffer: str, site: str, attrs: str) -> str:
        length = BUFFER_BYTES[BUFFERS.index(buffer)]
        return (f'    "mx_gemmini.{op}"(%{buffer}) {{site_id = "{site}", '
                f'{attrs}, {common}}} : (memref<{length}xi8>) -> ()')
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
        f'    "mx_gemmini.wait"() {{site_id = "scales:q_fence", {wait_common}}} : () -> ()',
        memory("dma_matrix", "Q_IN", "operand:q",
               "matrix_rows = 64 : i32, matrix_cols = 64 : i32, "
               "burst_cols = 64 : i32, spad_row = 0 : i32"),
        memory("dma_matrix", "KT_IN", "operand:kt0",
               "matrix_rows = 64 : i32, matrix_cols = 32 : i32, "
               "burst_cols = 32 : i32, spad_row = 4096 : i32, "
               "source_stride_bytes = 128 : i32, source_offset_bytes = 0 : i32"),
        (f'    "mx_gemmini.flash_setup"() {{site_id = "qk0:setup", '
         f'store_stride_bytes = 64 : i32, {common}}} : () -> ()'),
        memory("gated_scale_dma", "KT_SCALES", "qk0:weight_scales",
               "row_bytes = 32 : i32, rows = 2 : i32, pitch_bytes = 128 : i32, "
               "destination_byte = 0 : i32, selector = 1 : i32"),
        (f'    "mx_gemmini.managed_matmul_spad"() {{site_id = "qk0:matmul", '
         "m = 64 : i32, n = 32 : i32, k = 64 : i32, activation_row = 0 : i32, "
         "weight_end_row = 4224 : i32, output_row = 8192 : i32, "
         "weight_half = 0 : i32, activation_half = 1 : i32, "
         f'scale_sink_buffer = "scale_sink", {common}}} : () -> ()'),
        memory("spad_mvout_linear", "S0_HW", "qk0:readout",
               "total_bytes = 4096 : i32, tile_cols = 16 : i32, spad_row = 8192 : i32"),
        "    func.return", "  }", "}", "",
    ]
    mlir = "\n".join(lines)
    report = verify_ir(mlir, profile)
    if (report["memory_phases"], report["flash_setups"],
            report["spad_matmuls"]) != (5, 1, 1):
        raise ValueError("flash QK0 typed stage inventory differs")
    return mlir, binding


def lower_flash_qk0(mlir_text: str, frontend: str, source_core: str,
                   source_wrapper: str, source_header: str, profile: dict
                   ) -> tuple[Command | Fence, ...]:
    """Reject changed typed bindings, then lower the QK0 graph to RoCC packets."""
    expected, _ = render_flash_qk0(frontend, source_core, source_wrapper,
                                   source_header, profile)
    if mlir_text != expected:
        raise ValueError("flash QK0 typed graph or source binding differs")
    return lower_flash_commands(mlir_text, profile, BUFFERS, BUFFER_BYTES)
