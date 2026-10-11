"""Profile-bound precision proposals for imported model2MLIR contractions.

This is a placement proposal, not an executable quantization manifest.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .target_profile import profile_sha256, require_compute


MODES = {
    "mxfp8": ("fp8_e4m3", "direct", 8),
    "mxfp4": ("fp4_e2m1", "direct", 0),
}


def _round_up(value: int, group: int = 32) -> int:
    return (value + group - 1) // group * group


def _legal(profile: dict, fmt: str) -> bool:
    operand, projection, pe_mode = MODES[fmt]
    try:
        require_compute(profile, operand, operand, pe_mode=pe_mode,
                        activation_projection=projection,
                        weight_projection=projection)
    except ValueError:
        return False
    return True


def build_model_precision_plan(worklist: dict, profile: dict) -> dict:
    """Choose FP4 dense matmuls and FP8 QK/PV on the selected profile.

    FP8 is the general MX baseline.  A linear projection remains a regular
    matrix multiplication even when its parent is an attention module.  QK/PV
    are identified only when a self-attention module has exactly two batched
    contractions; any ambiguous group stays on the host for review.
    """
    if (worklist.get("schema") != "mx_gemmini.model2mlir_contraction_worklist.v1"
            or worklist.get("profile_sha256") != profile_sha256(profile)):
        raise ValueError("model2MLIR worklist and MX profile do not match")
    if not _legal(profile, "mxfp8"):
        raise ValueError("selected profile lacks the required FP8 baseline")
    if profile["resources"]["scale_block_elements"] != 32:
        raise ValueError("model precision proposal needs 32-element MX scale blocks")

    attention = defaultdict(list)
    for site in worklist["other_contraction_regions"]:
        module = site["fqn"] or ""
        if (site["source_op"] == "batch_matmul" and
                module.rsplit(".", 1)[-1] in {"attn", "self_attn"}):
            attention[module].append(site["region_id"])
    attention_stages = {
        region: stage for regions in attention.values() if len(regions) == 2
        for region, stage in zip(regions, ("qk", "pv"))
    }
    batched_shapes = {site["region_id"]: site["shape_bmnk"]
                      for site in worklist["rank3_batch_matmuls"]}
    sites = []
    for site in worklist["rank2_matmuls"]:
        shape = site["shape_mnk"]
        ordinary = site["source_op"] in {"matmul", "addmm"}
        aligned = shape[1] % 32 == 0 and shape[2] % 32 == 0
        preferred = "mxfp4" if ordinary else "mxfp8"
        preferred_legal = _legal(profile, preferred)
        can_pad = (ordinary and profile["geometry"]["mesh_rows"] == 16 and
                   profile["geometry"]["mesh_columns"] == 16 and
                   preferred_legal)
        selected = (preferred if (aligned or can_pad) and preferred_legal else
                    "mxfp8" if aligned else "host")
        padded = [_round_up(value) for value in shape]
        needs_padding = selected != "host" and not aligned
        sites.append({
            "region_id": site["region_id"], "fqn": site["fqn"],
            "source_op": site["source_op"], "shape_mnk": shape,
            "role": "regular_matmul" if ordinary else "other_rank2_contraction",
            "preferred_format": preferred, "selected_format": selected,
            "mx_padded_shape_mnk": padded,
            "output_crop_mn": shape[:2],
            "post_padding_candidate_format": (
                preferred if preferred_legal else "host"),
            "status": ("requires_bf16_cast_padding_and_executable_lowering"
                       if needs_padding and site["element_type"] == "bf16" else
                       "requires_padding_quantization_and_executable_lowering"
                       if needs_padding else
                       "requires_bf16_operand_cast_and_lowering"
                       if selected != "host" and site["element_type"] == "bf16" else
                       "requires_quantization_and_executable_lowering" if selected != "host"
                       else "requires_shape_or_host_lowering"),
            "element_type": site["element_type"],
            "excluded_addmm_bias_epilogue": site["source_op"] == "addmm",
            "reason": ("N_or_K_requires_zero_padding" if needs_padding else
                       None if selected == preferred else
                       "profile_lacks_preferred_mode" if aligned else
                       "K_or_N_is_not_32_aligned"),
        })
    for site in worklist["other_contraction_regions"]:
        stage = attention_stages.get(site["region_id"])
        if stage:
            selected, preferred = "mxfp8", "mxfp8"
            status, reason = "requires_fp8_batched_lowering", None
        else:
            selected, preferred = "host", "host"
            status, reason = "requires_host_lowering", None
        shape = batched_shapes.get(site["region_id"])
        sites.append({
            "region_id": site["region_id"], "fqn": site["fqn"],
            "source_op": site["source_op"], "role": stage or "unclassified",
            **({"shape_bmnk": shape,
                "mx_padded_shape_bmnk": [shape[0]] +
                [_round_up(value) for value in shape[1:]],
                "output_crop_bmn": shape[:3]} if stage and shape else {}),
            "preferred_format": preferred, "selected_format": selected,
            "required_operand_boundary": (
                "quantize_q_and_k_to_mxfp8_with_e8m0_scales" if stage == "qk" else
                "quantize_softmax_probabilities_and_v_to_mxfp8_with_e8m0_scales"
                if stage == "pv" else None),
            "status": status, "reason": reason,
        })
    selected_counts = Counter(site["selected_format"] for site in sites)
    return {
        "schema": "mx_gemmini.model2mlir_precision_proposal.v1",
        "status": "proposal_only_no_whole_model_executable",
        "source_mlir_sha256": worklist["source_mlir_sha256"],
        "profile_sha256": profile_sha256(profile),
        "profile_name": profile["name"],
        "profile_has_vpu": profile["resources"]["vpu"],
        "baseline_format": "mxfp8",
        "mx_scale_format": "e8m0",
        "mx_scale_block_elements": 32,
        "preferred_regular_matmul_format": "mxfp4",
        "preferred_attention_qk_pv_format": "mxfp8",
        "selected_format_counts": dict(sorted(selected_counts.items())),
        "sites": sites,
    }
