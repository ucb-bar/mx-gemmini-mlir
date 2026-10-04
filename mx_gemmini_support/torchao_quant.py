"""MX Gemmini contract gate over the shared mxq operand capture kernel."""

from __future__ import annotations

import torch

from mxq.nn.operand_capture import (
    MXOperandContractionOperands as MXGemminiContractionOperands,
    MXOperandFakeQuantConfig as MXGemminiFakeQuantConfig,
    MXOperandLinear as MXGemminiLinear,
    dequant_mx_operand_for_graph as _dequant_operand_for_graph,
    expose_sdpa_contractions,
    functional_contraction_operands as _functional_contraction_operands,
    linear_contraction_operands,
    quantize_functional_contractions_ as _quantize_functional_contractions_,
    quantize_mx_operand as quantize_mx_gemmini,
)

from .legality import shape_reason

GROUP = 32
_OPERAND_REVISIONS = frozenset({
    ("f0167390b56fb315deea90ac1fc3983772e92d82",
     "a27ce3cd81513210c21f971ec3977defd13fa21e"),
    ("2029218197f771ce71416f859d975bea47b7aabc",
     "56ef1c6810924e1cb0af07add09156b0e2f53576"),
})
RTL_CONFIG = "GemminiMxFPConfigs.standaloneMxFPConfig"
RTL_CONFIG_CLASS = "GemminiMxFPStandaloneConfig"
# (exponent bits, fraction bits, exponent bias, highest finite positive code)
_FORMATS = {
    "mxfp8": (4, 3, 7, 0x7E),  # OCP E4M3, max 448
    "mxfp6": (3, 2, 3, 0x1F),  # E3M2, max 28
    "mxfp4": (2, 1, 1, 0x07),  # E2M1, max 6
    "e3m1": (3, 1, 3, 0x0F),
}


def verify_kernel_contract(contract: dict) -> None:
    """Reject a spec update that this handwritten numerical kernel cannot run."""
    if ((contract.get("rtl_commit"), contract.get("mxgen_commit")) not in _OPERAND_REVISIONS or
        contract.get("rtl_config") != RTL_CONFIG or
        contract.get("rtl_config_class") != RTL_CONFIG_CLASS):
        raise ValueError("MX kernel needs review for the selected RTL and MxGen revisions")
    if contract.get("block_size") != GROUP or contract.get("scale_encoding") != "e8m0":
        raise ValueError("MX kernel does not implement the selected block/scale contract")
    if contract.get("operand_rounding") != "rne" or contract.get("scale_rule") != (
        "floor_log2_of_bf16_block_max_with_2pow_minus23_floor"
    ) or contract.get("zero_block_scale_e8m0") != 104:
        raise ValueError("MX kernel does not implement the selected rounding/scale contract")
    formats = contract.get("formats") or {}
    for name in ("mxfp8", "mxfp6", "mxfp4"):
        row = (formats.get(name) or {}).get("element") or {}
        expected = _FORMATS[name]
        observed = tuple(row.get(key) for key in (
            "exponent_bits", "fraction_bits", "exponent_bias", "max_positive_code"
        ))
        if observed != expected:
            raise ValueError(f"MX kernel format {name} differs from selected software contract")
    intermediate = contract.get("e3m1_intermediate") or {}
    observed = tuple(intermediate.get(key) for key in (
        "exponent_bits", "fraction_bits", "exponent_bias", "max_positive_code"
    ))
    if observed != _FORMATS["e3m1"]:
        raise ValueError("MX kernel FP4 intermediate differs from selected software contract")


def apply_mx_gemmini_(model, format: str = "mxfp8"):
    """Standalone operand diagnostic with this RTL's tile bounds."""
    from torchao.quantization import quantize_

    tile = 16 if format == "mxfp8" else 32
    quantize_(model, MXGemminiFakeQuantConfig(
        format=format, m_rule=(tile, tile), n_rule=(tile, tile), k_rule=(GROUP, GROUP)))
    return model


def functional_contraction_operands(lhs, rhs, format: str = "mxfp8", *,
                                    activation_codebook=None, weight_codebook=None):
    """Check this RTL's geometry before using the shared operand encoder."""
    tile = 16 if format == "mxfp8" else 32
    if (lhs.ndim not in (2, 3, 4) or rhs.ndim != lhs.ndim
            or lhs.shape[-2] < tile or lhs.shape[-2] % tile
            or rhs.shape[-1] < tile or rhs.shape[-1] % tile):
        raise ValueError(f"MX {format} matmul needs M/N tile {tile}")
    return _functional_contraction_operands(
        lhs, rhs, format, activation_codebook=activation_codebook,
        weight_codebook=weight_codebook)


def quantize_functional_contractions_(graph_module: torch.fx.GraphModule, select,
                                      module_fqns: frozenset[str] = frozenset(), *,
                                      contract: dict, insert: bool = True,
                                      codebooks_for=None) -> list[dict]:
    """Apply the shared graph rewrite with this target's shape contract."""
    return _quantize_functional_contractions_(
        graph_module, select, module_fqns, contract=contract,
        shape_reason=shape_reason, insert=insert, codebooks_for=codebooks_for,
    )
