"""MX Gemmini operand preparation and a TorchAO module extension.

The selected RTL revision is f0167390b56fb315deea90ac1fc3983772e92d82.
This module models its BF16-to-MX *operand* conversion. It does not model the
mesh product, 16-lane reduction, packing/LUT projection, or host transfers.
Those require separate accelerator-oracle comparison before qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

GROUP = 32
RTL_COMMIT = "f0167390b56fb315deea90ac1fc3983772e92d82"
MXGEN_COMMIT = "a27ce3cd81513210c21f971ec3977defd13fa21e"
RTL_CONFIG = "GemminiMxFPConfigs.standaloneMxFPConfig"
RTL_CONFIG_CLASS = "GemminiMxFPStandaloneConfig"
# (exponent bits, fraction bits, exponent bias, highest finite positive code)
_FORMATS = {
    "mxfp8": (4, 3, 7, 0x7E),  # OCP E4M3, max 448
    "mxfp6": (3, 2, 3, 0x1F),  # E3M2, max 28
    "mxfp4": (2, 1, 1, 0x07),  # E2M1, max 6
}


def verify_kernel_contract(contract: dict) -> None:
    """Reject a spec update that this handwritten numerical kernel cannot run."""
    if (contract.get("rtl_commit") != RTL_COMMIT or
        contract.get("mxgen_commit") != MXGEN_COMMIT or
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


def _positive_grid(fmt: str, device: torch.device) -> Tensor:
    exp_bits, fraction_bits, bias, highest_code = _FORMATS[fmt]
    codes = torch.arange(highest_code + 1, device=device, dtype=torch.int32)
    exponent = codes >> fraction_bits
    fraction = codes & ((1 << fraction_bits) - 1)
    normal = torch.ldexp(1.0 + fraction.to(torch.float32) / (1 << fraction_bits), exponent - bias)
    subnormal = torch.ldexp(fraction.to(torch.float32), torch.full_like(exponent, 1 - bias - fraction_bits))
    return torch.where(exponent == 0, subnormal, normal)


def _codebook_values(codes: tuple[int, ...], fmt: str, device: torch.device) -> Tensor:
    exponent_bits, fraction_bits, bias, _ = _FORMATS[fmt]
    width = exponent_bits + fraction_bits
    values = []
    for code in codes:
        magnitude_code = code & ((1 << width) - 1)
        exponent = magnitude_code >> fraction_bits
        fraction = magnitude_code & ((1 << fraction_bits) - 1)
        magnitude = (fraction * 2.0 ** (1 - bias - fraction_bits) if exponent == 0
                     else (1.0 + fraction / (1 << fraction_bits)) * 2.0 ** (exponent - bias))
        values.append(-magnitude if code & (1 << width) else magnitude)
    return torch.tensor(values, dtype=torch.float32, device=device)


def _round_grid(x: Tensor, fmt: str, codebook: tuple[int, ...] | None = None) -> tuple[Tensor, Tensor]:
    """Nearest representable number, with even low code on an exact tie."""
    if codebook is not None:
        if fmt != "mxfp6" or len(codebook) != 16 or len(set(codebook)) != 16:
            raise ValueError("FP6 codebook must contain sixteen distinct element codes")
        values = _codebook_values(codebook, fmt, x.device)
        codes = torch.tensor(codebook, dtype=torch.uint8, device=x.device)
        distance = (x.float().unsqueeze(-1) - values).abs()
        # Ties prefer an even low code, then the lower byte value.
        minimum = distance.amin(dim=-1, keepdim=True)
        ties = distance == minimum
        even = (codes.to(torch.int32) & 1) == 0
        rank_codes = torch.where(ties, codes.to(torch.int32) + (~even).to(torch.int32) * 256, 1024)
        chosen = rank_codes.argmin(dim=-1)
        return values[chosen], codes[chosen]
    grid = _positive_grid(fmt, x.device)
    mag = x.abs().to(torch.float32)
    upper = torch.searchsorted(grid.contiguous(), mag.contiguous()).clamp(max=grid.numel() - 1)
    lower = (upper - 1).clamp(min=0)
    lower_distance = mag - grid[lower]
    upper_distance = grid[upper] - mag
    choose_upper = (upper_distance < lower_distance) | (
        (upper_distance == lower_distance) & ((lower & 1) != 0)
    )
    code = torch.where(choose_upper, upper, lower).to(torch.uint8)
    code = code | (torch.signbit(x).to(torch.uint8) << (_FORMATS[fmt][0] + _FORMATS[fmt][1]))
    unsigned_code = code.to(torch.int32) & ((1 << (_FORMATS[fmt][0] + _FORMATS[fmt][1])) - 1)
    if fmt == "mxfp4":
        # E3M1Tofp4 emits code 0 for underflow, independent of the input sign.
        code = torch.where(unsigned_code == 0, torch.zeros_like(code), code)
    magnitude = grid[unsigned_code]
    sign = torch.signbit(x) & ((unsigned_code != 0) if fmt == "mxfp4" else True)
    return torch.where(sign, -magnitude, magnitude), code


def quantize_mx_gemmini(
    value: Tensor, format: str = "mxfp8", axis: int = -1,
    *, codebook: tuple[int, ...] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Quantize each contraction-axis group of 32 BF16 values.

    Returns dequantized BF16 values, unsigned element codes, and E8M0 exponent
    bytes. The exponent tensor replaces the selected dimension with K/32.
    Nonfinite input is refused until its block-poisoning behavior is verified.
    """
    if format not in ("mxfp8", "mxfp6", "mxfp4"):
        raise ValueError(f"unsupported MX format: {format}")
    if (value.ndim < 1 or not -value.ndim <= axis < value.ndim
            or value.shape[axis] < GROUP or value.shape[axis] % GROUP):
        raise ValueError("MX contraction K must be a nonempty multiple of 32")
    bf16 = value.to(torch.bfloat16).movedim(axis, -1)
    if not torch.compiler.is_compiling() and not bool(torch.isfinite(bf16).all()):
        raise ValueError("nonfinite MX blocks require RTL-verified poisoning")
    grouped = bf16.float().reshape(*bf16.shape[:-1], bf16.shape[-1] // GROUP, GROUP)
    maximum = grouped.abs().amax(dim=-1)
    # MxRequantizer uses a 2^-23 floor before floor(log2(max)).
    exponent = torch.floor(torch.log2(maximum.clamp(min=2.0**-23))).to(torch.int32)
    # E8M0 code 255 is reserved; finite BF16 maxima cannot exceed code 254.
    exponent = exponent.clamp(-23, 127)
    scale = torch.exp2(exponent.to(torch.float32)).unsqueeze(-1)
    normalized = grouped / scale
    if format == "mxfp4":
        # RTL rounds via E3M1 before packing E2M1.
        normalized, _ = _round_grid(normalized, "e3m1")
    rounded, codes = _round_grid(normalized, format, codebook)
    dequant = (rounded * scale).reshape(bf16.shape).to(torch.bfloat16).movedim(-1, axis)
    codes = codes.reshape(bf16.shape).movedim(-1, axis)
    return dequant, codes, (exponent + 127).to(torch.uint8).movedim(-1, axis)


# E3M1 is the FP4 intermediate used by BF16ScaleRoundToTiny.
_FORMATS["e3m1"] = (3, 1, 3, 0x0F)
_MAX_MAGNITUDE = {"mxfp8": 448.0, "mxfp6": 28.0, "mxfp4": 6.0, "e3m1": 24.0}


class MXGemminiLinear(nn.Module):
    """Static MX weights and dynamic MX activations for a Linear module.

    Eager output uses PyTorch's BF16 matmul, so it is a fake-quantization
    diagnostic and cannot serve as an accelerator arithmetic golden.
    """

    def __init__(self, source: nn.Linear, format: str,
                 activation_codebook: tuple[int, ...] | None = None,
                 weight_codebook: tuple[int, ...] | None = None) -> None:
        super().__init__()
        self.format = format
        self.activation_codebook = activation_codebook
        self.weight_codebook = weight_codebook
        with torch.no_grad():
            weight, codes, scales = quantize_mx_gemmini(
                source.weight.detach(), format, codebook=weight_codebook)
        self.register_buffer("weight", weight)
        self.register_buffer("weight_codes", codes)
        self.register_buffer("weight_scale_e8m0", scales)
        self.bias = source.bias

    def forward(self, x: Tensor) -> Tensor:
        tile = 16 if self.format == "mxfp8" else 32
        if x.ndim < 2 or x.shape[-2] < tile or x.shape[-2] % tile:
            raise ValueError(f"MX {self.format} Linear M must contain a full tile of {tile}")
        activation = _dequant_operand_for_graph(
            x, self.format, -1, self.activation_codebook).to(torch.bfloat16)
        bias = self.bias.to(torch.bfloat16) if self.bias is not None else None
        return F.linear(activation, self.weight.to(torch.bfloat16), bias).to(x.dtype)


@dataclass(frozen=True)
class MXGemminiContractionOperands:
    """Logical matrices and E8M0 scales before target-specific packing.

    ``activation_codes`` is A[..., M, K], ``weight_codes`` is B[..., K, N],
    ``activation_scales`` is [..., M, K/32], and ``weight_scales`` is
    [..., N, K/32]. The leading batch axes, if any, match exactly.
    This is an operand handoff, not an executable or numerically certified
    accelerator contraction. Bias is intentionally outside the payload.
    """

    format: str
    activation_codes: Tensor
    weight_codes: Tensor
    activation_scales: Tensor
    weight_scales: Tensor


def linear_contraction_operands(module: MXGemminiLinear, activation: Tensor) -> MXGemminiContractionOperands:
    """Prepare one rank-2 Linear site for an out-of-tree MX layout compiler.

    A Linear stores weights as [N, K]. A contraction compiler consumes B[K, N],
    while E8M0 weight scales remain grouped by output channel [N, K/32].
    Dynamic activations are quantized from the caller's actual input.
    """
    if not isinstance(module, MXGemminiLinear):
        raise TypeError("expected an MXGemminiLinear module")
    tile = 16 if module.format == "mxfp8" else 32
    if activation.ndim != 2 or activation.shape[0] < tile or activation.shape[0] % tile:
        raise ValueError(f"MX {module.format} Linear activation M must contain a full tile of {tile}")
    if activation.shape[1] != module.weight_codes.shape[1]:
        raise ValueError("Linear activation K differs from the static weight K")
    _, activation_codes, activation_scales = quantize_mx_gemmini(
        activation, module.format, codebook=module.activation_codebook)
    return MXGemminiContractionOperands(
        module.format,
        activation_codes,
        module.weight_codes.transpose(0, 1),
        activation_scales,
        module.weight_scale_e8m0,
    )


def functional_contraction_operands(
    lhs: Tensor, rhs: Tensor, format: str = "mxfp8", *,
    activation_codebook: tuple[int, ...] | None = None,
    weight_codebook: tuple[int, ...] | None = None,
) -> MXGemminiContractionOperands:
    """Prepare visible matmul operands with independent rank-2 to rank-4 batches.

    The LHS groups along its last axis and the RHS groups along its K axis.
    This exposes element codes and E8M0 bytes only; it does not identify the
    site in an exported graph or schedule the hardware contraction.
    """
    if format not in ("mxfp8", "mxfp6", "mxfp4"):
        raise ValueError(f"unsupported MX format: {format}")
    if lhs.ndim not in (2, 3, 4) or rhs.ndim != lhs.ndim or lhs.shape[:-2] != rhs.shape[:-2]:
        raise ValueError("MX functional matmul needs matching rank-2 to rank-4 batch axes")
    m, k = lhs.shape[-2:]
    rhs_k, n = rhs.shape[-2:]
    tile = 16 if format == "mxfp8" else 32
    if k != rhs_k or k < GROUP or k % GROUP or m < tile or n < tile or m % tile or n % tile:
        raise ValueError(f"MX {format} matmul needs matching K/32 and M/N tile {tile}")
    _, activation_codes, activation_scales = quantize_mx_gemmini(
        lhs, format, axis=-1, codebook=activation_codebook)
    _, weight_codes, weight_scales = quantize_mx_gemmini(
        rhs, format, axis=-2, codebook=weight_codebook)
    return MXGemminiContractionOperands(
        format, activation_codes, weight_codes,
        activation_scales, weight_scales.transpose(-2, -1),
    )


try:
    from torchao.core.config import AOBaseConfig
    from torchao.quantization.transform_module import register_quantize_module_handler
except ImportError:
    AOBaseConfig = object  # type: ignore[assignment,misc]
    def register_quantize_module_handler(_config):  # type: ignore[no-redef]
        return lambda function: function


@dataclass(frozen=True)
class MXGemminiFakeQuantConfig(AOBaseConfig):
    """TorchAO ``quantize_`` config for static weights and dynamic activations."""

    format: str = "mxfp8"
    activation_codebook: tuple[int, ...] | None = None
    weight_codebook: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.format not in ("mxfp8", "mxfp6", "mxfp4"):
            raise ValueError(f"unsupported MX format: {self.format}")


@register_quantize_module_handler(MXGemminiFakeQuantConfig)
def _mx_gemmini_transform(module: nn.Module, config: MXGemminiFakeQuantConfig) -> nn.Module:
    if not isinstance(module, nn.Linear):
        raise TypeError("MXGemminiFakeQuantConfig applies only to nn.Linear")
    if module.in_features % GROUP:
        raise ValueError(f"{module.in_features} is not divisible by MX block size {GROUP}")
    tile = 16 if config.format == "mxfp8" else 32
    if module.out_features % tile:
        raise ValueError(f"{module.out_features} is not divisible by MX N tile {tile}")
    return MXGemminiLinear(module, config.format,
                           config.activation_codebook, config.weight_codebook)


def apply_mx_gemmini_(model: nn.Module, format: str = "mxfp8") -> nn.Module:
    """Apply the registered TorchAO transform, preserving its native API."""
    try:
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError("torchao is required for the MX module transform") from exc
    quantize_(model, MXGemminiFakeQuantConfig(format=format))
    return model


def _dequant_operand_for_graph(value: Tensor, format: str, axis: int,
                               codebook: tuple[int, ...] | None = None) -> Tensor:
    """Exportable arithmetic Q/DQ, equivalent to finite BF16 grid lookup."""
    if not torch.compiler.is_compiling() and not bool(torch.isfinite(value).all()):
        raise ValueError("nonfinite MX blocks require RTL-verified poisoning")
    bf16 = value.to(torch.bfloat16).movedim(axis, -1)
    grouped = bf16.float().reshape(*bf16.shape[:-1], bf16.shape[-1] // GROUP, GROUP)
    exponent = torch.floor(torch.log2(grouped.abs().amax(dim=-1).clamp(min=2.0**-23)))
    scale = torch.exp2(exponent.clamp(-23, 127)).unsqueeze(-1)
    normalized = grouped / scale
    if format == "mxfp4":
        normalized = _round_grid_arithmetic(normalized, "e3m1")
    rounded = _round_grid_arithmetic(normalized, format, codebook)
    return (rounded * scale).reshape(bf16.shape).to(torch.bfloat16).movedim(-1, axis).to(value.dtype)


def _round_grid_arithmetic(value: Tensor, format: str,
                           codebook: tuple[int, ...] | None = None) -> Tensor:
    if codebook is not None:
        if format != "mxfp6":
            raise ValueError("codebooks apply only to FP6")
        values = _codebook_values(codebook, format, value.device)
        priority = torch.tensor([code + (code % 2) * 256 for code in codebook],
                                dtype=torch.int32, device=value.device)
        distance = (value.unsqueeze(-1) - values).abs()
        minimum = distance.amin(dim=-1, keepdim=True)
        ranks = torch.where(distance == minimum, priority,
                            torch.full_like(priority, 1024))
        return values[ranks.argmin(dim=-1)]
    _exp_bits, fraction_bits, bias, _highest_code = _FORMATS[format]
    emin = 1 - bias
    smallest_normal = 2.0 ** emin
    magnitude = value.abs()
    normal_exponent = torch.floor(torch.log2(magnitude.clamp(min=2.0**-126)))
    exponent = torch.where(magnitude < smallest_normal,
                           torch.full_like(normal_exponent, float(emin)), normal_exponent)
    quantum = torch.exp2(exponent - fraction_bits)
    rounded = (torch.round(magnitude / quantum) * quantum).clamp(max=_MAX_MAGNITUDE[format])
    signed = torch.where(value < 0, -rounded, rounded)
    if format == "mxfp4":
        # The RTL's E3M1-to-E2M1 conversion canonicalizes every zero to +0.
        signed = torch.where(rounded == 0, torch.zeros_like(signed), signed)
    else:
        # A comparison treats -0 as zero. Multiplication by +0 retains its sign
        # and uses ops that the generic MLIR importer can lower.
        signed = torch.where(value == 0, value * 0.0, signed)
    return signed


def expose_sdpa_contractions(exported: torch.export.ExportedProgram) -> torch.export.ExportedProgram:
    """Expose supported inference SDPA as QK, host softmax, and PV."""
    target = torch.ops.aten.scaled_dot_product_attention.default
    if not any(node.target == target for node in exported.graph_module.graph.nodes):
        return exported

    def decompose(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, *, scale=None, enable_gqa=False):
        if dropout_p != 0 or is_causal or enable_gqa:
            raise ValueError("MX attention capture does not support dropout, causal mode, or grouped heads")
        if (query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or
                query.shape[:-2] != key.shape[:-2] or
                query.shape[:-2] != value.shape[:-2] or
                key.shape[-2] != value.shape[-2]):
            raise ValueError("MX attention capture requires matching rank-4 Q/K/V batches and heads")
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            raise ValueError("MX attention capture currently requires a boolean mask")
        factor = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])
        scores = torch.matmul(query, key.transpose(-2, -1)) * factor
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        probabilities = torch.ops.aten._safe_softmax.default(scores, -1, None)
        return torch.matmul(probabilities, value)

    return exported.run_decompositions({target: decompose})


def quantize_functional_contractions_(graph_module: torch.fx.GraphModule, select,
                                      module_fqns: frozenset[str] = frozenset()) -> list[dict]:
    """Insert MX Q/DQ on eligible functional matmul edges of an exported graph.

    TorchAO's module handler covers ``nn.Linear``. This pass covers matmul and
    functional linear sites visible after ``torch.export``. Unsupported or
    unknown shapes are reported; a fused SDPA node raises rather than silently
    claiming that attention was quantized.
    """
    import torch.fx as fx

    targets = {
        torch.ops.aten.matmul.default: (0, 1, False),
        torch.ops.aten.mm.default: (0, 1, False),
        torch.ops.aten.bmm.default: (0, 1, False),
        torch.ops.aten.linear.default: (0, 1, True),
        torch.ops.aten.addmm.default: (1, 2, False),
    }
    census: list[dict] = []
    for node in tuple(graph_module.graph.nodes):
        if "scaled_dot_product_attention" in str(node.target):
            raise ValueError("fused SDPA hides attention contractions; export eager attention for MX")
        if node.target not in targets:
            continue
        stack = node.meta.get("nn_module_stack") or {}
        if node.target == torch.ops.aten.linear.default and any(
            isinstance(row, (tuple, list)) and row and row[0] in module_fqns
            for row in stack.values()
        ):
            # Every original Linear, including host and skipped sites, has a
            # module census entry and must keep that disposition.
            continue
        site_id = f"functional:{node.name}"
        format, codebooks = select(site_id)
        if format == "host":
            census.append({"site_id": site_id, "kind": "functional", "status": "host"})
            continue
        if format not in ("mxfp8", "mxfp6", "mxfp4"):
            raise ValueError(f"unsupported MX format: {format}")
        tile = 16 if format == "mxfp8" else 32
        # The TorchAO handler already inserted dynamic activation quantization
        # and materialized static weight codes for this Linear.
        if any("MXGemminiLinear" in str(value) for value in stack.values()):
            # The module handler already prepared both operands.
            continue
        lhs_index, rhs_index, is_linear = targets[node.target]
        lhs, rhs = node.args[lhs_index], node.args[rhs_index]
        if not isinstance(lhs, fx.Node) or not isinstance(rhs, fx.Node):
            census.append({"site_id": site_id, "kind": "functional", "status": "skipped", "reason": "unobserved operand"})
            continue
        lhs_val, rhs_val = lhs.meta.get("val"), rhs.meta.get("val")
        if not isinstance(lhs_val, Tensor) or not isinstance(rhs_val, Tensor):
            census.append({"site_id": site_id, "kind": "functional", "status": "skipped", "reason": "unobserved shape"})
            continue
        a, b = tuple(lhs_val.shape), tuple(rhs_val.shape)
        if len(a) < 2 or len(b) != 2 and len(b) != len(a):
            census.append({"site_id": site_id, "kind": "functional", "status": "skipped", "reason": "rank or broadcasting"})
            continue
        m, k = a[-2:]
        n = b[-2] if is_linear else b[-1]
        rhs_k = b[-1] if is_linear else b[-2]
        if not all(isinstance(dim, int) for dim in (m, n, k, rhs_k)):
            census.append({"site_id": site_id, "kind": "functional", "status": "skipped", "reason": "symbolic dimensions"})
            continue
        if k != rhs_k or k % GROUP or m % tile or n % tile:
            census.append({"site_id": site_id, "kind": "functional", "status": "skipped", "reason": f"shape M={m} N={n} K={k} outside MX tile"})
            continue
        if len(a) > 2 and not is_linear and a[:-2] != b[:-2]:
            census.append({"site_id": site_id, "kind": "functional", "status": "skipped", "reason": "batch broadcasting"})
            continue
        with graph_module.graph.inserting_before(node):
            qlhs = graph_module.graph.call_function(_dequant_operand_for_graph,
                                                    args=(lhs, format, -1, codebooks[0]))
            qrhs = graph_module.graph.call_function(_dequant_operand_for_graph,
                                                    args=(rhs, format, -1 if is_linear else -2, codebooks[1]))
        arguments = list(node.args)
        arguments[lhs_index] = qlhs
        arguments[rhs_index] = qrhs
        node.args = tuple(arguments)
        census.append({"site_id": site_id, "kind": "functional", "status": "quantized",
                       "format": format, "shape": [int(m), int(n), int(k)],
                       "fp6_codebook_sha256": None})
    graph_module.graph.lint()
    graph_module.recompile()
    return census
