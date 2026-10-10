"""Lower a payload-bound MX contraction to a physical Rocket command stream.

The stream is source-independent after payload export. The Spike serial mode
uses explicit operand DMA because the pinned Spike model does not execute the
source loop-FSM DMA-only launches. FP6 also reloads scale half zero because
the pinned model ignores the alternating selector in LUT compute. The RTL
mode keeps alternating halves; an isolated corrected Spike experiment covers
the two-wave FP6 source case without qualifying the RTL or pinned model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping

from .command_ir import Command, Fence, Operand, spad_requant_command, vpu_command
from .quant_reference import (bf16_add_scalar, bf16_mul_scalar, exact_bf16_x2,
                              quantize_bf16_radiance_header_fp8,
                              quantize_bf16_radiance_header_fp6)
from .source_gemm import plan_mx_gemm
from .source_payload import (manifest_json, manifest_sha256,
                             TARGET_MESH_QUANT_CONVENTION,
                             vpu_requant_shape_is_legal)
from .target_profile import profile_sha256
from .verify_profile_ir import _bool_attr, _int_attr, _operation_name, _text_attr, verify_ir


@dataclass(frozen=True)
class PhysicalStep:
    phase: str
    wave: int | None
    command: Command | Fence


@dataclass(frozen=True)
class PhysicalProgram:
    profile_sha256: str
    payload_manifest_sha256: str
    mode: str
    shape: tuple[int, int, int]
    plan: dict
    steps: tuple[PhysicalStep, ...]
    source_golden_preserving: bool = True
    derived_expected_bf16: bytes | None = None
    output_format: str = "bf16"
    tiled_quant_readout: bool = False
    derived_vpu_scalar_bf16: int | None = None
    derived_vpu_scalar_chain: tuple[tuple[str, int], ...] | None = None
    golden_origin: str = "source_header"

    def receipt(self) -> dict:
        steps = []
        for step in self.steps:
            serialized = asdict(step)
            for field in ("rs1", "rs2"):
                operand = serialized["command"].get(field)
                if operand is not None and operand.get("address_shift") == 0:
                    # Preserve v1 receipt hashes for commands that predate
                    # buffered, shifted runtime pointers.
                    del operand["address_shift"]
            steps.append(serialized)
        receipt = {"schema": "mx_gemmini.physical_program.v1",
                "profile_sha256": self.profile_sha256,
                "payload_manifest_sha256": self.payload_manifest_sha256,
                "mode": self.mode, "shape_mnk": list(self.shape),
                "plan": self.plan,
                "steps": steps}
        if (not self.source_golden_preserving or
                any(step.phase in {"vpu", "spad_requant"} for step in self.steps)):
            receipt["source_golden_preserving"] = self.source_golden_preserving
        if self.derived_expected_bf16 is not None:
            receipt["golden_derivation"] = (
                "bf16_scalar_chain_rne" if self.derived_vpu_scalar_chain is not None
                else "bf16_exact_multiply_by_two" if self.derived_vpu_scalar_bf16 is None
                else "bf16_scalar_adds_rne" if self.plan.get("vector_tile_policy") ==
                "bf16_adds_scalar_each_output_tile_v1" else "bf16_scalar_muls_rne")
            receipt["derived_expected_bf16_sha256"] = hashlib.sha256(
                self.derived_expected_bf16).hexdigest()
            if self.derived_vpu_scalar_bf16 is not None:
                receipt["derived_vpu_scalar_bf16"] = self.derived_vpu_scalar_bf16
            if self.derived_vpu_scalar_chain is not None:
                receipt["derived_vpu_scalar_chain"] = [
                    {"kind": kind, "immediate_bf16": scalar}
                    for kind, scalar in self.derived_vpu_scalar_chain]
        if self.output_format != "bf16":
            receipt["output_format"] = self.output_format
            receipt["source_golden_preserving"] = self.source_golden_preserving
        if self.tiled_quant_readout:
            receipt["tiled_quant_readout"] = True
        if self.golden_origin != "source_header":
            receipt["golden_origin"] = self.golden_origin
        return receipt


def _imm(value: int) -> Operand:
    return Operand(immediate=value)


def _cmd(funct: int, rs1: int | Operand, rs2: int | Operand) -> Command:
    return Command(funct, _imm(rs1) if isinstance(rs1, int) else rs1,
                   _imm(rs2) if isinstance(rs2, int) else rs2)


def _config_ld(stride: int, *, id: int = 0, dim: int = 16) -> Command:
    # Nicolas gemmini.h gemmini_extended5_config_ld, scale identity = 0.
    return _cmd(0, (dim << 16) | (1 << 8) | (id << 3) | 1, stride)


def _config_st(stride: int) -> Command:
    return _cmd(0, 2, stride)


def _transfer(funct: int, buffer: str, offset: int, row: int, *, dim: int = 16) -> Command:
    return _cmd(funct, Operand(buffer=buffer, byte_offset=offset),
                (dim << 48) | (dim << 32) | row)


def _transfer_rect(funct: int, buffer: str, offset: int, row: int, *,
                   rows: int, cols: int = 16) -> Command:
    if not 1 <= rows <= 16 or cols not in {8, 16, 32}:
        raise ValueError("MX BF16 row-major transfer needs 1..16 mesh-width rows")
    return _cmd(funct, Operand(buffer=buffer, byte_offset=offset),
                (rows << 48) | (cols << 32) | row)


def _exact_bf16_x2(source: bytes) -> bytes:
    return exact_bf16_x2(source)


def _check_binding(mlir_text: str, profile: dict, manifest: dict) -> tuple[
        list[tuple[str, dict]], str | None, str | None, str | None]:
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, ReturnOp
    from xdsl.parser import Parser

    checked = verify_ir(mlir_text, profile)
    if (checked["contracts"], checked["encodes"], checked["requantizes"]) != (1, 0, 0):
        raise ValueError("physical MX source lowering needs exactly one BF16 contraction")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    digest = manifest_sha256(manifest)
    if (_text_attr(module, "mx.payload_manifest_sha256") != digest or
            (_text_attr(module, "mx.payload_manifest_json") is not None and
             _text_attr(module, "mx.payload_manifest_json") != manifest_json(manifest)) or
            manifest.get("profile_sha256") != profile_sha256(profile)):
        raise ValueError("physical MX lowering payload or target digest differs")
    ops = [op for op in module.walk()
           if _operation_name(op).startswith("mx_gemmini.") and
           _operation_name(op) not in {"mx_gemmini.resource", "mx_gemmini.upload_lut"}]
    contract = [op for op in ops if _operation_name(op) == "mx_gemmini.contract"]
    output_format = manifest.get("output_format", "bf16")
    specialization = _text_attr(module, "mx.output_specialization")
    host_kind = {
        "radiance_header_fp8_host_requant": "radiance_header_fp8",
        "radiance_header_fp6_host_requant": "radiance_header_fp6",
    }.get(specialization)
    host_requant = host_kind is not None
    vector_requant = specialization == "matrix_vpu_x2_spad_requant_fp8"
    readout_name = ("mx_gemmini.readout_bf16" if output_format == "bf16" or
                    host_requant or vector_requant
                    else "mx_gemmini.readout_quantized")
    readout = [op for op in ops if _operation_name(op) == readout_name]
    names = [_operation_name(op) for op in ops]
    tilewise_policy = _text_attr(module, "mx.vector_tile_policy")
    tilewise_scalar = _int_attr(module, "mx.vector_scalar_bf16")
    tilewise_chain_json = _text_attr(module, "mx.vector_scalar_chain_json")
    if (len(contract) != 1 or len(readout) != 1 or
            names[0] != "mx_gemmini.contract" or
            names[-1] != ("mx_gemmini.spad_requant" if vector_requant else
                          "mx_gemmini.host_requantize" if host_requant else readout_name) or
            (not vector_requant and any(
                name not in {"mx_gemmini.vpu_execute", "mx_gemmini.spad_requant"}
                for name in names[1:-2 if host_requant else -1])) or
            (vector_requant and names != ["mx_gemmini.contract",
                                         "mx_gemmini.readout_bf16",
                                         "mx_gemmini.vpu_execute",
                                         "mx_gemmini.spad_requant"]) or
            (host_requant and (names[-2] != readout_name or len(names) != 3))):
        raise ValueError("physical MX lowering requires a matching output readout")
    memory_layout = _text_attr(readout[0], "memory_layout")
    if memory_layout is not None and (
            memory_layout not in {"row_major_bf16", "output_tile_major_bf16"} or
            output_format != "bf16" or host_requant or vector_requant):
        raise ValueError("explicit MX memory layout requires final BF16 readout")
    if tilewise_policy is not None:
        function = readout[0].parent_op()
        if (tilewise_policy not in {"bf16_muls_x2_each_output_tile_v1",
                                    "bf16_muls_scalar_each_output_tile_v1",
                                    "bf16_adds_scalar_each_output_tile_v1",
                                    "bf16_scalar_chain_each_output_tile_v1"} or
                (tilewise_policy == "bf16_muls_x2_each_output_tile_v1" and
                 tilewise_scalar is not None) or
                (tilewise_policy in {"bf16_muls_scalar_each_output_tile_v1",
                                     "bf16_adds_scalar_each_output_tile_v1"} and
                 (tilewise_scalar is None or not 0 <= tilewise_scalar <= 0xffff or
                  (tilewise_scalar & 0x7f80) == 0x7f80)) or
                (tilewise_policy == "bf16_scalar_chain_each_output_tile_v1" and
                 (tilewise_scalar is not None or tilewise_chain_json is None)) or
                (tilewise_policy != "bf16_scalar_chain_each_output_tile_v1" and
                 tilewise_chain_json is not None) or
                output_format != "bf16" or
                (names != ["mx_gemmini.contract", *[
                    "mx_gemmini.vpu_execute"] * (len(names) - 2),
                    "mx_gemmini.readout_bf16"]) or
                (len(names) != 3 if tilewise_policy !=
                 "bf16_scalar_chain_each_output_tile_v1" else
                 not 4 <= len(names) <= 18) or
                list(ops[1].operands) != list(contract[0].results) or
                any(list(right.operands) != list(left.results)
                    for left, right in zip(ops[1:-2], ops[2:-1])) or
                list(readout[0].operands) != list(ops[-2].results) or
                not isinstance(function, FuncOp) or
                not isinstance(function.get_return_op(), ReturnOp) or
                list(function.get_return_op().operands) != list(readout[0].results)):
            raise ValueError("physical MX tilewise VPU policy differs from bound contraction")
    if (manifest.get("site_id") != _text_attr(contract[0], "site_id") or
            manifest.get("site_id") != _text_attr(readout[0], "site_id") or
            _text_attr(contract[0], "payload_manifest_sha256") != digest):
        raise ValueError("physical MX lowering site or payload binding differs")
    if output_format != "bf16" and not host_requant and not vector_requant and \
            _text_attr(readout[0], "output_format") != output_format:
        raise ValueError("physical MX output format differs from source specialization")
    if vector_requant:
        first, bf16_readout, vpu, requant = ops
        function = requant.parent_op()
        m, n, _ = manifest["shape_mnk"]
        if (manifest.get("output_format") != "fp8_e4m3" or
                list(bf16_readout.operands) != list(first.results) or
                list(vpu.operands) != list(bf16_readout.results) or
                list(requant.operands) != list(vpu.results) or
                not isinstance(function, FuncOp) or
                not isinstance(function.get_return_op(), ReturnOp) or
                list(function.get_return_op().operands) != list(requant.results) or
                [str(result.type) for result in (*bf16_readout.results, *vpu.results,
                                                 *requant.results)] != [
                    f"tensor<{m}x{n}xbf16>", f"tensor<{m}x{n}xbf16>",
                    f"tensor<{m}x{n}xi8>", f"tensor<{m}x{n // 32}xi8>"]):
            raise ValueError("physical MX VPU/SPAD SSA handoff differs from source specialization")
    if host_requant:
        host = ops[-1]
        function = host.parent_op()
        precision = manifest.get("precision")
        policy = ("radiance_header_fp6_lut_v1" if host_kind == "radiance_header_fp6"
                  else "radiance_header_fp8_v1")
        output = "fp6_e3m2" if host_kind == "radiance_header_fp6" else "fp8_e4m3"
        allowed_precision = ({"FP6"} if host_kind == "radiance_header_fp6"
                             else {"FP8", "FP4"})
        expected_operands = list(readout[0].results)
        if host_kind == "radiance_header_fp6":
            lut = [op for op in module.walk()
                   if _operation_name(op) == "mx_gemmini.resource" and
                   _text_attr(op, "resource_name") == "output_lut"]
            if len(lut) != 1:
                raise ValueError("physical FP6 header epilogue needs a checked output LUT")
            expected_operands += list(lut[0].results)
        if (precision not in allowed_precision or
                manifest.get("source_quant_golden_convention") not in {
                    "source_header", TARGET_MESH_QUANT_CONVENTION} or
                _text_attr(host, "site_id") != manifest["site_id"] or
                _text_attr(host, "quant_policy") != policy or
                _text_attr(host, "output_format") != output or
                list(host.operands) != expected_operands or
                list(readout[0].operands) != list(contract[0].results) or
                not isinstance(function, FuncOp) or
                not isinstance(function.get_return_op(), ReturnOp) or
                list(function.get_return_op().operands) != list(host.results)):
            raise ValueError("physical MX Radiance header epilogue binding differs")
    vector_ops = []
    for op in (ops[2:] if vector_requant else
               ops[1:-2 if host_requant else -1]):
        if _text_attr(op, "site_id") != manifest["site_id"]:
            raise ValueError("physical MX vector operation belongs to another contraction site")
        if _operation_name(op) == "mx_gemmini.vpu_execute":
            vector_ops.append(("vpu", {
                "kind": _text_attr(op, "kind"),
                "src1_row": _int_attr(op, "src1_row"),
                "src2_row": _int_attr(op, "src2_row"),
                "dst_row": _int_attr(op, "dst_row"),
                "rows": _int_attr(op, "rows"),
                "reduction_length": _int_attr(op, "reduction_length"),
                "broadcast": _bool_attr(op, "broadcast"),
                "immediate_bf16": _int_attr(op, "immediate_bf16"),
                "second_dst_row": _int_attr(op, "second_dst_row"),
            }))
        else:
            vector_ops.append(("spad_requant", {
                "source_row": _int_attr(op, "source_row"),
                "destination_row": _int_attr(op, "destination_row"),
                "m": _int_attr(op, "m"), "n": _int_attr(op, "n"),
                "output_format": _text_attr(op, "output_format"),
                "tiled": _bool_attr(op, "tiled"),
                "resident": _bool_attr(op, "resident"),
                "scale_dram_address": _int_attr(op, "scale_dram_address"),
                "scale_buffer": _text_attr(op, "scale_buffer"),
            }))
    if (tilewise_policy in {"bf16_muls_scalar_each_output_tile_v1",
                            "bf16_adds_scalar_each_output_tile_v1"} and
            (len(vector_ops) != 1 or vector_ops[0][0] != "vpu" or
             vector_ops[0][1]["immediate_bf16"] != tilewise_scalar)):
        raise ValueError("physical MX tilewise VPU scalar differs from command immediate")
    if tilewise_policy == "bf16_scalar_chain_each_output_tile_v1":
        try:
            chain = json.loads(tilewise_chain_json)
        except (TypeError, ValueError) as error:
            raise ValueError("physical MX scalar chain policy is malformed") from error
        expected = [{"kind": op[1]["kind"],
                     "immediate_bf16": op[1]["immediate_bf16"]}
                    for op in vector_ops]
        if (type(chain) is not list or chain != expected or
                len(chain) != len(vector_ops) or
                any(type(item) is not dict or set(item) !=
                    {"kind", "immediate_bf16"} or
                    item["kind"] not in {"muls", "adds"} or
                    type(item["immediate_bf16"]) is not int or
                    not 0 <= item["immediate_bf16"] <= 0xffff or
                    (item["immediate_bf16"] & 0x7f80) == 0x7f80
                    for item in chain)):
            raise ValueError("physical MX scalar chain differs from VPU commands")
    if tilewise_policy is None and (tilewise_scalar is not None or
                                    tilewise_chain_json is not None):
        raise ValueError("physical MX VPU scalar requires a tilewise policy")
    return vector_ops, host_kind, tilewise_policy, memory_layout


def lower_bound_source(mlir_text: str, profile: dict, manifest: dict,
                       resources: Mapping[str, bytes], *,
                       mode: str = "spike_serial") -> PhysicalProgram:
    """Lower a checked payload-bound contraction into ordered RoCC commands."""
    vector_ops, host_kind, tilewise_policy, memory_layout = _check_binding(
        mlir_text, profile, manifest)
    if manifest.get("origin") == "radiance_source_target_mesh_reference" and (
            manifest["target_mesh_reference"]["mesh_dim"] !=
            profile["geometry"]["mesh_columns"]):
        raise ValueError("physical MX target mesh reference differs from profile")
    host_requant = host_kind is not None
    if mode not in {"spike_serial", "rtl_alternating"}:
        raise ValueError("unknown MX physical scheduling mode")
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("physical source lowering currently requires Rocket RoCC")
    precision = manifest["precision"]
    shape = tuple(manifest["shape_mnk"])
    tile = tuple(manifest["tile_mnk"])
    output_format = (host_kind if host_requant else
                     manifest.get("output_format", "bf16"))
    vector_requant = manifest.get("output_specialization") == "matrix_vpu_x2_spad_requant_fp8"
    if (precision, output_format) not in {("FP8", "bf16"), ("FP4", "bf16"),
                                          ("FP6", "bf16"), ("FP8", "fp8_e4m3"),
                                          ("FP4", "fp4_e2m1"), ("FP6", "fp6_e3m2"),
                                          ("FP8", "radiance_header_fp8"),
                                          ("FP4", "radiance_header_fp8"),
                                          ("FP6", "radiance_header_fp6")}:
        raise ValueError("physical MX source precision and output format differ")
    quant_output = output_format not in {"bf16", "radiance_header_fp8",
                                         "radiance_header_fp6"}
    matrix_quant_output = quant_output and not vector_requant
    plan = plan_mx_gemm(shape=shape, tile=tile, datatype=precision,
                        quant_output=matrix_quant_output, acc_to_gmem=False,
                        scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
                        profile=profile)
    m, n, k = shape
    tm, tn, tk = tile
    dim = profile["geometry"]["mesh_columns"]
    if m % tm or n % tn or tk % dim or plan["c_rows"] % dim:
        raise ValueError("physical source lowering needs complete mesh-width output tiles")
    output_tiles = plan.get("output_tiles", [{"index": 0, "m_start": 0, "n_start": 0}])
    if memory_layout == "row_major_bf16" and len(output_tiles) > 1:
        if tn * 2 % dim or not 1 <= tn * 2 // dim <= 16:
            raise ValueError("row-major BF16 tile row exceeds selected mesh transfer")
        plan = {**plan, "bf16_output_layout": "row_major_bf16"}
    if tilewise_policy and len(output_tiles) == 1:
        raise ValueError("physical MX tilewise VPU policy needs multiple output tiles")
    if len(output_tiles) != 1 and vector_ops:
        # The C tile is reused across output tiles. A pointwise, in-place VPU
        # operation can run after each tile's final K wave and before its
        # readout; an operation spanning tiles or producing another resident
        # tensor needs a separate lifetime planner.
        scalar_bf16 = (vector_ops[0][1]["immediate_bf16"]
                       if len(vector_ops) == 1 and vector_ops[0][0] == "vpu" else None)
        tilewise_kinds = (["adds"] if tilewise_policy ==
                          "bf16_adds_scalar_each_output_tile_v1" else ["muls"] if
                          tilewise_policy != "bf16_scalar_chain_each_output_tile_v1" else
                          [op[1]["kind"] for op in vector_ops])
        tilewise = (tilewise_policy in {"bf16_muls_x2_each_output_tile_v1",
                                       "bf16_muls_scalar_each_output_tile_v1",
                                       "bf16_adds_scalar_each_output_tile_v1",
                                       "bf16_scalar_chain_each_output_tile_v1"} and
                    len(vector_ops) == len(tilewise_kinds) and
                    all(op[0] == "vpu" for op in vector_ops) and
                    output_format == "bf16" and
                    all(op[1] == {
                        "kind": kind, "src1_row": plan["c_spad_dest"],
                        "src2_row": 0, "dst_row": plan["c_spad_dest"],
                        "rows": plan["c_rows"], "reduction_length": 1,
                        "broadcast": False, "immediate_bf16": op[1]["immediate_bf16"],
                        "second_dst_row": None}
                        for op, kind in zip(vector_ops, tilewise_kinds)) and
                    (tilewise_policy != "bf16_muls_x2_each_output_tile_v1" or
                     scalar_bf16 == 0x4000))
        if not tilewise:
            raise ValueError("multi-output MX vector epilogue needs in-place BF16 scalar ops per tile")
        plan = {**plan, "vector_tile_policy": tilewise_policy}
    if host_requant and len(output_tiles) != 1:
        raise ValueError("Radiance header epilogue requires a single BF16 output tile")
    if quant_output and (len(output_tiles) != 1 or vector_ops and not vector_requant):
        raise ValueError("physical quantized output needs one tile without vector epilogues")
    if vector_requant:
        if (precision != "FP8" or
                not vpu_requant_shape_is_legal(shape, tile) or
                len(vector_ops) != 2 or
                vector_ops[0][0] != "vpu" or vector_ops[1][0] != "spad_requant"):
            raise ValueError("physical VPU/SPAD composition requires selected FP8 source shape and ops")
        vpu, requant = vector_ops[0][1], vector_ops[1][1]
        expected_destination = plan["c_spad_dest"] + plan["c_rows"] + 256
        if (vpu != {"kind": "muls", "src1_row": plan["c_spad_dest"],
                    "src2_row": 0, "dst_row": plan["c_spad_dest"],
                    "rows": plan["c_rows"], "reduction_length": 1,
                    "broadcast": False, "immediate_bf16": 0x4000,
                    "second_dst_row": None} or
                requant != {"source_row": plan["c_spad_dest"],
                            "destination_row": expected_destination,
                            "m": m, "n": n, "output_format": "fp8_e4m3",
                            "tiled": True, "resident": True,
                            "scale_dram_address": 0,
                            "scale_buffer": "scratch_output_scales"} or
                expected_destination + m * n // dim > plan["scratchpad_rows"]):
            raise ValueError("physical VPU/SPAD scratchpad lifetime or operation differs")
    values_per_byte = 1 if precision == "FP8" else 2
    pe = dim * values_per_byte
    ti, tj, tki = tm // pe, tn // pe, tk // dim
    a_stride = k
    b_stride = n if precision == "FP8" else n // 2
    if (len(resources["activation"]) != a_stride * (m if precision == "FP8" else m // 2) or
            len(resources["weight"]) != b_stride * k or
            len(resources["golden_bf16"]) != m * n * 2):
        raise ValueError("physical source resources differ from packed tensor shapes")
    quant_name = {"FP8": "nicolas_fp8", "FP4": "nicolas_fp4", "FP6": "nicolas_fp6"}[precision]
    expected_quant_bytes = m * n // (2 if precision in {"FP4", "FP6"} else 1)
    if quant_output and (len(resources["golden_output_scales"]) != m * n // 32 or
                         len(resources[quant_name]) != expected_quant_bytes or
                         len(resources["nicolas_output_scales"]) != m * n // 32):
        raise ValueError("physical source quantized golden differs from output shape")
    if quant_output and precision == "FP6" and len(resources["source_fp6_packed"]) != expected_quant_bytes:
        raise ValueError("physical FP6 source projection differs from output shape")
    if quant_output and precision != "FP6" and len(resources["golden_fp8"]) != m * n:
        raise ValueError("physical FP8/FP4 source golden differs from output shape")
    if host_requant:
        codes, scales = (
            quantize_bf16_radiance_header_fp6(
                resources["golden_bf16"], m, n, resources["output_lut"])
            if host_kind == "radiance_header_fp6" else
            quantize_bf16_radiance_header_fp8(resources["golden_bf16"], m, n))
        if (resources.get("source_fp6_packed" if host_kind == "radiance_header_fp6"
                          else "golden_fp8") != codes or
                resources.get("golden_output_scales") != scales):
            raise ValueError("Radiance source header quantized golden differs from BF16 epilogue")

    steps: list[PhysicalStep] = []

    def issue(phase: str, command: Command | Fence, wave: int | None = None) -> None:
        steps.append(PhysicalStep(phase, wave, command))

    # gemmini_extended3_config_ex(..., WS, formats, BF16 output, LUT).
    fmt = {"FP8": 0, "FP6": 1, "FP4": 2}[precision]
    config_base = (1 << 16) | (fmt << 12) | (fmt << 10) | \
                  (int(precision == "FP6") << 5) | (1 << 2)
    # A requant postpass modifies the accumulator storage. Keep intermediate
    # K waves in BF16 and select the requested output encoding for the final
    # wave only, as the handwritten last_k schedule does.
    config_ex = config_base | ((3 if matrix_quant_output and len(plan["waves"]) > 1
                                else fmt if matrix_quant_output else 3) << 14)
    issue("configure", _cmd(7, 0, 0))
    issue("configure", _cmd(0, config_ex, 1 << 48))
    issue("configure", _config_ld(a_stride, dim=dim))
    issue("configure", _config_ld(b_stride, id=1, dim=dim))
    issue("configure", _config_st(dim if matrix_quant_output else n * 2))
    issue("configure", Fence())

    if precision == "FP6":
        for name, sel in (("weight_lut", 0), ("activation_lut", 1), ("output_lut", 2)):
            if len(resources[name]) != 64 * 3 * 4:
                raise ValueError(f"physical FP6 {name} has the wrong byte count")
            issue("upload_lut", _cmd(29, Operand(buffer=name),
                                     (6 << 34) | (sel << 32) | 64))
        issue("upload_lut", Fence())

    scale_half = profile["resources"]["scale_mem_config"]["size_bytes"] // 4
    if scale_half != 4096:
        raise ValueError("physical MX scale-half geometry is not yet supported")
    def lower_output_tile(output_tile: dict) -> None:
        m_start, n_start = output_tile["m_start"], output_tile["n_start"]
        for wave in plan["waves"]:
            index = wave["index"]
            if matrix_quant_output and index == len(plan["waves"]) - 1 and index > 0:
                issue("configure_final_output", _cmd(0, config_base | (fmt << 14),
                                                       1 << 48), index)
                issue("configure_final_output", Fence(), index)
            group = wave["k_start"] // 32
            groups = tk // 32
            # The pinned Spike LUT compute path ignores selector 1; keep that
            # workaround local to this explicit execution mode.
            selector = 0 if precision == "FP6" and mode == "spike_serial" else index & 1
            dest = selector * scale_half
            for name, row_bytes, pitch, start, sel in (
                    ("activation_scales", tm, m, m_start, 0),
                    ("weight_scales", tn, n, n_start, 1)):
                offset = group * pitch + start
                if (offset + (groups - 1) * pitch + row_bytes > len(resources[name]) or
                        row_bytes * groups > scale_half):
                    raise ValueError("physical MX scale wave exceeds bound resource or target half")
                rs1 = Operand(buffer=name, byte_offset=offset,
                              address_mask=(1 << 40) - 1, or_bits=pitch << 40)
                rs2 = (groups << 46) | (dest << 33) | (sel << 32) | row_bytes
                issue("upload_scales", _cmd(27, rs1, rs2), index)
            issue("upload_scales", Fence(), index)

            a_row = wave["a_spad_start"]
            b_end = wave["b_spad_end"]
            b_start = b_end - tki * tj * dim
            issue("move_activation", _config_ld(a_stride, dim=dim), index)
            for i in range(ti):
                for ki in range(tki):
                    offset = (m_start // values_per_byte + i * dim) * a_stride + \
                             wave["k_start"] + ki * dim
                    if offset + (dim - 1) * a_stride + dim > len(resources["activation"]):
                        raise ValueError("physical MX activation tile exceeds payload")
                    row = a_row + (i * tki + ki) * dim
                    issue("move_activation", _transfer(2, "activation", offset, row,
                                                         dim=dim), index)
            issue("move_weight", _config_ld(b_stride, dim=dim), index)
            for ki in range(tki):
                for j in range(tj):
                    offset = (wave["k_start"] + ki * dim) * b_stride + \
                             n_start // values_per_byte + j * dim
                    if offset + (dim - 1) * b_stride + dim > len(resources["weight"]):
                        raise ValueError("physical MX weight tile exceeds payload")
                    row = b_start + (ki * tj + j) * dim
                    issue("move_weight", _transfer(2, "weight", offset, row,
                                                     dim=dim), index)
            issue("move_weight", Fence(), index)

            selector_bits = (selector << 60) | (selector << 61)
            selector_bits |= (tki << 51) | (tj << 42) | (ti << 33)
            issue("select_scales", _cmd(26, Operand(buffer="scratch_output_scales",
                                                    address_mask=(1 << 33) - 1,
                                                    or_bits=selector_bits), 1), index)
            issue("compute", _cmd(9, 0, (tki << 32) | (tj << 16) | ti), index)
            issue("compute", _cmd(24, a_row, b_end), index)
            skips = 0x38 if wave["move_acc_to_spad"] else 0xb8
            issue("compute", _cmd(8, int(wave["accumulate"]),
                                  (plan["c_spad_dest"] << 32) | 0x200 | skips), index)
            issue("compute", Fence(), index)

        for kind, kwargs in vector_ops:
            if kind == "vpu":
                issue("vpu", vpu_command(profile, **kwargs))
            else:
                issue("spad_requant", spad_requant_command(profile, **kwargs))
        if vector_ops:
            issue("vpu_sync", Fence())

        issue("readout", _config_st(dim))
        readout_rows = m * n // dim if vector_requant else plan["c_rows"]
        readout_source = (vector_ops[1][1]["destination_row"] if vector_requant
                          else plan["c_spad_dest"])
        if plan.get("bf16_output_layout") == "row_major_bf16":
            rows_per_logical = tn * 2 // dim
            if readout_rows != tm * rows_per_logical:
                raise ValueError("row-major BF16 readout disagrees with C scratchpad rows")
            for local_row in range(tm):
                offset = ((m_start + local_row) * n + n_start) * 2
                if offset + tn * 2 > m * n * 2:
                    raise ValueError("row-major BF16 readout exceeds output shape")
                issue("readout", _transfer_rect(
                    3, "output_bf16", offset,
                    readout_source + local_row * rows_per_logical,
                    rows=rows_per_logical, cols=dim))
        else:
            output_tile_base = output_tile["index"] * (
                tm * tn // (2 if precision in {"FP4", "FP6"} else 1)
                if quant_output else tm * tn * 2)
            for row in range(0, readout_rows, dim):
                offset = output_tile_base + row * dim
                if offset + dim * dim > (expected_quant_bytes if quant_output else m * n * 2):
                    raise ValueError("physical MX readout exceeds output shape")
                issue("readout", _transfer(
                    3, "output_quantized" if quant_output else "output_bf16",
                    offset, readout_source + row, dim=dim))
        issue("readout", Fence())

    for output_tile in output_tiles:
        lower_output_tile(output_tile)

    source_golden_preserving = (not vector_ops and not quant_output and
                                manifest.get("origin") !=
                                "radiance_source_target_mesh_reference")
    derived_expected_bf16 = None
    derived_vpu_scalar_bf16 = None
    derived_vpu_scalar_chain = None
    if tilewise_policy == "bf16_scalar_chain_each_output_tile_v1":
        derived_vpu_scalar_chain = tuple(
            (op[1]["kind"], op[1]["immediate_bf16"]) for op in vector_ops)
        derived_expected_bf16 = resources["golden_bf16"]
        for kind, scalar in derived_vpu_scalar_chain:
            derived_expected_bf16 = (
                bf16_mul_scalar if kind == "muls" else bf16_add_scalar)(
                    derived_expected_bf16, scalar)
        source_golden_preserving = False
    if len(vector_ops) == 1 and vector_ops[0][0] == "vpu":
        vpu = vector_ops[0][1]
        in_place_full_output = (
            vpu["src1_row"] == plan["c_spad_dest"] and
            vpu["dst_row"] == plan["c_spad_dest"] and
            vpu["rows"] == plan["c_rows"] and vpu["src2_row"] == 0 and
            vpu["reduction_length"] == 1 and not vpu["broadcast"] and
            vpu["second_dst_row"] is None)
        source_golden_preserving = (
            in_place_full_output and vpu["kind"] == "adds" and
            vpu["immediate_bf16"] == 0 and
            all((word == 0 or 0 < (word & 0x7f80) < 0x7f80)
                for word in (int.from_bytes(resources["golden_bf16"][i:i + 2], "little")
                             for i in range(0, len(resources["golden_bf16"]), 2))))
        if in_place_full_output and vpu["kind"] == "muls" and vpu["immediate_bf16"] == 0x4000:
            derived_expected_bf16 = _exact_bf16_x2(resources["golden_bf16"])
        if in_place_full_output and tilewise_policy == "bf16_muls_scalar_each_output_tile_v1":
            derived_vpu_scalar_bf16 = vpu["immediate_bf16"]
            derived_expected_bf16 = bf16_mul_scalar(
                resources["golden_bf16"], derived_vpu_scalar_bf16)
        if in_place_full_output and tilewise_policy == "bf16_adds_scalar_each_output_tile_v1":
            derived_vpu_scalar_bf16 = vpu["immediate_bf16"]
            derived_expected_bf16 = bf16_add_scalar(
                resources["golden_bf16"], derived_vpu_scalar_bf16)

    return PhysicalProgram(profile_sha256(profile), manifest_sha256(manifest),
                           mode, shape, plan, tuple(steps), source_golden_preserving,
                           derived_expected_bf16, output_format, vector_requant,
                           derived_vpu_scalar_bf16, derived_vpu_scalar_chain,
                           golden_origin=("target_mesh_reference" if manifest.get("origin") ==
                                          "radiance_source_target_mesh_reference" else
                                          "source_header"))
