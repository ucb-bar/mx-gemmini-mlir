"""Lower one payload-bound MX contraction to a physical Rocket command stream.

The stream is source-independent after payload export. The Spike serial mode
uses explicit operand DMA because the pinned Spike model does not execute the
source loop-FSM DMA-only launches. FP6 also reloads scale half zero because
the pinned model ignores the alternating selector in LUT compute. The RTL
mode keeps alternating halves and is structurally checked, not yet qualified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .command_ir import Command, Fence, Operand
from .source_gemm import plan_mx_gemm
from .source_payload import manifest_sha256
from .target_profile import profile_sha256
from .verify_profile_ir import _operation_name, _text_attr, verify_ir


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

    def receipt(self) -> dict:
        return {"schema": "mx_gemmini.physical_program.v1",
                "profile_sha256": self.profile_sha256,
                "payload_manifest_sha256": self.payload_manifest_sha256,
                "mode": self.mode, "shape_mnk": list(self.shape),
                "plan": self.plan,
                "steps": [asdict(step) for step in self.steps]}


def _imm(value: int) -> Operand:
    return Operand(immediate=value)


def _cmd(funct: int, rs1: int | Operand, rs2: int | Operand) -> Command:
    return Command(funct, _imm(rs1) if isinstance(rs1, int) else rs1,
                   _imm(rs2) if isinstance(rs2, int) else rs2)


def _config_ld(stride: int, *, id: int = 0) -> Command:
    # Nicolas gemmini.h gemmini_extended5_config_ld, scale identity = 0.
    return _cmd(0, (16 << 16) | (1 << 8) | (id << 3) | 1, stride)


def _config_st(stride: int) -> Command:
    return _cmd(0, 2, stride)


def _transfer(funct: int, buffer: str, offset: int, row: int) -> Command:
    return _cmd(funct, Operand(buffer=buffer, byte_offset=offset),
                (16 << 48) | (16 << 32) | row)


def _check_binding(mlir_text: str, profile: dict, manifest: dict) -> None:
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    checked = verify_ir(mlir_text, profile)
    if (checked["contracts"], checked["encodes"], checked["requantizes"],
            checked["vpu_commands"], checked["spad_requants"]) != (1, 0, 0, 0, 0):
        raise ValueError("physical MX source lowering needs exactly one BF16 contraction")
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    digest = manifest_sha256(manifest)
    if (_text_attr(module, "mx.payload_manifest_sha256") != digest or
            manifest.get("profile_sha256") != profile_sha256(profile)):
        raise ValueError("physical MX lowering payload or target digest differs")
    ops = [op for op in module.walk() if _operation_name(op).startswith("mx_gemmini.")]
    contract = [op for op in ops if _operation_name(op) == "mx_gemmini.contract"]
    readout = [op for op in ops if _operation_name(op) == "mx_gemmini.readout_bf16"]
    if len(ops) != 2 or len(contract) != 1 or len(readout) != 1:
        raise ValueError("physical MX lowering requires a matching BF16 readout")
    if (manifest.get("site_id") != _text_attr(contract[0], "site_id") or
            manifest.get("site_id") != _text_attr(readout[0], "site_id") or
            _text_attr(contract[0], "payload_manifest_sha256") != digest):
        raise ValueError("physical MX lowering site or payload binding differs")


def lower_bound_source(mlir_text: str, profile: dict, manifest: dict,
                       resources: Mapping[str, bytes], *,
                       mode: str = "spike_serial") -> PhysicalProgram:
    """Lower a checked payload-bound contraction into ordered RoCC commands."""
    _check_binding(mlir_text, profile, manifest)
    if mode not in {"spike_serial", "rtl_alternating"}:
        raise ValueError("unknown MX physical scheduling mode")
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("physical source lowering currently requires Rocket RoCC")
    precision = manifest["precision"]
    shape = tuple(manifest["shape_mnk"])
    tile = tuple(manifest["tile_mnk"])
    plan = plan_mx_gemm(shape=shape, tile=tile, datatype=precision,
                        quant_output=False, acc_to_gmem=False,
                        scratchpad_bytes=profile["resources"]["scratchpad_bytes"],
                        profile=profile)
    m, n, k = shape
    tm, tn, tk = tile
    dim = profile["geometry"]["mesh_columns"]
    if dim != 16 or tm != m or tn != n or tk % 16 or plan["c_rows"] % 16:
        raise ValueError("physical source lowering needs one DIM16 output tile")
    pe = 16 if precision == "FP8" else 32
    ti, tj, tki = tm // pe, tn // pe, tk // 16
    a_stride = k
    b_stride = n if precision == "FP8" else n // 2
    if (len(resources["activation"]) != a_stride * (m if precision == "FP8" else m // 2) or
            len(resources["weight"]) != b_stride * k or
            len(resources["golden_bf16"]) != m * n * 2):
        raise ValueError("physical source resources differ from packed tensor shapes")

    steps: list[PhysicalStep] = []

    def issue(phase: str, command: Command | Fence, wave: int | None = None) -> None:
        steps.append(PhysicalStep(phase, wave, command))

    # gemmini_extended3_config_ex(..., WS, formats, BF16 output, LUT).
    fmt = {"FP8": 0, "FP6": 1, "FP4": 2}[precision]
    config_ex = (1 << 16) | (3 << 14) | (fmt << 12) | (fmt << 10) | \
                (int(precision == "FP6") << 5) | (1 << 2)
    issue("configure", _cmd(7, 0, 0))
    issue("configure", _cmd(0, config_ex, 1 << 48))
    issue("configure", _config_ld(a_stride))
    issue("configure", _config_ld(b_stride, id=1))
    issue("configure", _config_st(n * 2))
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
    for wave in plan["waves"]:
        index = wave["index"]
        group = wave["k_start"] // 32
        groups = tk // 32
        # The pinned Spike LUT compute path ignores selector 1; keep that
        # workaround local to this explicit execution mode.
        selector = 0 if precision == "FP6" and mode == "spike_serial" else index & 1
        dest = selector * scale_half
        for name, width, sel in (("activation_scales", m, 0),
                                 ("weight_scales", n, 1)):
            offset = group * width
            if offset + groups * width > len(resources[name]) or width * groups > scale_half:
                raise ValueError("physical MX scale wave exceeds bound resource or target half")
            rs1 = Operand(buffer=name, byte_offset=offset,
                          address_mask=(1 << 40) - 1, or_bits=width << 40)
            rs2 = (groups << 46) | (dest << 33) | (sel << 32) | width
            issue("upload_scales", _cmd(27, rs1, rs2), index)
        issue("upload_scales", Fence(), index)

        a_row = wave["a_spad_start"]
        b_end = wave["b_spad_end"]
        b_start = b_end - tki * tj * dim
        issue("move_activation", _config_ld(a_stride), index)
        for i in range(ti):
            for ki in range(tki):
                offset = (i * dim) * a_stride + wave["k_start"] + ki * dim
                if offset + 15 * a_stride + dim > len(resources["activation"]):
                    raise ValueError("physical MX activation tile exceeds payload")
                row = a_row + (i * tki + ki) * dim
                issue("move_activation", _transfer(2, "activation", offset, row), index)
        issue("move_weight", _config_ld(b_stride), index)
        for ki in range(tki):
            for j in range(tj):
                offset = (wave["k_start"] + ki * dim) * b_stride + j * dim
                if offset + 15 * b_stride + dim > len(resources["weight"]):
                    raise ValueError("physical MX weight tile exceeds payload")
                row = b_start + (ki * tj + j) * dim
                issue("move_weight", _transfer(2, "weight", offset, row), index)
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

    issue("readout", _config_st(dim))
    for row in range(0, plan["c_rows"], dim):
        offset = row * dim
        if offset + dim * dim > len(resources["golden_bf16"]):
            raise ValueError("physical MX BF16 readout exceeds output shape")
        issue("readout", _transfer(3, "output_bf16", offset,
                                   plan["c_spad_dest"] + row))
    issue("readout", Fence())
    return PhysicalProgram(profile_sha256(profile), manifest_sha256(manifest),
                           mode, shape, plan, tuple(steps))
