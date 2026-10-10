"""Typed source-bound two-tile MX+VPU graph and physical command lowering."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .command_ir import Command, Fence, spad_requant_command, vpu_command
from .physical_program import _cmd, _config_ld, _config_st, _transfer
from .resident_lowering import lower_resident_contract
from .target_profile import profile_sha256
from .verify_profile_ir import _bool_attr, _int_attr, _operation_name, _text_attr, verify_ir


INPUTS = ("c1_bf16", "b2_weight", "b2_scales")
OUTPUTS = tuple(f"{name}_{tile}" for tile in (0, 1)
                for name in ("c1_scales", "c1_tiled", "c2_scales", "c2_tiled"))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(resources: dict[str, bytes]) -> str:
    return _sha(json.dumps({name: _sha(resources[name]) for name in INPUTS},
                           sort_keys=True, separators=(",", ":")).encode())


def _module(text: str):
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    return Parser(context, text).parse_module()


def validate_original_branch_trace(trace: dict) -> None:
    """Require two scalar branches from one MM1 and one shared B2 input."""
    try:
        nodes = trace["graphs"]["original"]["nodes"]
        targets = [node["target"] for node in nodes]
        edges = [[arg.get("node_id") if isinstance(arg, dict) else arg
                  for arg in node["args"]] for node in nodes]
        ids = [node["id"] for node in nodes]
    except (KeyError, TypeError) as error:
        raise ValueError("model2MLIR original graph trace is incomplete") from error
    if (targets != ["a", "b1", "b2", "aten.matmul.default",
                    "aten.mul.Tensor", "aten.matmul.default",
                    "aten.mul.Tensor", "aten.matmul.default", "output"] or
            edges[3] != ids[:2] or edges[4] != [ids[3], 2.0] or
            edges[5] != [ids[4], ids[2]] or edges[6] != [ids[3], 4.0] or
            edges[7] != [ids[6], ids[2]] or edges[8] != [[
                {"node_id": ids[5], "value_id": f"{ids[5]}:v0"},
                {"node_id": ids[7], "value_id": f"{ids[7]}:v0"}]]):
        raise ValueError("model2MLIR original graph lost the shared two-branch chain")


def render_chain_pipelined(frontend: str, trace: dict, manifest: dict,
                           profile: dict, resources: dict[str, bytes],
                           facts: dict) -> str:
    """Bind the captured branches to Nicolas's preloaded BF16 source data."""
    validate_original_branch_trace(trace)
    report = verify_ir(frontend, profile)
    if (report["contracts"], report["vpu_commands"],
            report["spad_requants"], report["resident_contracts"]) != (3, 0, 0, 0):
        raise ValueError("two-tile source binding needs three captured FP8 sites")
    expected_sites = [f"functional:matmul{'' if i == 0 else '_' + str(i)}"
                      for i in range(3)]
    sites = manifest.get("sites", [])
    if [(site.get("site_id"), site.get("status"), site.get("format"),
         site.get("shape")) for site in sites] != [
             (site, "quantized", "mxfp8", [64, 64, 64]) for site in expected_sites]:
        raise ValueError("two-tile source binding capture sites differ")
    module = _module(frontend)
    if (_text_attr(module, "mx.profile_sha256") != profile_sha256(profile) or
            _text_attr(module, "prov.quantization_manifest_sha256") != _sha(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode())):
        raise ValueError("two-tile source binding profile or manifest differs")
    contracts = [op for op in module.walk()
                 if _operation_name(op) == "mx_gemmini.contract"]
    if ([_text_attr(op, "site_id") for op in contracts] != expected_sites or
            any((_text_attr(op, "activation_format"),
                 _text_attr(op, "weight_format"),
                 _text_attr(op, "activation_projection"),
                 _text_attr(op, "weight_projection"), _int_attr(op, "pe_mode")) !=
                ("fp8_e4m3", "fp8_e4m3", "direct", "direct", 8)
                for op in contracts)):
        raise ValueError("two-tile source binding precision differs")
    if (facts.get("profile_sha256") != profile_sha256(profile) or
            facts.get("tile_factors_bf16") != [0x4000, 0x4080] or
            facts.get("bf16_rows") != [0x1000, 0x1200] or
            facts.get("c1_rows") != [128, 1024] or
            facts.get("c2_rows") != [512, 1536] or
            any(facts.get("resource_sha256", {}).get(name) != _sha(data)
                for name, data in resources.items()) or
            {name: len(resources.get(name, b"")) for name in INPUTS} !=
            {"c1_bf16": 8192, "b2_weight": 4096, "b2_scales": 128}):
        raise ValueError("two-tile source placement or resources differ")
    source_digest = _sha(json.dumps(facts, sort_keys=True,
                                    separators=(",", ":")).encode())
    policy_digest = _sha(b"bf16_exact_x2_x4;fp8_e4m3_po2_rne;resident_two_tile")
    digest = profile_sha256(profile)
    attrs = (f'contract_sha256 = "{source_digest}", '
             f'policy_sha256 = "{policy_digest}", '
             f'manifest_sha256 = "{facts["header_sha256"]}", '
             f'profile_sha256 = "{digest}"')

    def branch(tile: int) -> str:
        row = (0x1000, 0x1200)[tile]
        a = (128, 1024)[tile]
        c = (512, 1536)[tile]
        scalar = (0x4000, 0x4080)[tile]
        site = expected_sites[tile + 1]
        b_row = profile["resources"]["scratchpad_bytes"] // 16 - 256
        return f'''    %v{tile} = "mx_gemmini.vpu_execute"(%bf16) {{
      site_id = "{site}", kind = "muls", src1_row = {row} : i32,
      src2_row = 0 : i32, dst_row = {row} : i32, rows = 512 : i32,
      reduction_length = 1 : i32, broadcast = false,
      immediate_bf16 = {scalar} : i32, {attrs}}}
      : (tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %c1_{tile}, %c1s_{tile} = "mx_gemmini.spad_requant"(%v{tile}) {{
      site_id = "{site}", source_row = {row} : i32,
      destination_row = {a} : i32, m = 64 : i32, n = 64 : i32,
      output_format = "fp8_e4m3", tiled = true, resident = true,
      scale_dram_address = 0 : i64, scale_buffer = "c1_scales_{tile}", {attrs}}}
      : (tensor<64x64xbf16>) -> (tensor<64x64xi8>, tensor<64x2xi8>)
    %c2_{tile}, %c2s_{tile} = "mx_gemmini.resident_contract"(
      %c1_{tile}, %c1s_{tile}, %b2, %b2s) {{
      site_id = "{site}", activation_row = {a} : i32,
      weight_row = {b_row} : i32, output_row = {c} : i32,
      m = 64 : i32, n = 64 : i32, k = 64 : i32,
      activation_format = "fp8_e4m3", weight_format = "fp8_e4m3",
      output_format = "fp8_e4m3", weight_buffer = "b2_weight",
      weight_scales_buffer = "b2_scales",
      output_scales_buffer = "c2_scales_{tile}", {attrs}}}
      : (tensor<64x64xi8>, tensor<64x2xi8>, tensor<64x64xi8>, tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>)
'''

    graph = f'''module attributes {{mx.profile_sha256 = "{digest}",
  mx.contract_sha256 = "{source_digest}", mx.policy_sha256 = "{policy_digest}",
  prov.quantization_manifest_sha256 = "{facts["header_sha256"]}",
  mx.frontend_mlir_sha256 = "{_sha(frontend.encode())}",
  mx.original_graph_sha256 = "{_sha(json.dumps(trace["graphs"]["original"], sort_keys=True, separators=(",", ":")).encode())}",
  mx.runtime_resources_sha256 = "{_digest(resources)}"}} {{
  func.func @nicolas_chain_pipelined(
      %bf16: tensor<64x64xbf16>, %b2: tensor<64x64xi8>,
      %b2s: tensor<2x64xi8>)
      -> (tensor<64x64xi8>, tensor<64x2xi8>,
          tensor<64x64xi8>, tensor<64x2xi8>) {{
{branch(0)}{branch(1)}    func.return %c2_0, %c2s_0, %c2_1, %c2s_1
      : tensor<64x64xi8>, tensor<64x2xi8>, tensor<64x64xi8>, tensor<64x2xi8>
  }}
}}
'''
    lower_chain_pipelined(graph, profile, resources)
    return graph


@dataclass(frozen=True)
class TwoTileChain:
    commands: tuple[Command | Fence, ...]
    sites: tuple[str, str]


def lower_chain_pipelined(mlir_text: str, profile: dict,
                          resources: dict[str, bytes], *,
                          issue_schedule: str = "program_order_with_dependency_fences"
                          ) -> TwoTileChain:
    """Issue the typed branches with B2 loaded once."""
    from xdsl.dialects.func import FuncOp, ReturnOp

    report = verify_ir(mlir_text, profile)
    if (report["contracts"], report["vpu_commands"],
            report["spad_requants"], report["resident_contracts"]) != (0, 2, 2, 2):
        raise ValueError("two-tile chain requires two VPU/requant/MM2 branches")
    module = _module(mlir_text)
    if (_text_attr(module, "mx.runtime_resources_sha256") != _digest(resources) or
            any(len(resources.get(name, b"")) != size for name, size in
                (("c1_bf16", 8192), ("b2_weight", 4096), ("b2_scales", 128)))):
        raise ValueError("two-tile chain runtime resources differ")
    funcs = [op for op in module.walk() if isinstance(op, FuncOp)]
    if len(funcs) != 1:
        raise ValueError("two-tile chain needs one function")
    func = funcs[0]
    args, ops = list(func.body.block.args), list(func.body.block.ops)
    if (tuple(str(arg.type) for arg in args) !=
            ("tensor<64x64xbf16>", "tensor<64x64xi8>", "tensor<2x64xi8>") or
            [_operation_name(op) for op in ops] != [
                "mx_gemmini.vpu_execute", "mx_gemmini.spad_requant",
                "mx_gemmini.resident_contract"] * 2 + ["func.return"]):
        raise ValueError("two-tile chain SSA operation order differs")
    sites: list[str] = []
    vectors: list[Command] = []
    requants: list[Command] = []
    resident: list[tuple[Command | Fence, ...]] = []
    for tile in (0, 1):
        vpu, requant, mm2 = ops[tile * 3:tile * 3 + 3]
        site = f"functional:matmul_{tile + 1}"
        bf16_row, c1_row, c2_row, scalar = (
            (0x1000, 128, 512, 0x4000) if tile == 0 else
            (0x1200, 1024, 1536, 0x4080))
        if ([_text_attr(op, "site_id") for op in (vpu, requant, mm2)] !=
                [site] * 3 or list(vpu.operands) != [args[0]] or
                list(requant.operands) != list(vpu.results) or
                list(mm2.operands) != [*requant.results, *args[1:]] or
                tuple(str(result.type) for result in vpu.results) !=
                ("tensor<64x64xbf16>",) or
                tuple(str(result.type) for result in requant.results) !=
                ("tensor<64x64xi8>", "tensor<64x2xi8>") or
                tuple(str(result.type) for result in mm2.results) !=
                ("tensor<64x64xi8>", "tensor<64x2xi8>") or
                _text_attr(vpu, "kind") != "muls" or
                [_int_attr(vpu, key) for key in
                 ("src1_row", "src2_row", "dst_row", "rows",
                  "reduction_length", "immediate_bf16")] !=
                [bf16_row, 0, bf16_row, 512, 1, scalar] or
                _bool_attr(vpu, "broadcast") or
                [_int_attr(requant, key) for key in
                 ("source_row", "destination_row", "m", "n",
                  "scale_dram_address")] != [bf16_row, c1_row, 64, 64, 0] or
                _text_attr(requant, "scale_buffer") != f"c1_scales_{tile}" or
                _text_attr(requant, "output_format") != "fp8_e4m3" or
                not _bool_attr(requant, "tiled") or
                not _bool_attr(requant, "resident")):
            raise ValueError("two-tile chain branch SSA or vector placement differs")
        attrs = {key: _int_attr(mm2, key) for key in
                 ("activation_row", "weight_row", "output_row", "m", "n", "k")}
        attrs.update({key: _text_attr(mm2, key) for key in
                      ("activation_format", "weight_format", "output_format",
                       "weight_buffer", "weight_scales_buffer", "output_scales_buffer")})
        if (attrs["activation_row"] != c1_row or
                attrs["weight_row"] != profile["resources"]["scratchpad_bytes"] // 16 - 256 or
                attrs["output_row"] != c2_row or
                (attrs["m"], attrs["n"], attrs["k"]) != (64, 64, 64) or
                any(attrs[key] != "fp8_e4m3" for key in
                    ("activation_format", "weight_format", "output_format")) or
                (attrs["weight_buffer"], attrs["weight_scales_buffer"],
                 attrs["output_scales_buffer"]) !=
                ("b2_weight", "b2_scales", f"c2_scales_{tile}")):
            raise ValueError("two-tile resident MM2 placement or shared B2 differs")
        sites.append(site)
        vectors.append(vpu_command(
            profile, kind="muls", src1_row=bf16_row, src2_row=0,
            dst_row=bf16_row, rows=512, immediate_bf16=scalar))
        requants.append(spad_requant_command(
            profile, source_row=bf16_row, destination_row=c1_row,
            m=64, n=64, output_format="fp8_e4m3", tiled=True,
            resident=True, scale_dram_address=0,
            scale_buffer=f"c1_scales_{tile}"))
        resident.append(lower_resident_contract(profile, attrs))
    ret = ops[-1]
    if not isinstance(ret, ReturnOp) or list(ret.operands) != [
            *ops[2].results, *ops[5].results]:
        raise ValueError("two-tile chain output SSA edges differ")
    # Reuse the qualified resident lowerer. Only its B2 prelude is shared;
    # each branch retains its own resident output-scale and LOOP commands.
    split = next(i for i, command in enumerate(resident[0])
                 if isinstance(command, Command) and command.funct == 0 and
                 command.rs1.immediate == 2 and command.rs2.immediate == 2)
    if resident[0][:split] != resident[1][:split]:
        raise ValueError("two-tile resident branches cannot share B2 prelude")
    commands: list[Command | Fence] = [_cmd(7, 0, 0), *resident[0][:split]]
    for tile in (0, 1):
        bf16_row = (0x1000, 0x1200)[tile]
        commands += [_config_ld(16), Fence()]
        for row in range(0, 512, 16):
            commands.append(_transfer(2, "c1_bf16", row * 16, bf16_row + row))
        commands += [Fence(), vectors[tile], Fence(), requants[tile], Fence(),
                     *resident[tile][split:]]
    commands.append(_config_st(16))
    for tile, starts in enumerate(((128, 512), (1024, 1536))):
        for name, start in zip((f"c1_tiled_{tile}", f"c2_tiled_{tile}"), starts):
            for row in range(0, 256, 16):
                commands.append(_transfer(3, name, row * 16, start + row))
    commands.append(Fence())
    serial = TwoTileChain(tuple(commands), tuple(sites))
    if issue_schedule == "program_order_with_dependency_fences":
        return serial
    if issue_schedule != "pipelined":
        raise ValueError("unknown two-tile MX issue schedule")
    return pipeline_two_tile_commands(serial, reload_buffer="c1_bf16")


def pipeline_two_tile_commands(serial: TwoTileChain, *,
                               reload_buffer: str) -> TwoTileChain:
    """Issue both BF16 transfers before VPU and overlap the two branches."""
    commands = serial.commands

    def indices(funct: int) -> list[int]:
        return [i for i, item in enumerate(commands)
                if isinstance(item, Command) and item.funct == funct]

    vpu, quant, loops = indices(33), indices(34), indices(8)
    if len(vpu) != 2 or len(quant) != 2 or len(loops) not in (2, 3) or not (
            vpu[0] < quant[0] < loops[-2] < vpu[1] < quant[1] < loops[-1]):
        raise ValueError("two-tile serial stages cannot form source pipeline")
    first_reload = next((i for i in range(loops[-2] + 1, vpu[1])
                         if isinstance(commands[i], Command) and
                         commands[i].funct == 2 and
                         commands[i].rs1.buffer == reload_buffer), None)
    if first_reload is None:
        raise ValueError("second tile lacks BF16 reload")
    reload_start = max((i for i in range(loops[-2] + 1, first_reload)
                        if isinstance(commands[i], Command) and
                        commands[i].funct == 0 and
                        commands[i].rs1.immediate == (16 << 16) | (1 << 8) | 1 and
                        commands[i].rs2.immediate == 16), default=-1)
    first_output = next((i for i in range(loops[-1] + 1, len(commands))
                         if isinstance(commands[i], Command) and
                         commands[i].funct == 3 and
                         commands[i].rs1.buffer == "c1_tiled_0"), None)
    if (reload_start < 0 or first_reload - reload_start != 2 or
            first_output is None or first_output < 1 or
            not isinstance(commands[first_output - 1], Command) or
            commands[first_output - 1].funct != 0 or
            sum(isinstance(item, Command) and item.rs1.buffer == reload_buffer
                for item in commands[reload_start:vpu[1]]) != 32):
        raise ValueError("two-tile pipeline preload or readout boundary differs")
    def without_fences(values: tuple[Command | Fence, ...]) -> list[Command]:
        return [item for item in values if isinstance(item, Command)]

    prefix: tuple[Command | Fence, ...] = commands[:vpu[0]]
    if reload_buffer == "c1_bf16":
        first_load = next((i for i, item in enumerate(prefix)
                           if isinstance(item, Command) and item.funct == 2 and
                           item.rs1.buffer == reload_buffer), None)
        first_config = max((i for i in range(first_load or 0)
                            if isinstance(prefix[i], Command) and
                            prefix[i].funct == 0 and
                            prefix[i].rs1.immediate == (16 << 16) | (1 << 8) | 1 and
                            prefix[i].rs2.immediate == 16), default=-1)
        if first_load is None or first_config < 0:
            raise ValueError("first tile lacks BF16 source load")
        prefix = (*prefix[:first_config], *without_fences(prefix[first_config:]))

    pipelined = (
        *prefix,
        *without_fences(commands[reload_start:vpu[1]]),
        commands[vpu[0]], commands[quant[0]], commands[vpu[1]],
        *without_fences(commands[quant[0] + 1:loops[-2] + 1]),
        commands[quant[1]],
        *without_fences(commands[quant[1] + 1:loops[-1] + 1]),
        Fence(), *commands[first_output - 1:],
    )
    if ([item.funct for item in pipelined if isinstance(item, Command)
         and item.funct in (33, 34, 8)][-6:] != [33, 34, 33, 8, 34, 8]):
        raise ValueError("two-tile pipeline command order differs from source")
    return TwoTileChain(pipelined, serial.sites)
