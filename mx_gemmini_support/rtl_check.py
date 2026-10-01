"""Source-scoped RTL cross-check for the selected MX software contract.

This checks the exact source files read and derives the small format table from
MxRequantizer. It is not elaboration, simulation, or numerical qualification.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import yaml

from .contract import compile_contract
from .candidate_protocol import RTL_COMMIT as CANDIDATE_RTL_COMMIT
from .config_facts import check_selected_config, selected_config_facts


def _head(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _table(text: str, start: str, stop: str) -> dict[str, int]:
    block = text.split(start, 1)[1].split(stop, 1)[0]
    result = {}
    for line in block.splitlines():
        if "->" not in line:
            continue
        left, right = line.strip().rstrip(",").split("->", 1)
        name = left.strip()
        value = right.strip().split(".U", 1)[0]
        if name in {"FP8", "FP6", "FP4", "BF16"}:
            result[name] = int(value)
    if set(result) != {"FP8", "FP6", "FP4", "BF16"}:
        raise ValueError(f"incomplete RTL format table after {start}")
    return result


def _check_candidate_protocol(root: Path, spec_bytes: bytes) -> dict:
    """Cross-check the candidate's declared command fields against source slices."""
    fields = yaml.safe_load(spec_bytes)["transfer_contracts"]["scale_load"]["semantics"]
    controller = (root / "src/main/scala/gemmini/Controller.scala").read_text()
    assignments = re.findall(
        r"scale_loader_start\.get\.bits\.(\w+)\s*:=\s*"
        r"unrolled_cmd\.bits\.cmd\.(rs[12])\((\d+)(?:,\s*(\d+))?\)",
        controller,
    )
    actual = {
        name: f"{register}[{high}:{low}]" if low else f"{register}[{high}]"
        for name, register, high, low in assignments
    }
    declared = {
        "addr": fields["source_address_bits"],
        "pitch": fields["source_row_pitch_bits"],
        "len": fields["row_bytes_bits"],
        "sel": fields["operand_select_bit"],
        "dest": fields["destination_byte_offset_bits"],
        "rows": fields["row_count_bits"],
        "gated": fields["gated_load_bit"],
    }
    if len(assignments) != len(declared) or actual != declared:
        raise ValueError("candidate scale-load fields differ from RTL command slices")

    isa = (root / "src/main/scala/gemmini/GemminiISA.scala").read_text()
    functs = {
        name: int(value)
        for name, value in re.findall(r"^\s*val\s+(\w+)\s*=\s*(\d+)\.U\b", isa, re.MULTILINE)
    }
    selected_functs = {
        "MX_LOAD_SCALES": fields["upload_funct"],
        "LOOP_WS_CONFIG_SCALES": fields["loop_scale_address_funct"],
        "LOOP_WS_CONFIG_SCALE_STRIDES": fields["loop_scale_stride_funct"],
    }
    if any(functs.get(name) != value for name, value in selected_functs.items()):
        raise ValueError("candidate scale-load functs differ from RTL")

    execute = (root / "src/main/scala/gemmini/ExecuteController.scala").read_text()
    controls = {
        "config_scale_mem_wait_landed_bit": 16,
        "config_scale_mem_wait_managed_ready_bit": 17,
        "config_scale_mem_activation_reuse_bit": 18,
    }
    for key, bit in controls.items():
        if fields.get(key) != f"rs2[{bit}]" or f"rs2s(0)({bit})" not in execute:
            raise ValueError(f"candidate {key} differs from RTL")
    loop = (root / "src/main/scala/gemmini/LoopMatmul.scala").read_text()
    if (not fields.get("row_count_zero_means_one") or
        not fields.get("row_pitch_zero_means_contiguous") or
        not fields.get("loop_scale_address_zero_disables_managed_path") or
        "rows_left  := Mux(sq.rows === 0.U, 0.U, sq.rows - 1.U)" not in controller or
        "row_step   := Mux(sq.pitch === 0.U, len8, sq.pitch)" not in controller or
        "val len8 = Cat(sq.len(31, 3), 0.U(3.W))" not in controller or
        "when (lrl.a_scale_addr === 0.U)" not in loop):
        raise ValueError("candidate scale-load zero and truncation rules differ from RTL")
    if (fields.get("loop_managed_formats") != ["mxfp8"] or
        fields.get("explicit_scale_load_required_formats") != ["mxfp6", "mxfp4"] or
        "val lds_row_bytes = Mux(lds_is_b, lds_loop.max_j, lds_loop.max_i) * block_size.U" not in loop):
        raise ValueError("candidate loop-managed scale policy differs from RTL diagnostic")
    return {"command_fields": actual, "functs": selected_functs,
            "config_scale_mem_bits": {key: f"rs2[{bit}]" for key, bit in controls.items()}}


def check_sources(rtl_root: str | Path, spec_bytes: bytes, source_record: bytes) -> dict:
    root = Path(rtl_root).resolve()
    record = yaml.safe_load(source_record)
    contract = compile_contract(spec_bytes)
    if record.get("schema") != "mx_gemmini.rtl_sources.v1":
        raise ValueError("invalid RTL source record")
    if _head(root) != record["gemmini_commit"] or _head(root / "mxgen") != record["mxgen_commit"]:
        raise ValueError("selected Gemmini or MxGen revision differs from source record")
    if contract["rtl_commit"] != record["gemmini_commit"] or contract["mxgen_commit"] != record["mxgen_commit"]:
        raise ValueError("software contract revision differs from RTL source record")
    if (contract["rtl_config"] != "GemminiMxFPConfigs.standaloneMxFPConfig" or
        contract["rtl_config_class"] != "GemminiMxFPStandaloneConfig"):
        raise ValueError("software contract selects a different RTL configuration")
    for member, expected in record["files"].items():
        path = (root / member).resolve()
        if not path.is_relative_to(root):
            raise ValueError("RTL source record escapes checkout")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"RTL source changed: {member}")
    source = (root / "src/main/scala/gemmini/MxRequantizer.scala").read_text()
    codes = {}
    for line in source.splitlines():
        stripped = line.strip()
        for name in ("FP8", "FP6", "FP4", "BF16"):
            prefix = f"val {name} = "
            if stripped.startswith(prefix):
                codes[name] = int(stripped[len(prefix):].split(".U", 1)[0])
    if codes != {"FP8": 0, "FP6": 1, "FP4": 2, "BF16": 3}:
        raise ValueError("selected RTL format codes differ from software projection")
    exp = _table(source, "val exp_bits =", "val mant_bits =")
    fraction = _table(source, "val mant_bits =", "val pmax =")
    maximum = _table(source, "val pmax =", "val log2_pmax_floor =")
    for name, rtl_name in (("mxfp8", "FP8"), ("mxfp6", "FP6"), ("mxfp4", "FP4")):
        fmt = contract["formats"][name]
        element = fmt["element"]
        if (element["exponent_bits"], element["fraction_bits"], element["max_finite"],
            fmt["config_ex_format_code"]) != (exp[rtl_name], fraction[rtl_name],
                                             maximum[rtl_name], codes[rtl_name]):
            raise ValueError(f"{name} contract differs from selected RTL format table")
    if contract["output_requantization"]["bf16_readout_code"] != codes["BF16"]:
        raise ValueError("BF16 readout code differs from RTL")
    epsilon = re.search(r"\bval EPS_BIASED_EXP = (\d+)\.U\(8\.W\)", source)
    if epsilon is None:
        raise ValueError("selected RTL block-scale floor is missing")
    if (contract["zero_block_scale_e8m0"] != int(epsilon.group(1)) or
        contract["scale_rule"] != "floor_log2_of_bf16_block_max_with_2pow_minus23_floor" or
        "val clamped_exp = Mux(max_biased_exp < EPS_BIASED_EXP, EPS_BIASED_EXP, max_biased_exp)" not in source or
        "scale_exponent := clamped_exp.zext.asSInt - 127.S - log2_pmax_floor.zext.asSInt" not in source or
        "val log2_pmax_floor = 0.U" not in source):
        raise ValueError("block-scale rule differs from selected RTL")
    rounding = (root / "src/main/scala/gemmini/BF16ScalaRoundToTiny.scala").read_text()
    if (contract["operand_rounding"] != "rne" or
        "val fp8_out = Mux(io.mx_fp8_altfmt, BF16ToE5M2(scaled_bf16), BF16ToE4M3(scaled_bf16))" not in rounding or
        "val fp6_out = Mux(io.mx_fp8_altfmt, BF16ToE2M3(scaled_bf16), BF16ToE3M2(scaled_bf16))" not in rounding or
        "roundAnyRawFNToRecFN.io.roundingMode  := consts.round_near_even" not in rounding or
        "roundToMx(scaled_bf16, inputexpWidth, inputsigWidth, format_fp4, (in: UInt) => E3M1Tofp4(in))" not in rounding):
        raise ValueError("operand rounding path differs from selected RTL")
    config = (root / "src/main/scala/gemmini/ConfigsFP.scala").read_text()
    if ("val standaloneMxFPConfig = defaultMxFPConfig.copy(" not in config or
        "class GemminiMxFPStandaloneConfig extends Config" not in config or
        "LazyModule(new Gemmini(GemminiMxFPConfigs.standaloneMxFPConfig))" not in config):
        raise ValueError("selected standalone config missing")
    selected_config = selected_config_facts(config)
    check_selected_config(yaml.safe_load(spec_bytes), contract, selected_config)
    if "val scale_resident = Input(Bool())" not in source:
        raise ValueError("selected requantizer lacks resident output-scale path")
    report = {"schema": "mx_gemmini.rtl_source_check.v1", "status": "source_crosscheck",
            "rtl_commit": record["gemmini_commit"], "mxgen_commit": record["mxgen_commit"],
            "checked_files": dict(record["files"]),
            "format_codes": codes, "exponent_bits": exp, "fraction_bits": fraction,
            "max_finite": maximum, "zero_block_scale_e8m0": int(epsilon.group(1)),
            "operand_rounding": contract["operand_rounding"],
            "selected_config": selected_config}
    if contract["rtl_commit"] == CANDIDATE_RTL_COMMIT:
        report["candidate_protocol"] = _check_candidate_protocol(root, spec_bytes)
    return report


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Cross-check a selected RTL checkout")
    parser.add_argument("rtl_root", help="Gemmini checkout containing the pinned mxgen submodule")
    parser.add_argument("--contract", default="contracts/software-spec.yaml")
    parser.add_argument("--sources", default="contracts/rtl_sources.yaml")
    args = parser.parse_args()
    report = check_sources(args.rtl_root, Path(args.contract).read_bytes(),
                           Path(args.sources).read_bytes())
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
