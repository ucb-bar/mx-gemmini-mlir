"""Compile current model2MLIR two-matmul captures for Nicolas FP8/FP4/FP6 chains.

Each case binds the portable graph to the pinned source operands, emits a
data-free public RV64 object, and checks both source-header outputs on Spike.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.fp4_plain_chain import (
    render_fp4_plain_chain, source_resources as fp4_resources)
from mx_gemmini_support.fp6_plain_chain import (
    render_fp6_plain_chain, source_resources as fp6_resources)
from mx_gemmini_support.plain_chain_128 import render_plain_chain
from mx_gemmini_support.resident_pair_graph import FP6_INPUTS, INPUTS
from mx_gemmini_support.standard_chain_handoff import portable_chain_manifest
from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.compile_mx import _git_revision, _require_gitlink, _run, _sha, _source_closure
from tools.qualify_nicolas_fp4_resident_chain import (
    BUFFERS as FP4_BUFFERS, _spike as fp4_spike,
    compiler_driver as fp4_driver)
from tools.qualify_nicolas_fp6_resident_chain import (
    BUFFERS as FP6_BUFFERS, _spike as fp6_spike,
    compiler_driver as fp6_driver)
from tools.qualify_nicolas_resident_128 import _source_resources, _write_sources
from tools.qualify_nicolas_spad_requant_fp4 import _compile_program
from tools.qualify_nicolas_vpu_public_objects import _build_extension, _flags


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxGemminiRocketConfig.json"
RTL_REVISION = "266c593f2cb51d7e3fe83fc0317072b585ac3c52"
MODEL2MLIR_REVISION = "a042643e31366724ca0482389c4ee85a9f88e343"
MODEL2MLIR_CLOSURE = "6eb6648cf2eebd72cd481dff084b0e154e034fbe50b7bce31935091b1a237e39"
FP8_64_BASELINE = ROOT / (
    "docs/evidence/nicolas_plain_fp8_chain64_public_52e6132_266c593/"
    "object/object_manifest.json")
PACKED_BASELINES = {
    ("FP4", 64): ROOT / "docs/evidence/nicolas_fp4_connected_resident_266c593/chain",
    ("FP4", 128): ROOT / "docs/evidence/nicolas_fp4_connected_resident_128_266c593/chain",
    ("FP6", 64): ROOT / "docs/evidence/nicolas_fp6_connected_resident_64_266c593/chain",
    ("FP6", 128): ROOT / "docs/evidence/nicolas_fp6_connected_resident_128_266c593/chain",
}


def capture(model2mlir_root: Path, dimension: int, destination: Path) -> str:
    import torch

    if _source_closure(model2mlir_root,
                       list((model2mlir_root / "m2m").rglob("*.py"))) != MODEL2MLIR_CLOSURE:
        raise ValueError("current model2MLIR source closure differs from pinned revision")
    sys.path.insert(0, str(model2mlir_root))
    import m2m
    if Path(m2m.__file__).resolve().parents[1] != model2mlir_root.resolve():
        raise ValueError("model2MLIR import differs from selected source root")

    class Chain(torch.nn.Module):
        def forward(self, a: torch.Tensor, b1: torch.Tensor,
                    b2: torch.Tensor) -> torch.Tensor:
            return torch.matmul(torch.matmul(a, b1), b2)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        examples = tuple(torch.randn(dimension, dimension) for _ in range(3))
    result = m2m.convert(Chain().eval(), examples, backend="fx_importer")
    if not result.ok or result.path_taken != "fx_importer":
        raise ValueError(f"current model2MLIR chain capture failed: {result.diagnostics}")
    destination.write_text(result.mlir_text)
    return result.mlir_text


def _source_order_wrapper(driver: str, source_order: tuple[str, ...],
                          object_order: list[str]) -> str:
    if (set(source_order) != set(object_order) or
            driver.count("mx_issue(") != 2):
        raise ValueError("source checker driver and public object ABI differ")
    driver = driver.replace("mx_issue(", "mx_issue_source_order(")
    parameters = ", ".join(f"const void *{name}" for name in source_order)
    prototype = ", ".join("const void *" for _ in object_order)
    return (driver + f"\nvoid mx_issue({prototype});\n"
            f"void mx_issue_source_order({parameters}) {{\n"
            f"  mx_issue({', '.join(object_order)});\n}}\n")


def _compile_fp8_driver(case: Path, software: Path, riscv: Path,
                        issuer: Path) -> Path:
    cc = riscv / "bin/riscv64-unknown-elf-gcc"
    bench = software / "riscv-tests/benchmarks/common"
    build = case / "run"
    flags = _flags(software, bench, build)
    sources = [build / "mx_driver.c", build / "mx_data.S",
               *sorted(bench.glob("*.c")), *sorted(bench.glob("*.S"))]
    objects = []
    for index, source in enumerate(sources):
        obj = build / f"driver_{index}.o"
        _run([str(cc), *flags, "-c", str(source), "-o", str(obj)],
             cwd=build, log=build / f"compile_{index}.log")
        objects.append(obj)
    elf = build / "mx_program.elf"
    _run([str(cc), "-nostdlib", "-nostartfiles", "-static", "-T",
          str(bench / "test.ld"), str(issuer), *(str(obj) for obj in objects),
          "-lm", "-lgcc", "-o", str(elf)], cwd=build, log=build / "link.log")
    return elf


def _run_fp8_spike(spike: Path, extension: Path, elf: Path, log: Path,
                   dimension: int) -> dict:
    result = subprocess.run([str(spike), f"--extlib={extension}",
                             "--extension=gemmini", str(elf)],
                            cwd=log.parent, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    marker = (f"lowered connected {dimension}x{dimension}: C1 0 codes 0 scales; "
              "C2 0 codes 0 scales")
    return {"matched": result.returncode == 0 and marker in result.stdout,
            "exit_code": result.returncode, "elf_sha256": _sha(elf),
            "spike_log_sha256": _sha(log)}


def _check_prior_commands(precision: str, dimension: int, commands: list,
                          source_sha: str, header_sha: str) -> dict:
    if (precision, dimension) not in PACKED_BASELINES:
        return {"scope": "no prior physical command archive selected"}
    archive = PACKED_BASELINES[(precision, dimension)]
    baseline = json.loads((archive / "receipt.json").read_text())
    prior_bytes = gzip.decompress((archive / "physical_program.json.gz").read_bytes())
    prior = json.loads(prior_bytes)
    if (baseline.get("source_sha256") != source_sha or
            baseline.get("header_sha256") != header_sha or
            baseline.get("status") != "source_and_compiler_matched_on_pinned_spike" or
            baseline.get("physical_sha256") != hashlib.sha256(prior_bytes).hexdigest() or
            prior.get("commands") != commands):
        raise ValueError("current portable chain commands differ from source-qualified stream")
    return {"scope": "all physical commands equal to source-qualified stream",
            "baseline_receipt_sha256": _sha(archive / "receipt.json"),
            "baseline_command_count": len(commands)}


def qualify_case(precision: str, dimension: int, *, frontend: str,
                 args: argparse.Namespace, profile: dict,
                 extension: Path) -> dict:
    software = args.rtl_root / "software/gemmini-rocc-tests"
    name = f"matmul_tiled_{precision.lower()}_{dimension}x{dimension}_chain"
    source = software / f"bareMetalC/{name}.c"
    header = software / f"include/matmul_{precision.lower()}_{dimension}x{dimension}_chain.h"
    source_sha, header_sha = _sha(source), _sha(header)
    case = args.out_dir / f"{precision.lower()}_{dimension}"
    case.mkdir()
    (case / "model2mlir.mlir").write_text(frontend)
    manifest = portable_chain_manifest(
        frontend, profile, precision=precision,
        source_driver_sha256=source_sha,
        source_header_sha256=header_sha)
    (case / "frontend_binding.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if precision == "FP8":
        resources = _source_resources(source, header, with_mm1=True,
                                      source_rows=dimension)
        bound = render_plain_chain(
            frontend, manifest, profile, resources,
            source_sha256=source_sha, header_sha256=header_sha,
            width=dimension)
        inputs = INPUTS
    elif precision == "FP4":
        resources = fp4_resources(source.read_text(), header.read_text(),
                                  dimension=dimension)
        bound = render_fp4_plain_chain(
            profile, resources, source_sha256=source_sha,
            header_sha256=header_sha, frontend_mlir=frontend,
            frontend_manifest=manifest, dimension=dimension)
        inputs = INPUTS
    else:
        resources = fp6_resources(source.read_text(), header.read_text(),
                                  dimension=dimension)
        bound = render_fp6_plain_chain(
            profile, resources, source_sha256=source_sha,
            header_sha256=header_sha, frontend_mlir=frontend,
            frontend_manifest=manifest, dimension=dimension)
        inputs = FP6_INPUTS
    mlir = case / "connected.mlir"
    mlir.write_text(bound)
    data = case / "resources"
    data.mkdir()
    for name in inputs:
        (data / f"{name}.bin").write_bytes(resources[name])
    abi = json.loads((ROOT / "examples/resident-pair-abi.json").read_text())
    for name in inputs:
        abi["inputs"][name] = name
    abi_path = case / "abi.json"
    abi_path.write_text(json.dumps(abi, indent=2, sort_keys=True) + "\n")
    objdir = case / "object"
    _run([sys.executable, "-m", "tools.compile_object",
          "--mlir", str(mlir), "--profile", str(args.profile),
          "--rtl-root", str(args.rtl_root), "--riscv-root", str(args.riscv_root),
          "--resources-dir", str(data), "--abi-json", str(abi_path),
          "--mx-opt", str(args.mx_opt), "--out-dir", str(objdir)],
         cwd=ROOT, log=case / "object_compile.log")
    object_manifest = json.loads((objdir / "object_manifest.json").read_text())
    compile_manifest = json.loads((objdir / "compile_manifest.json").read_text())
    if (compile_manifest.get("lowering_family") != "resident_pair" or
            object_manifest.get("allocated_data_section_bytes") != 0 or
            object_manifest.get("transport") != "rocket_rocc"):
        raise ValueError("portable MX chain did not build a data-free resident object")
    issuer = objdir / "mx_issue.o"
    order = [slot["name"] for slot in object_manifest["buffer_abi"]]
    physical = json.loads((objdir / "physical_program.json").read_text())
    prior = _check_prior_commands(precision, dimension, physical["commands"],
                                  source_sha, header_sha)
    if precision == "FP8" and dimension == 64:
        baseline = json.loads(FP8_64_BASELINE.read_text())
        capture_receipt = FP8_64_BASELINE.parents[1] / "capture/receipt.json"
        capture = json.loads(capture_receipt.read_text())
        if (capture.get("source_sha256") != source_sha or
                capture.get("header_sha256") != header_sha or
                capture.get("rtl_revision") != RTL_REVISION or
                _sha(issuer) != baseline["object_sha256"]):
            raise ValueError("current portable FP8 64 object differs from published object")
        prior = {"scope": "object bytes equal to source-qualified public object",
                 "baseline_manifest_sha256": _sha(FP8_64_BASELINE),
                 "baseline_capture_receipt_sha256": _sha(capture_receipt)}
    spike = args.riscv_root / "bin/spike"
    if precision == "FP8":
        from mx_gemmini_support.plain_chain_128 import lower_plain_chain

        commands = lower_plain_chain(
            bound, frontend, manifest, profile, resources,
            source_sha256=source_sha, header_sha256=header_sha,
            width=dimension)
        build = case / "run"
        _write_sources(build, commands, resources, with_mm1=True,
                       source_rows=dimension, width=dimension)
        elf = _compile_fp8_driver(case, software, args.riscv_root, issuer)
        execution = _run_fp8_spike(spike, extension, elf, build / "spike.log",
                                    dimension)
    else:
        source_order = FP4_BUFFERS if precision == "FP4" else FP6_BUFFERS
        driver = (fp4_driver if precision == "FP4" else fp6_driver)(
            source.read_text(), dimension=dimension)
        driver = _source_order_wrapper(driver, source_order, order)
        driver_path = case / "compiler_driver.c"
        driver_path.write_text(driver)
        elf = _compile_program(case / "run", driver_path, software,
                               args.riscv_root / "bin/riscv64-unknown-elf-gcc", issuer)
        execution = (fp4_spike if precision == "FP4" else fp6_spike)(
            spike, extension, elf, case / "run/spike.log")
    if not execution["matched"]:
        raise ValueError(f"portable {precision} {dimension} compiler result differs from source")
    case_receipt = {
        "schema": "mx_gemmini.nicolas_portable_chain_case.v1",
        "status": "compiler_matched_both_source_header_outputs_on_pinned_spike",
        "precision": precision, "dimension": dimension,
        "source_sha256": source_sha, "header_sha256": header_sha,
        "profile_sha256": profile_sha256(profile),
        "frontend_mlir_sha256": _sha(case / "model2mlir.mlir"),
        "frontend_binding_sha256": _sha(case / "frontend_binding.json"),
        "bound_mlir_sha256": _sha(mlir),
        "object_manifest_sha256": _sha(objdir / "object_manifest.json"),
        "object_sha256": _sha(issuer),
        "physical_sha256": _sha(objdir / "physical_program.json"),
        "prior_source_qualified_stream": prior,
        "compiler_spike": execution,
        "checked_outputs": {"c1_codes": dimension * dimension,
                            "c1_scales": dimension * dimension // 32,
                            "c2_codes": dimension * dimension,
                            "c2_scales": dimension * dimension // 32},
    }
    (case / "receipt.json").write_text(
        json.dumps(case_receipt, indent=2, sort_keys=True) + "\n")
    return {"case": f"{precision.lower()}_{dimension}",
            "receipt_sha256": _sha(case / "receipt.json"),
            "object_sha256": case_receipt["object_sha256"],
            "spike_log_sha256": execution["spike_log_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model2mlir-root", "rtl-root", "riscv-root", "mx-opt", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--profile", type=Path, default=PROFILE)
    args = parser.parse_args()
    for name in ("model2mlir_root", "rtl_root", "riscv_root", "mx_opt",
                 "out_dir", "profile"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    if _git_revision(args.rtl_root) != RTL_REVISION:
        raise ValueError("portable chain suite requires Nicolas's pinned RTL")
    for gitlink in ("software/gemmini-rocc-tests", "software/libgemmini"):
        _require_gitlink(args.rtl_root, gitlink)
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if profile["name"] != "MxGemminiRocketConfig":
        raise ValueError("portable chain suite requires Nicolas's plain MX profile")
    args.out_dir.mkdir(parents=True)
    extension = _build_extension(
        args.rtl_root / "software/libgemmini", args.riscv_root, args.out_dir)
    rows = []
    for dimension in (64, 128):
        capture_path = args.out_dir / f"model2mlir_{dimension}.mlir"
        frontend = capture(args.model2mlir_root, dimension, capture_path)
        for precision in ("FP8", "FP4", "FP6"):
            rows.append(qualify_case(
                precision, dimension, frontend=frontend, args=args,
                profile=profile, extension=extension))
    receipt = {
        "schema": "mx_gemmini.nicolas_portable_chain_suite.v1",
        "status": "six_current_frontend_chains_matched_source_headers_on_pinned_spike",
        "rtl_revision": RTL_REVISION,
        "model2mlir_revision": MODEL2MLIR_REVISION,
        "model2mlir_source_closure_sha256": MODEL2MLIR_CLOSURE,
        "compiler_revision": _git_revision(ROOT),
        "profile_sha256": profile_sha256(profile),
        "spike_sha256": _sha(args.riscv_root / "bin/spike"),
        "riscv_gcc_sha256": _sha(args.riscv_root / "bin/riscv64-unknown-elf-gcc"),
        "mx_opt_sha256": _sha(args.mx_opt),
        "spike_extension_sha256": _sha(extension),
        "rows": rows,
    }
    (args.out_dir / "index.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt["status"])


if __name__ == "__main__":
    main()
