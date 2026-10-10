"""Compile one verified MX MLIR module to a data-free RV64 RoCC object.

The graph selects an existing physical lowerer. Source-specialized modules
take a checked payload bundle; connected runtime graphs take their input files
and an explicit buffer ABI. Unsupported graphs fail before object emission.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from mx_gemmini_support.verify_profile_ir import verify_ir
from tools.compile_mx import _git_revision, _source_closure


ROOT = Path(__file__).resolve().parents[1]
EMITTERS = {
    "source_contract": "tools.emit_mx_object",
    "resident_pair": "tools.emit_resident_pair_object",
    "resident_vpu_pair": "tools.emit_resident_vpu_object",
}
MANIFEST_SCHEMAS = {
    "source_contract": "mx_gemmini.linkable_object.v1",
    "resident_pair": "mx_gemmini.resident_pair_linkable_object.v1",
    "resident_vpu_pair": "mx_gemmini.resident_vpu_linkable_object.v1",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(mlir_text: str, profile: dict) -> tuple[str, dict]:
    """Select only executable graph families with an existing object lowerer."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    report = verify_ir(mlir_text, profile)
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    source = module.attributes.get("mx.payload_binding_schema") is not None
    runtime = module.attributes.get("mx.runtime_resources_sha256") is not None
    if source == runtime:
        raise ValueError("MX object needs exactly one payload binding scheme")
    counts = (report["contracts"], report["resident_contracts"],
              report["vpu_commands"], report["spad_requants"])
    if source and counts[0] == 1 and counts[1] == 0 and report["source_resources"]:
        return "source_contract", report
    if runtime and counts == (1, 1, 0, 0):
        return "resident_pair", report
    if runtime and counts[0:2] == (1, 1) and 1 <= counts[2] <= 16 and counts[3] == 1:
        return "resident_vpu_pair", report
    raise ValueError(f"MX object has no qualified lowering for graph counts {counts}")


def _resident_pair_precision(mlir_text: str) -> str:
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func
    from xdsl.parser import Parser

    from mx_gemmini_support.verify_profile_ir import _operation_name, _text_attr

    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    mm2 = [op for op in module.walk()
           if _operation_name(op) == "mx_gemmini.resident_contract"]
    if len(mm2) != 1:
        raise ValueError("connected MX object needs one resident contraction")
    precision = _text_attr(mm2[0], "activation_format")
    if precision not in ("fp8_e4m3", "fp4_e2m1", "fp6_e3m2"):
        raise ValueError("connected MX object has no resident precision lowerer")
    return precision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("mlir", "profile", "rtl-root", "riscv-root", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--bundle", type=Path,
                        help="checked payload bundle for one source contraction")
    parser.add_argument("--resources-dir", type=Path,
                        help="checked runtime input files for a connected pair")
    parser.add_argument("--abi-json", type=Path,
                        help="runtime buffer map for a connected pair")
    parser.add_argument("--mx-opt", type=Path,
                        help="also run the native MX dialect verifier")
    args = parser.parse_args()
    for name in ("mlir", "profile", "rtl_root", "riscv_root", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    profile = load_profile(args.profile, rtl_root=args.rtl_root)
    if profile.get("transport") != "rocket_rocc":
        raise ValueError("MX object compiler requires Rocket RoCC transport")
    raw = args.mlir.read_bytes()
    mlir_bytes = gzip.decompress(raw) if args.mlir.name.endswith(".mlir.gz") else raw
    if not (args.mlir.suffix == ".mlir" or args.mlir.name.endswith(".mlir.gz")):
        raise ValueError("MX object input must be .mlir or .mlir.gz")
    mlir_text = mlir_bytes.decode()
    family, report = classify(mlir_text, profile)
    if family == "source_contract":
        if args.bundle is None or args.resources_dir or args.abi_json:
            raise ValueError("source MX object needs --bundle only")
        if args.mlir.name.endswith(".gz"):
            raise ValueError("source MX object currently needs plain .mlir")
        extra = ["--bundle", str(args.bundle.resolve())]
    else:
        if args.bundle or args.resources_dir is None or args.abi_json is None:
            raise ValueError("connected MX object needs --resources-dir and --abi-json only")
        extra = ["--resources-dir", str(args.resources_dir.resolve()),
                 "--abi-json", str(args.abi_json.resolve())]
        if family == "resident_pair":
            extra += ["--precision", _resident_pair_precision(mlir_text)]
    if args.mx_opt is not None:
        with tempfile.TemporaryDirectory(prefix="mx-object-verify-") as temp:
            native_input = Path(temp) / "input.mlir"
            native_input.write_bytes(mlir_bytes)
            subprocess.run([str(args.mx_opt.resolve()), str(native_input),
                            "-o", "/dev/null"], check=True)
    emitter = EMITTERS[family]
    command = [sys.executable, "-m", emitter,
               "--mlir", str(args.mlir), "--profile", str(args.profile),
               "--rtl-root", str(args.rtl_root),
               "--riscv-root", str(args.riscv_root),
               "--out-dir", str(args.out_dir), *extra]
    subprocess.run(command, cwd=ROOT, check=True)
    object_manifest = args.out_dir / "object_manifest.json"
    built = json.loads(object_manifest.read_text())
    if (built.get("schema") != MANIFEST_SCHEMAS[family] or
            built.get("transport") != "rocket_rocc" or
            built.get("bound_mlir_sha256") != hashlib.sha256(mlir_bytes).hexdigest() or
            built.get("object_sha256") != _sha(args.out_dir / "mx_issue.o") or
            built.get("profile_sha256") != profile_sha256(profile) or
            built.get("allocated_data_section_bytes") != 0 or
            built.get("embedded_operand_bytes") != 0 or
            built.get("embedded_golden_bytes") != 0):
        raise ValueError("MX object emitter returned an unverified object")
    manifest = {
        "schema": "mx_gemmini.compiler_object_dispatch.v1",
        "status": "rv64_rocc_object_built",
        "lowering_family": family,
        "emitter": emitter,
        "compiler_revision": _git_revision(ROOT),
        "compiler_source_closure_sha256": _source_closure(
            ROOT, sorted((ROOT / "mx_gemmini_support").glob("*.py")) +
            sorted((ROOT / "tools").glob("*.py"))),
        "profile_sha256": profile_sha256(profile),
        "bound_mlir_sha256": hashlib.sha256(mlir_bytes).hexdigest(),
        "object_manifest_sha256": _sha(object_manifest),
        "object_sha256": built["object_sha256"],
        "physical_program_sha256": built["physical_program_sha256"],
        "verified_graph_counts": {
            name: report[name] for name in
            ("contracts", "resident_contracts", "vpu_commands", "spad_requants",
             "source_resources", "lut_uploads", "runtime_luts")},
    }
    if args.mx_opt is not None:
        manifest["native_verifier_sha256"] = _sha(args.mx_opt.resolve())
    (args.out_dir / "compile_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"MX {family} object: {args.out_dir / 'mx_issue.o'}")


if __name__ == "__main__":
    main()
