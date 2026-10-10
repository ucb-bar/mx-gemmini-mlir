"""Reproduce selected compiler-issued Muon MX+VPU ELFs on isolated Cyclotron.

The model patch adds only the FP8/FP4 command subset exercised by these fixtures.
This is experimental functional-simulator evidence, not RTL/FPGA parity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from mx_gemmini_support.quant_reference import exact_bf16_x2
from mx_gemmini_support.source_payload import load_bundle
from tools.link_mx_muon_payload import _tile_major_reference


REVISION = "2d6adad4ad94d1621fdff9c9a1e5eac871048f24"
MODEL_SHA256 = "3bbb9ba9f1f61d838b240f0bca3571f9f17a2717b12d819b8f42c6d78d56e3ea"
PATCH = (Path(__file__).resolve().parents[1] /
         "docs/evidence/fp8_vpu_muon_payload_266c593/cyclotron_compiler_stream.patch")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], cwd: Path, log: Path, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command[0]}; see {log}")


def _symbol(nm: Path, elf: Path, name: str) -> tuple[int, int]:
    symbols = subprocess.check_output([str(nm), "-S", str(elf)], text=True)
    found = re.search(rf"^([0-9a-f]+)\s+([0-9a-f]+)\s+[BDR]\s+{name}$",
                      symbols, re.MULTILINE)
    if found is None:
        raise ValueError(f"payload ELF lacks {name} symbol")
    return int(found.group(1), 16), int(found.group(2), 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cyclotron-root", "payload-dir", "bundle", "muon-llvm", "out-dir"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    for name in ("cyclotron_root", "payload_dir", "bundle", "muon_llvm", "out_dir"):
        setattr(args, name, getattr(args, name).resolve())
    if args.out_dir.exists():
        parser.error(f"refusing to overwrite {args.out_dir}")
    revision = subprocess.check_output(
        ["git", "-C", str(args.cyclotron_root), "rev-parse", "HEAD"],
        text=True).strip()
    if (revision != REVISION or
            _sha(args.cyclotron_root / "src/muon/mxgemmini/mod.rs") != MODEL_SHA256 or
            not PATCH.is_file()):
        parser.error("selected Cyclotron source differs from pinned unmodified model")
    payload_receipt = json.loads((args.payload_dir / "payload_manifest.json").read_text())
    elf = args.payload_dir / "mx_kernel.elf"
    source_manifest, resources = load_bundle(args.bundle)
    expected = exact_bf16_x2(resources["golden_bf16"])
    layout = payload_receipt.get("bf16_output_layout")
    if layout == "output_tile_major_bf16":
        expected = _tile_major_reference(expected, payload_receipt["shape_mnk"],
                                         payload_receipt["output_plan"])
    elif layout != "row_major_bf16":
        parser.error("selected Muon BF16 output layout is unsupported")
    if (payload_receipt.get("schema") != "mx_gemmini.muon_payload_elf.v1" or
            payload_receipt.get("precision") != source_manifest.get("precision") or
            payload_receipt.get("shape_mnk") != source_manifest.get("shape_mnk") or
            payload_receipt.get("files_sha256", {}).get("mx_kernel.elf") != _sha(elf) or
            payload_receipt.get("bundle_manifest_sha256") != _sha(args.bundle / "manifest.json") or
            payload_receipt.get("expected_bf16_sha256") != hashlib.sha256(expected).hexdigest() or
            payload_receipt.get("bf16_elements_to_compare") != 65536):
        parser.error("selected compiler ELF differs from checked MX+VPU payload")
    nm = args.muon_llvm / "bin/llvm-nm"
    if not nm.is_file():
        parser.error("Muon toolchain lacks llvm-nm")
    mismatch_addr, mismatch_size = _symbol(nm, elf, "mx_mismatch_count")
    complete_addr, complete_size = _symbol(nm, elf, "mx_completed")
    output_addr, output_size = _symbol(nm, elf, "output_bf16")
    if (mismatch_size != 4 or complete_size != 4 or output_size != len(expected) or
            min(mismatch_addr, complete_addr, output_addr) < 0x10000000):
        parser.error("Muon verifier symbols differ from checked BF16 ABI")
    args.out_dir.mkdir(parents=True)
    model = args.out_dir / "cyclotron"
    _run(["git", "clone", "--quiet", "--local", "--no-hardlinks",
          str(args.cyclotron_root), str(model)], args.out_dir,
         args.out_dir / "clone.log")
    _run(["git", "checkout", "--detach", REVISION], model,
         args.out_dir / "checkout.log")
    _run(["git", "apply", "--check", str(PATCH)], model,
         args.out_dir / "patch_check.log")
    _run(["git", "apply", str(PATCH)], model, args.out_dir / "patch_apply.log")
    target = args.out_dir / "target"
    build_env = os.environ.copy()
    build_env["CARGO_TARGET_DIR"] = str(target)
    _run(["cargo", "build", "--release", "--offline"], model,
         args.out_dir / "build.log", build_env)
    tests_log = args.out_dir / "mxgemmini_tests.log"
    _run(["cargo", "test", "--offline", "mxgemmini"], model, tests_log, build_env)
    if "test result: ok. 35 passed; 0 failed" not in tests_log.read_text():
        raise ValueError("patched Cyclotron MX regression tests differ")
    binary = target / "release/cyclotron"
    model_source = model / "src/muon/mxgemmini/mod.rs"
    start = min(mismatch_addr, complete_addr, output_addr)
    length = max(mismatch_addr + 4, complete_addr + 4,
                 output_addr + output_size) - start
    runs = []
    for label in ("first", "repro"):
        dump = args.out_dir / f"{label}.gmem.bin"
        env = os.environ.copy()
        env.update({"CYCLOTRON_MXGEMMINI": "1",
                    "CYCLOTRON_MXGEMMINI_COMPILER": "1",
                    "CYCLOTRON_DUMP_GMEM": f"{start:#x}:{length}:{dump}"})
        log = args.out_dir / f"{label}.cyclotron.log"
        _run([str(binary), "config.toml", "--binary-path", str(elf)], model, log, env)
        data = dump.read_bytes()
        if len(data) != length or "simulation finished" not in log.read_text():
            raise ValueError("patched Cyclotron failed to finish or dump the full verifier")
        mismatches = int.from_bytes(data[mismatch_addr-start:mismatch_addr-start+4], "little")
        completed = int.from_bytes(data[complete_addr-start:complete_addr-start+4], "little")
        output = data[output_addr-start:output_addr-start+output_size]
        independent = sum(output[i:i+2] != expected[i:i+2]
                          for i in range(0, len(expected), 2))
        if mismatches != 0 or completed != 1 or independent != 0:
            raise ValueError(f"patched Cyclotron BF16 parity failed: {mismatches}/{independent}")
        runs.append({"label": label, "gmem_sha256": _sha(dump),
                     "cyclotron_log_sha256": _sha(log),
                     "output_sha256": hashlib.sha256(output).hexdigest(),
                     "completed": completed, "reported_bf16_mismatches": mismatches,
                     "independent_bf16_mismatches": independent})
    if runs[0]["gmem_sha256"] != runs[1]["gmem_sha256"]:
        raise ValueError("patched Cyclotron output is not deterministic")
    receipt = {
        "schema": "mx_gemmini.muon_payload_patched_cyclotron.v1",
        "status": "all_65536_bf16_outputs_matched_on_isolated_patched_cyclotron",
        "qualification": "experimental_functional_model_only",
        "scope": (f"{source_manifest['precision']} "
                  f"{'x'.join(map(str, source_manifest['shape_mnk']))} MX+VPU compiler "
                  "Muon MMIO path; no RTL or FPGA claim"),
        "precision": source_manifest["precision"],
        "bf16_output_layout": layout,
        "cyclotron_revision": revision,
        "stock_model_sha256": MODEL_SHA256,
        "patched_model_sha256": _sha(model_source),
        "patch_sha256": _sha(PATCH),
        "qualifier_sha256": _sha(Path(__file__)),
        "cyclotron_binary_sha256": _sha(binary),
        "cyclotron_mx_tests_sha256": _sha(tests_log),
        "cyclotron_mx_tests_passed": 35,
        "payload_manifest_sha256": _sha(args.payload_dir / "payload_manifest.json"),
        "compiler_elf_sha256": _sha(elf),
        "expected_bf16_sha256": hashlib.sha256(expected).hexdigest(),
        "bf16_outputs_checked": len(expected) // 2,
        "output_address": output_addr,
        "mismatch_address": mismatch_addr,
        "completed_address": complete_addr,
        "runs": runs,
    }
    (args.out_dir / "qualification.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("compiler-issued Muon MX+VPU matched 65,536/65,536 BF16 outputs on isolated Cyclotron")


if __name__ == "__main__":
    main()
