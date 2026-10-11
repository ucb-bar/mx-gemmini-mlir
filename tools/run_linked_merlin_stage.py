"""Run an existing receipt-bound Merlin/MX stage ELF on pinned Spike.

Build and execution have separate receipts so a multi-gigabyte checkpoint
does not need relinking to retry the simulator or inspect a longer run.
Completion proves the selected stage ran, not quantized numerical parity or
a multi-program inference session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from mx_gemmini_support.target_profile import load_profile, profile_sha256
from tools.build_merlin_stage import _capture_model, _observe_spike_evidence


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    return path.resolve(strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path)
    parser.add_argument("--merlin-src", type=Path, required=True)
    parser.add_argument("--riscv-root", type=Path, required=True)
    parser.add_argument("--spike-extlib", type=Path,
                        help="required for any stage with MX calls")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=int, default=3600)
    args = parser.parse_args()
    if args.timeout_s < 1:
        parser.error("--timeout-s must be positive")
    out = args.out_dir.absolute()
    if out.exists() or out.is_symlink():
        parser.error("--out-dir must be a fresh directory")
    stage = args.stage.resolve(strict=True)
    source = _capture_model(stage)
    receipt_path = _regular(args.build_receipt, "build receipt")
    build = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (build.get("schema") != "mx_gemmini.merlin_stage_probe.v1"
            or build.get("status") not in {
                "linked_experimental_numeric_qualification_pending",
                "spike_inference_incomplete"}):
        parser.error("build receipt is not a linked MX stage probe")
    if build.get("capture_model_sha256") != _sha(source):
        parser.error("stage source differs from the linked capture")
    profile = load_profile(_regular(args.profile, "profile"), rtl_root=args.rtl_root)
    if build.get("profile_sha256") != profile_sha256(profile):
        parser.error("profile differs from the linked stage")
    elf = _regular(Path(build["elf"]), "linked ELF")
    if elf.stat().st_size != build.get("elf_bytes"):
        parser.error("linked ELF size differs from the build receipt")
    selected = build.get("selected_operations")
    if type(selected) is not int or selected < 0:
        parser.error("build receipt has no valid MX selection count")
    route = "mx_gemmini_and_host" if selected else "host_only"
    if build.get("execution_route", route) != route:
        parser.error("build receipt execution route conflicts with MX selection")
    extlib = None
    if args.spike_extlib is not None:
        extlib = _regular(args.spike_extlib, "Spike extension")
    if selected and extlib is None:
        parser.error("MX-routed stage requires --spike-extlib")
    spike = _regular(args.riscv_root.resolve(strict=True) / "bin/spike", "Spike executable")
    merlin_src = args.merlin_src.resolve(strict=True)
    if not (merlin_src / "merlin/runtime/backends/spike_model.py").is_file():
        parser.error("--merlin-src must be Merlin's src directory")
    sys.path.insert(0, str(merlin_src))
    from merlin.runtime.backends import spike_model

    mem_bytes = build.get("mem_bytes")
    if type(mem_bytes) is not int or mem_bytes < elf.stat().st_size:
        parser.error("build receipt has no valid Spike memory size")
    cmd = [str(spike), "--isa=rv64gc"]
    if extlib is not None:
        cmd += [f"--extlib={extlib}", "--extension=gemmini"]
    cmd += [f"-m{hex(spike_model.DRAM_BASE)}:{hex(mem_bytes)}", str(elf)]
    elf_sha = _sha(elf)
    out.mkdir(parents=True)
    log = out / "spike.log"
    with log.open("wb") as stream:
        try:
            completed = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=args.timeout_s, check=False)
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    log_bytes = log.read_bytes()
    console = log_bytes.decode("utf-8", errors="replace")
    done = exit_code == 0 and "DONE\n" in console
    result = {
        "schema": "mx_gemmini.linked_stage_spike_run.v1",
        "status": "complete_numeric_qualification_pending" if done else "incomplete",
        "build_receipt_sha256": _sha(receipt_path),
        "capture_model_sha256": build["capture_model_sha256"],
        "profile_sha256": build["profile_sha256"],
        "build_hash": build["build_hash"],
        "execution_route": route,
        "selected_operations": selected,
        "elf_sha256": elf_sha,
        "elf_bytes": elf.stat().st_size,
        "spike_sha256": _sha(spike),
        "extension_sha256": _sha(extlib) if extlib is not None else None,
        "command": cmd,
        "exit_code": exit_code,
        "done": done,
        "log": str(log),
        "log_sha256": hashlib.sha256(log_bytes).hexdigest(),
    }
    if done:
        try:
            result["output_digests"] = _observe_spike_evidence(
                console, source, build["build_hash"],
                first_typed=bool(build.get("first_output_sha256_enabled")))
        except (ValueError, spike_model.SpikeModelError) as exc:
            result["done"] = False
            result["status"] = "incomplete"
            result["output_evidence_error"] = str(exc)
    if _sha(elf) != elf_sha:
        result["done"] = False
        result["status"] = "incomplete"
        result["elf_changed_during_run"] = True
    path = out / "mx_spike_run.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(path), "status": result["status"],
                      "selected_operations": selected}, sort_keys=True))
    if not result["done"]:
        raise SystemExit("linked MX Spike run did not complete; inspect its receipt and log")


if __name__ == "__main__":
    main()
