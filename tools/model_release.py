"""Load checksum-verified model inputs and plan a portable build of the release."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "examples/models/release.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def registry():
    return json.loads(REGISTRY.read_text())


def reader(path):
    spec = importlib.util.spec_from_file_location("release_reader_" + path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence(name):
    if not __debug__:
        raise ValueError("archive verification requires Python assertions; unset PYTHONOPTIMIZE")
    item = registry()["models"][name]
    directory = ROOT / item["archive"]
    if item["kind"] == "forward":
        module = reader(directory / "verify.py")
        files = module.composed_evidence(directory, json.loads((directory / "artifacts.json").read_text()), set())
    else:
        module = reader(ROOT / registry()["models"]["tinyllama-cached"]["archive"] / "verify.py")
        files = (module.assembled_members(directory)[1] if item["kind"] == "session"
                 else module.archive_members(directory)[1])
    return item, files


def probe(files, item, stage):
    prefix = "candidate/" if item["kind"] == "forward" else f"candidate/stages/{stage}/"
    return json.loads(files[prefix + "mx_stage_probe.json"])


def export_inputs(name, out, weights, *, link_weights=False, capture_only=False):
    item, files = evidence(name)
    captures = {key.removeprefix("capture/"): value for key, value in files.items() if key.startswith("capture/")}
    if item["kind"] == "forward":
        supplement = registry()["forward_capture"]
        path = ROOT / supplement["archive"]
        if path.stat().st_size != supplement["bytes"] or sha(path) != supplement["sha256"]:
            raise ValueError("forward capture supplement changed")
        with tarfile.open(path, "r:xz") as archive:
            for member in archive:
                if not member.isfile() or member.name not in supplement["members"]:
                    raise ValueError("unlisted capture supplement member")
                raw = archive.extractfile(member).read()
                record = supplement["members"][member.name]
                if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
                    raise ValueError("capture supplement member changed")
                key = member.name.removeprefix("capture/")
                if key in captures and captures[key] != raw:
                    raise ValueError("capture supplement differs from measured input")
                captures[key] = raw
    prepared = {}
    for stage, record in item["stages"].items():
        if record["weights_sha256"] is None:
            continue
        if capture_only:
            continue
        source = weights.get(stage, weights.get("all"))
        if source is None or not source.is_file() or source.is_symlink() or sha(source) != record["weights_sha256"]:
            raise ValueError(f"{stage} needs plain weights.safetensors with SHA-256 {record['weights_sha256']}")
        prepared[stage] = source.resolve()
    if out.exists():
        raise ValueError("export directory must be fresh")
    # Validate every member before writing. Keep archived receipt bytes intact.
    for key in captures:
        path = PurePosixPath(key)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe capture path")
    out.mkdir(parents=True)
    for key, raw in captures.items():
        path = out / "capture" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for stage, source in prepared.items():
        directory = out / "capture" / ("" if item["kind"] == "forward" else "stages/" + stage)
        target = directory / "weights.safetensors"
        if link_weights:
            os.link(source, target)
        else:
            shutil.copyfile(source, target)
        if sha(target) != item["stages"][stage]["weights_sha256"]:
            raise ValueError("checkpoint changed during export")
    for stage, record in item["stages"].items():
        if "policy" in record:
            policy = ROOT / record["policy"]["path"]
            if sha(policy) != record["policy"]["sha256"]:
                raise ValueError("published policy changed")
            (out / (stage + "-policy.json")).write_bytes(policy.read_bytes())
    guards = [probe(files, item, stage).get("host_math_guard") for stage in item["stages"]]
    guards = [guard for guard in guards if guard]
    if not guards or any(guard != guards[0] for guard in guards):
        raise ValueError("stage host-math contracts differ")
    (out / "guard.json").write_text(json.dumps(guards[0], indent=2) + "\n")
    provenance = {"schema": "mx_gemmini.exported_model_inputs.v1", "model": name,
                  "checkpoint_files_present": not capture_only,
                  "release_sha256": sha(REGISTRY), "weight_storage": "shared_inode" if link_weights else "copy",
                  "files": {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()}}
    (out / "inputs.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def build_commands(name, inputs, build, merlin, llvm, riscv, python, *, extension=None):
    release = registry()
    item = release["models"][name]
    receipt = json.loads((inputs / "inputs.json").read_text())
    if receipt["model"] != name or receipt["release_sha256"] != sha(REGISTRY):
        raise ValueError("input export belongs to another model release")
    if not receipt.get("checkpoint_files_present"):
        raise ValueError("capture-only export needs a new export with matching checkpoint files before compilation")
    if any(sha(inputs / path) != digest for path, digest in receipt["files"].items()):
        raise ValueError("exported input or policy changed")
    # The local manifest cannot authorize a replacement release policy or checkpoint.
    for stage, record in item["stages"].items():
        capture = inputs / "capture" / ("" if item["kind"] == "forward" else "stages/" + stage)
        if sha(capture / "model.mlir") != record["capture_model_sha256"]:
            raise ValueError("captured graph differs from the published model")
        if "policy" in record and sha(inputs / (stage + "-policy.json")) != record["policy"]["sha256"]:
            raise ValueError("execution policy differs from the published model")
        if record["weights_sha256"] is not None and sha(capture / "weights.safetensors") != record["weights_sha256"]:
            raise ValueError("checkpoint differs from the published model")
    _, frozen = evidence(name)
    for key, raw in frozen.items():
        if key.startswith("capture/") and (inputs / key).read_bytes() != raw:
            raise ValueError("captured input differs from the published model: " + key)
    if item["kind"] == "forward":
        for key, record in release["forward_capture"]["members"].items():
            if sha(inputs / key) != record["sha256"]:
                raise ValueError("forward capture differs from the published supplement")
    guard = next(probe(frozen, item, stage)["host_math_guard"] for stage in item["stages"]
                 if probe(frozen, item, stage).get("host_math_guard"))
    if json.loads((inputs / "guard.json").read_text()) != guard:
        raise ValueError("host-math contract differs from the published model")
    profile = ROOT / release["profile"]["path"]
    common = ["--prepack-checkpoint-weights", "--prune-unused-checkpoint-weights", "--connect-vpu-multiplies",
              "--connect-vpu-sigmoid-multiplies", "--connect-vpu-broadcast-multiplies", "--prepare-vpu-broadcast-operands",
              "--share-activation-packing", "--integer-quant-groups", "--matrix-vpu-transport", "--vpu-matrix-transport"]
    commands = []
    for stage in item["stages"]:
        capture = inputs / "capture" / ("" if item["kind"] == "forward" else "stages/" + stage)
        command = [python, "-m", "tools.build_merlin_stage", "--stage", str(capture), "--workdir", str(build / stage),
                   "--profile", str(profile), "--merlin-src", str(merlin / "src"), "--llvm-install", str(llvm),
                   "--riscv-root", str(riscv), "--structural-elf-admission", "--compile-timeout-s", "6000"]
        if stage != "action_decode":
            command += common + ["--execution-policy", str(inputs / (stage + "-policy.json")),
                                 "--host-math-guard", str(inputs / "guard.json"), "--arena-mb", "2048",
                                 "--spike-arena-base", "0xd0000000"]
        if name == "tinyllama-forward":
            command += ["--native-tile-policy", "movement", "--reuse-batch-rhs", "--compact-batch-rhs",
                        "--output-dump-cap", "5120000"]
            if extension:
                command += ["--spike-extlib", str(extension), "--spike-timeout", "43200"]
        elif name == "tinyllama-cached" and stage == "decode":
            command += ["--virtual-zero-padding", "--reuse-batch-rhs", "--compact-batch-rhs"]
        commands.append(command)
    if item["kind"] != "forward":
        link = [python, "-m", "tools.link_merlin_session", "--session-root", str(inputs / "capture"),
                "--workdir", str(build / "link"), "--profile", str(profile), "--merlin-src", str(merlin / "src"),
                "--riscv-root", str(riscv), "--structural-elf-admission", "--measure-preparation",
                "--host-math-guard", str(inputs / "guard.json")]
        for stage in item["stages"]:
            link += ["--stage-build", stage + "=" + str(build / stage)]
        for stage in (["prefill", "decode"] if name == "tinyllama-cached" else ["flow_denoise", "action_decode"]):
            link += ["--dump-output", stage + ":0"]
        link += ["--dump-limit-bytes", "128000" if name == "tinyllama-cached" else "6400"]
        commands.append(link)
        if extension:
            commands.append([python, "-m", "tools.run_linked_merlin_session", "--session-root", str(inputs / "capture"),
                             "--link-receipt", str(build / "link/mx_session_link.json"), "--merlin-src", str(merlin / "src"),
                             "--riscv-root", str(riscv), "--spike-extlib", str(extension), "--out-dir", str(build / "run"),
                             "--timeout-s", "43200", "--progress-interval-s", "60", "--perf-config", "e4m3_vpu",
                             "--perf-set", "mem.host_tracking=0"])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    export = sub.add_parser("export", help="export measured capture and policy; supply matching checkpoint files")
    export.add_argument("--model", choices=tuple(registry()["models"]), required=True)
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--weights", action="append", default=[], help="STAGE=weights.safetensors; all=FILE shares a checkpoint")
    export.add_argument("--capture-only", action="store_true", help="export reference inputs without checkpoint files; cannot compile this export")
    export.add_argument("--link-weights", action="store_true", help="share immutable checkpoint inodes instead of copying")
    build = sub.add_parser("build", help="write portable build commands; --execute runs them")
    build.add_argument("--model", choices=tuple(registry()["models"]), required=True)
    for key in ["inputs", "build-dir", "merlin-root", "llvm-install", "riscv-root", "plan"]:
        build.add_argument("--" + key, type=Path, required=True)
    build.add_argument("--python", default=sys.executable)
    build.add_argument("--spike-extlib", type=Path)
    build.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.action == "export":
        if bool(args.weights) == args.capture_only:
            parser.error("select --weights or --capture-only")
        weights = {}
        for entry in args.weights:
            stage, separator, path = entry.partition("=")
            if not separator or stage in weights:
                parser.error("weights must be distinct STAGE=FILE selections")
            weights[stage] = Path(path)
        export_inputs(args.model, args.out.resolve(), weights, link_weights=args.link_weights, capture_only=args.capture_only)
        print("Exported capture, exact policies and checkpoint identities to", args.out)
    else:
        if args.build_dir.exists() or args.plan.exists():
            parser.error("build and plan paths must be fresh")
        commands = build_commands(args.model, args.inputs.resolve(), args.build_dir.resolve(), args.merlin_root.resolve(),
                                  args.llvm_install.resolve(), args.riscv_root.resolve(), args.python,
                                  extension=args.spike_extlib.resolve() if args.spike_extlib else None)
        timing_env = {"GEMMINI_MODE": "both", "GEMMINI_PERF_CONFIG": "e4m3_vpu", "GEMMINI_PERF_SET": "mem.host_tracking=0"}
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_text(json.dumps({"schema": "mx_gemmini.model_build_plan.v1", "model": args.model,
                                        "release_sha256": sha(REGISTRY), "environment": timing_env, "commands": commands}, indent=2) + "\n")
        print("Wrote", len(commands), "commands to", args.plan)
        if args.execute:
            for command in commands:
                subprocess.run(command, cwd=ROOT, env=os.environ | timing_env, check=True)


if __name__ == "__main__":
    main()
