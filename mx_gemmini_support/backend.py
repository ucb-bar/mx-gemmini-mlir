"""Installed Merlin service for the existing single-VPU source emitter."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .target_profile import load_profile, profile_sha256

name = "mx-gemmini"
_REQUIRED = {"mx.profile", "mx.buffers", "mx.emit"}
_OPTIONAL = {"mx.rtl-root"}


def validate(request) -> None:
    """Admit only typed MX IR, the declared buffer boundary and xDSL 0.65."""
    if request.source.kind != "mx-mlir":
        raise ValueError("MX source emission requires explicitly typed mx-mlir input")
    options = request.options
    if any(value is not None for value in (options.system, options.platform, options.harts)):
        raise ValueError("MX C-source emission does not build platform images or schedule harts")
    parameters = options.parameters
    if not _REQUIRED <= parameters.keys() <= _REQUIRED | _OPTIONAL:
        raise ValueError("MX emission needs mx.profile, mx.buffers and mx.emit; unknown options refuse")
    if parameters["mx.emit"] != "c-source":
        raise ValueError("MX service currently emits only explicit c-source artifacts")
    if not isinstance(parameters["mx.profile"], str) or not Path(parameters["mx.profile"]).is_file():
        raise ValueError("mx.profile must name an existing profile JSON file")
    buffers = parameters["mx.buffers"]
    if not isinstance(buffers, dict) or any(not isinstance(value, str) for value in buffers.values()):
        raise ValueError("mx.buffers must map explicit runtime buffer slots to C identifiers")
    if "mx.rtl-root" in parameters and (
        not isinstance(parameters["mx.rtl-root"], str) or not Path(parameters["mx.rtl-root"]).is_dir()
    ):
        raise ValueError("mx.rtl-root must name the selected existing RTL checkout")
    try:
        selected = version("xdsl")
    except PackageNotFoundError as exc:
        raise RuntimeError("MX typed-IR emission needs the package's compiler extra (xDSL 0.65.x)") from exc
    if selected.split(".")[:2] != ["0", "65"]:
        raise ValueError(f"MX typed-IR emission requires xDSL 0.65.x; found {selected}")


def compile(request):
    """Use the handwritten emitter; return source products without execution claims."""
    from merlin.compiler.artifact import Artifact, Compilation

    validate(request)
    from .vpu_emission import emit_vpu_elementwise, write_vpu_sources

    if request.output.exists():
        raise FileExistsError(f"MX compilation requires a fresh output directory: {request.output}")
    parameters = request.options.parameters
    profile_path = Path(parameters["mx.profile"])
    rtl = Path(parameters["mx.rtl-root"]) if "mx.rtl-root" in parameters else None
    profile = load_profile(profile_path, rtl_root=rtl)
    emission = emit_vpu_elementwise(request.source.text, profile, parameters["mx.buffers"])
    request.output.mkdir(parents=True)
    issuer, header, physical = write_vpu_sources(request.output, emission)
    abi = request.output / "buffer_abi.json"
    abi.write_text(json.dumps(emission.buffer_abi, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected_profile = request.output / "profile.json"
    selected_profile.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source = request.output / "input.mlir"
    source.write_text(request.source.text, encoding="utf-8")
    kinds = ((issuer, "c-source", "device-source"), (header, "c-header", "buffer-interface"),
             (physical, "json", "physical-program"), (abi, "json", "buffer-abi"),
             (selected_profile, "json", "selected-profile"), (source, "mlir", "input"))
    package = Path(__file__).parent
    result = Compilation(
        request.output, name, request.source.sha256,
        tuple(Artifact.from_file(path, kind=kind, role=role) for path, kind, role in kinds),
        metadata={
            "scope": "single typed VPU command source emission; no object, deployment or execution evidence",
            "profile_sha256": profile_sha256(profile), "qualification": profile["qualification"],
            "profile_source_checked": rtl is not None, "xdsl_version": version("xdsl"),
            "buffer_abi": emission.buffer_abi,
            "compiler_sources_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(package.glob("*.py"))
            },
        },
    )
    result.write()
    return result
