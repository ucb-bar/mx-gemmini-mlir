"""The ordinary installed boundary reuses real VPU emission and refuses other routes."""

from dataclasses import replace
from pathlib import Path

import pytest
from merlin.api import compile
from merlin.compiler.artifact import Compilation
from merlin.compiler.request import CompileOptions, CompileRequest, Source

from mx_gemmini_support import backend

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/gemmini-mx-cleanup-266c593/MxE4M3VpuGemminiRocketConfig.json"
CASE = ROOT / "docs/evidence/nicolas_vpu_elementwise_compiled_266c593/add"


def request(tmp_path):
    return CompileRequest(
        Source((CASE / "bound.mlir").read_text(), kind="mx-mlir"), tmp_path / "result",
        CompileOptions(parameters={"mx.profile": str(PROFILE), "mx.emit": "c-source",
                                   "mx.buffers": {"src1": "src1", "src2": "src2", "output": "output"}}),
    )


def test_public_service_emits_existing_source_with_content_bound_artifacts(tmp_path):
    selected = request(tmp_path)
    result = compile(selected.source, backend=backend, output=selected.output, options=selected.options)
    assert result.backend == "mx-gemmini" and result.deployment is None
    assert (result.directory / "mx_issue.c").read_text() == (CASE / "mx_issue.c").read_text()
    assert result.metadata["qualification"] == "structural_unqualified"
    assert result.metadata["profile_source_checked"] is False
    assert {item.kind for item in result.artifacts} == {"c-source", "c-header", "json", "mlir"}
    assert Compilation.load(result.manifest).to_dict() == result.to_dict()
    (result.directory / "mx_issue.c").write_text("changed")
    with pytest.raises(ValueError):
        result.verify()


def test_unsupported_options_and_shapes_refuse_before_publication(tmp_path, monkeypatch):
    selected = request(tmp_path)
    invalid = [
        replace(selected, source=replace(selected.source, kind="mlir")),
        replace(selected, options=replace(selected.options, harts=2)),
        replace(selected, options=CompileOptions(parameters={**selected.options.parameters, "mx.emit": "object"})),
        replace(selected, options=CompileOptions(parameters={**selected.options.parameters, "mx.unknown": True})),
        replace(selected, options=CompileOptions(parameters={**selected.options.parameters, "mx.buffers": {}})),
    ]
    for candidate in invalid:
        with pytest.raises(ValueError):
            backend.compile(candidate)
        assert not selected.output.exists()
    monkeypatch.setattr(backend, "version", lambda _name: "0.68.0")
    with pytest.raises(ValueError, match="0.65"):
        backend.validate(selected)
