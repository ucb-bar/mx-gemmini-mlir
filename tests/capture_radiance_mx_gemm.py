"""Capture a Radiance MX GEMM shape through current model2MLIR and the MX adapter.

This is a structural frontend/handoff test. The PyTorch inputs do not reproduce
the handwritten FP8 code and scale blobs; native MX output must later be
checked against radiance-kernels' mx_golden rather than ideal torch.matmul.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SOURCE_REVISION = "a27f6abd24830fdc7999d872d170ab778f1e662e"
SHAPE = (64, 64, 64)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha(root: Path, package: str) -> str:
    files = sorted((root / package).rglob("*.py"))
    record = {str(path.relative_to(root)): sha(path) for path in files}
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model2mlir-root", type=Path, required=True)
    parser.add_argument("--mxq-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mx-opt", type=Path, required=True)
    parser.add_argument("--radiance-opt", type=Path, required=True)
    args = parser.parse_args()
    m2m_root = args.model2mlir_root.resolve()
    mxq_root = args.mxq_root.resolve()
    source_root = args.source_root.resolve()
    support_root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if git(source_root, "rev-parse", "HEAD") != SOURCE_REVISION:
        parser.error("radiance-kernels source differs from the pinned baseline")
    for root in (m2m_root, mxq_root, support_root):
        sys.path.insert(0, str(root))
    import m2m
    import mxq
    import torch
    from m2m.capture.external_quantization import ExternalQuantizationConfig
    from m2m.coverage import opaque_report
    from mx_gemmini_support.handoff import render_handoff, validate_handoff

    if Path(m2m.__file__).resolve().parents[1] != m2m_root:
        raise RuntimeError("model2MLIR resolved to a different checkout")
    if Path(mxq.__file__).resolve().parents[1] != mxq_root:
        raise RuntimeError("microscaling-quant resolved to a different checkout")
    generator = source_root / "kernels/gemm_mxgemmini/gen_mxgemm_data.py"
    spec = importlib.util.spec_from_file_location("radiance_mx_data_generator", generator)
    assert spec and spec.loader
    source_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source_data)
    if SHAPE not in source_data.MISSING_FP8:
        raise RuntimeError("the chosen FP8 GEMM shape is absent from source generator")

    class Gemm(torch.nn.Module):
        def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
            return torch.matmul(lhs, rhs)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        lhs = torch.randn((SHAPE[0], SHAPE[2]), dtype=torch.float32)
        rhs = torch.randn((SHAPE[2], SHAPE[1]), dtype=torch.float32)
    contract = support_root / "mx_gemmini_support/contracts/software-spec-2029218-candidate.yaml"
    policy = support_root / "examples/default-policy.yaml"
    result = m2m.convert(
        Gemm().eval(), (lhs, rhs),
        quantization=ExternalQuantizationConfig("mx_gemmini", contract, policy),
        backend="fx_importer", capture_trace=True,
    )
    if not result.ok:
        raise RuntimeError(f"model2MLIR failed: {result.diagnostics}")
    opaque = opaque_report(result.mlir_text)
    if opaque:
        raise RuntimeError(f"model2MLIR left opaque calls: {opaque}")
    sites = result.quantization_manifest["sites"]
    if len(sites) != 1 or any(sites[0].get(key) != value for key, value in {
            "site_id": "functional:matmul", "status": "quantized",
            "format": "mxfp8", "shape": list(SHAPE)}.items()):
        raise RuntimeError(f"MX contraction site was not selected: {sites}")
    contract_bytes, policy_bytes = contract.read_bytes(), policy.read_bytes()
    validate_handoff(result, contract_bytes, policy_bytes)
    source_mlir = out / "mx_gemm.model2mlir.mlir"
    source_mlir.write_text(result.mlir_text)
    handoff = out / "mx_gemm.handoff.mlir"
    handoff.write_text(render_handoff(result, contract_bytes, policy_bytes))
    for name, tool in (("mx", args.mx_opt), ("radiance", args.radiance_opt)):
        completed = subprocess.run([str(tool.resolve()), str(handoff), "-o", "/dev/null"],
                                   capture_output=True, text=True)
        (out / f"{name}_parse.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{name} dialect rejected handoff; see {name}_parse.log")
    (out / "quantization_manifest.json").write_text(
        json.dumps(result.quantization_manifest, indent=2) + "\n")
    receipt = {
        "schema": "mx_gemmini_model2mlir_radiance_gemm_capture.v1",
        "status": "source_shape_frontend_handoff_only",
        "source_revision": SOURCE_REVISION,
        "source_generator_sha256": sha(generator),
        "source_shape": list(SHAPE),
        "model2mlir_revision": git(m2m_root, "rev-parse", "HEAD"),
        "model2mlir_source_tree_sha256": tree_sha(m2m_root, "m2m"),
        "mxq_revision": git(mxq_root, "rev-parse", "HEAD"),
        "mxq_source_tree_sha256": tree_sha(mxq_root, "mxq"),
        "mx_support_revision": git(support_root, "rev-parse", "HEAD"),
        "contract_sha256": sha(contract), "policy_sha256": sha(policy),
        "source_mlir_sha256": sha(source_mlir), "handoff_mlir_sha256": sha(handoff),
        "quantization_manifest_sha256": sha(out / "quantization_manifest.json"),
        "selected_site": sites[0], "opaque_calls": opaque,
        "numerical_scope": "PyTorch inputs are not source FP8 blobs; no MX numerical parity claimed",
    }
    destination = out / "receipt.json"
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"site": sites[0]["site_id"], "handoff": str(handoff),
                      "receipt": str(destination)}))


if __name__ == "__main__":
    main()
