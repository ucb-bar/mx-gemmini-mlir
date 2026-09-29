"""Digest-gated MX dialect handoff from a model2MLIR capture.

The output describes legal site operations and optional resident chaining. It
does not replace the source program or implement a target lowering/schedule.
"""

from __future__ import annotations

import hashlib
import json

from .contract import compile_contract, contract_digest
from .policy import load_policy


def _digest(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _string_attribute(module, name: str) -> str | None:
    attribute = module.attributes.get(name)
    return getattr(attribute, "data", None)


def validate_handoff(capture, contract_bytes: bytes, policy_bytes: bytes) -> dict:
    """Reject altered inputs, omitted census entries, and IR/manifest mismatches."""
    manifest = capture.quantization_manifest
    module = capture.module
    if not capture.ok or module is None or not isinstance(manifest, dict):
        raise ValueError("MX handoff needs a successful conversion and manifest")
    from xdsl.dialects.func import FuncOp

    external_calls = [op.sym_name.data for op in module.walk()
                      if isinstance(op, FuncOp) and op.is_declaration]
    if external_calls:
        raise ValueError(f"MX handoff source MLIR contains opaque calls: {sorted(external_calls)}")
    contract = compile_contract(contract_bytes)
    policy = load_policy(policy_bytes)
    required = {
        "schema": "m2m.quantization_manifest.v1", "adapter_id": "mx_gemmini",
        "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "policy_sha256": policy.source_sha256,
        "compiled_contract_sha256": contract_digest(contract),
        "numeric_status": "operand_fake_quant_only",
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise ValueError(f"MX handoff manifest {key} differs from selected input")
    digest = _digest(manifest)
    if _string_attribute(module, "prov.quantization_manifest_sha256") != digest:
        raise ValueError("MLIR module is not bound to selected quantization manifest")
    if _string_attribute(module, "prov.quantization") != "external:mx_gemmini":
        raise ValueError("MLIR module selects a different quantization adapter")
    sites = manifest.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("MX handoff requires a nonempty site census")
    by_site = {}
    for site in sites:
        if not isinstance(site, dict) or not isinstance(site.get("site_id"), str):
            raise ValueError("invalid MX handoff site")
        site_id = site["site_id"]
        if site_id in by_site:
            raise ValueError("duplicate MX handoff site ID")
        by_site[site_id] = site
        if site.get("status") == "quantized":
            fmt = site.get("format")
            if fmt not in contract["formats"]:
                raise ValueError(f"{site_id}: format absent from selected RTL contract")
            shape = site.get("shape")
            if not isinstance(shape, list) or len(shape) not in (2, 3):
                raise ValueError(f"{site_id}: missing selected site shape")
            bounds = contract["formats"][fmt]["shape_bounds"]
            axes = ("N", "K") if len(shape) == 2 else ("M", "N", "K")
            for axis, dim in zip(axes, shape):
                rule = bounds[axis]
                if type(dim) is not int or dim < rule["min"] or dim % rule["multiple_of"]:
                    raise ValueError(f"{site_id}: {axis} outside selected MX shape bounds")
            if fmt == "mxfp6" and site.get("fp6_codebook_sha256") != _digest(policy.codebooks(site_id)):
                raise ValueError(f"{site_id}: reviewed FP6 codebook binding differs")
        elif site.get("status") == "skipped":
            if not site.get("reason"):
                raise ValueError(f"{site_id}: skipped site lacks reason")
        elif site.get("status") != "host":
            raise ValueError(f"{site_id}: missing explicit site disposition")
    for source, chain in policy.output_chains.items():
        producer, consumer = by_site.get(source), by_site.get(chain["consumer"])
        if not producer or not consumer or producer.get("status") != "quantized" or consumer.get("status") != "quantized":
            raise ValueError("MX output chain has an absent or nonquantized endpoint")
        if producer.get("output_chain") != chain or consumer.get("format") != chain["format"]:
            raise ValueError("MX output chain differs from selected policy")
        if producer["shape"][-2] != consumer["shape"][-1]:
            raise ValueError("MX output chain dimensions differ")
        if chain["format"] == "mxfp6":
            # The resident output becomes this consumer's activation operand.
            output_book = policy.output_codebook(source)
            if output_book != policy.codebooks(chain["consumer"])[0]:
                raise ValueError("MX resident FP6 output LUT differs from consumer activation LUT")
            if producer.get("output_fp6_codebook_sha256") != _digest(output_book):
                raise ValueError("MX resident FP6 codebook digest differs from selected policy")
    return {"contract": contract, "manifest": manifest, "policy": policy,
            "manifest_sha256": digest,
            "source_mlir_sha256": hashlib.sha256(capture.mlir_text.encode()).hexdigest()}


def render_handoff(capture, contract_bytes: bytes, policy_bytes: bytes) -> str:
    """Emit a verifier-readable MX operation plan for the selected capture."""
    bound = validate_handoff(capture, contract_bytes, policy_bytes)
    manifest, contract, policy = bound["manifest"], bound["contract"], bound["policy"]
    contract_sha = contract_digest(contract)
    policy_sha = policy.source_sha256
    manifest_sha = bound["manifest_sha256"]
    source_sha = bound["source_mlir_sha256"]
    attrs = (f'contract_sha256 = "{contract_sha}", policy_sha256 = "{policy_sha}", '
             f'manifest_sha256 = "{manifest_sha}"')
    lines = [f'module attributes {{mx.contract_sha256 = "{contract_sha}", '
             f'mx.policy_sha256 = "{policy_sha}", '
             f'prov.quantization_manifest_sha256 = "{manifest_sha}", '
             f'mx.source_mlir_sha256 = "{source_sha}"}} {{']
    tensor_codes = "tensor<?x?xi8>"
    tensor_scales = "tensor<?x?xi8>"
    tensor_acc = "tensor<?x?xbf16>"
    by_site = {row["site_id"]: row for row in manifest["sites"]}
    for index, site in enumerate(manifest["sites"]):
        if site["status"] != "quantized":
            continue
        site_id = site["site_id"]
        fmt = site["format"]
        local = f'site_id = "{site_id}", format = "{fmt}", {attrs}'
        lines.append(f'  func.func @site_{index}(%a: {tensor_codes}, %as: {tensor_scales}, '
                     f'%b: {tensor_codes}, %bs: {tensor_scales}) -> {tensor_acc} {{')
        lines.append('    %acc = "mx_gemmini.contract"(%a, %as, %b, %bs) {' + local +
                     f'}} : ({tensor_codes}, {tensor_scales}, {tensor_codes}, {tensor_scales}) -> {tensor_acc}')
        if site.get("output_chain"):
            chain = site["output_chain"]
            target = by_site[chain["consumer"]]
            if target["format"] != chain["format"]:
                raise ValueError("output chain consumer format differs")
            lines.append('    %codes, %scales = "mx_gemmini.requantize"(%acc) {' +
                         f'site_id = "{site_id}", format = "{chain["format"]}", {attrs}' +
                         f'}} : ({tensor_acc}) -> ({tensor_codes}, {tensor_scales})')
        lines.append('    %out = "mx_gemmini.readout_bf16"(%acc) {' +
                     f'site_id = "{site_id}", {attrs}' +
                     f'}} : ({tensor_acc}) -> {tensor_acc}')
        lines.append(f'    func.return %out : {tensor_acc}')
        lines.append('  }')
        if site.get("output_chain"):
            chain = site["output_chain"]
            consumer = by_site[chain["consumer"]]
            next_local = (f'site_id = "{chain["consumer"]}", format = "{consumer["format"]}", '
                          f'{attrs}')
            lines.append(f'  func.func @chain_{index}(%a: {tensor_codes}, %as: {tensor_scales}, '
                         f'%b: {tensor_codes}, %bs: {tensor_scales}, '
                         f'%next_b: {tensor_codes}, %next_bs: {tensor_scales}) -> {tensor_acc} {{')
            lines.append('    %first = "mx_gemmini.contract"(%a, %as, %b, %bs) {' + local +
                         f'}} : ({tensor_codes}, {tensor_scales}, {tensor_codes}, {tensor_scales}) -> {tensor_acc}')
            lines.append('    %resident, %resident_scales = "mx_gemmini.requantize"(%first) {' +
                         f'site_id = "{site_id}", format = "{chain["format"]}", {attrs}' +
                         f'}} : ({tensor_acc}) -> ({tensor_codes}, {tensor_scales})')
            lines.append('    %next = "mx_gemmini.contract"(%resident, %resident_scales, %next_b, %next_bs) {' +
                         next_local +
                         f'}} : ({tensor_codes}, {tensor_scales}, {tensor_codes}, {tensor_scales}) -> {tensor_acc}')
            lines.append('    %read = "mx_gemmini.readout_bf16"(%next) {' +
                         f'site_id = "{chain["consumer"]}", {attrs}' +
                         f'}} : ({tensor_acc}) -> {tensor_acc}')
            lines.append(f'    func.return %read : {tensor_acc}')
            lines.append('  }')
    lines.append('}')
    return "\n".join(lines) + "\n"
