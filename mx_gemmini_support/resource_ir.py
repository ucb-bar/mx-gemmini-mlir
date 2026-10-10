"""Connect checked external MX resources to typed source-specialized MLIR."""

from __future__ import annotations

from .source_payload import manifest_json, manifest_sha256


SCHEMA = "source_resources_ssa_v1"
INPUTS = ("activation", "activation_scales", "weight", "weight_scales")
LUTS = ("activation_lut", "weight_lut", "output_lut")


def attach_source_resources(module, contract, manifest: dict) -> dict:
    """Wire packed codes/scales to the contraction and LUT banks to uploads.

    The binary arrays stay in the source bundle. Each typed resource result
    carries the bundle name, layout, and byte digest, and the module embeds
    the complete canonical manifest. Physical lowering checks both before
    materializing the actual bytes into its standalone object.
    """
    from xdsl.dialects.builtin import StringAttr, TensorType, UnregisteredOp, i8, i16, i32
    from xdsl.dialects.func import FuncOp

    block = contract.parent
    if block is None or len(contract.operands) != 4:
        raise ValueError("source resources require a four-operand MX contraction")
    digest = manifest_sha256(manifest)
    if module.attributes.get("mx.payload_manifest_sha256") != StringAttr(digest):
        raise ValueError("source resource manifest differs from module binding")
    if contract.attributes.get("payload_manifest_sha256") != StringAttr(digest):
        raise ValueError("source resource manifest differs from contraction binding")
    descriptions = manifest["resources"]
    if not set(INPUTS) <= set(descriptions):
        raise ValueError("source resource manifest lacks contraction operands")
    module.attributes["mx.payload_manifest_json"] = StringAttr(manifest_json(manifest))
    module.attributes["mx.payload_binding_schema"] = StringAttr(SCHEMA)
    binding = {name: contract.attributes[name] for name in (
        "site_id", "profile_sha256", "contract_sha256", "policy_sha256",
        "manifest_sha256", "payload_manifest_sha256")}
    element_types = {8: i8, 16: i16, 32: i32}
    produced = {}
    for name in (*INPUTS, *LUTS):
        if name not in descriptions:
            continue
        descriptor = descriptions[name]
        bits = descriptor["element_bits"]
        if bits not in element_types:
            raise ValueError(f"source resource {name} has an unknown element width")
        resource = UnregisteredOp.with_name("mx_gemmini.resource").create(
            result_types=[TensorType(element_types[bits], descriptor["shape"])],
            attributes={**binding,
                        "resource_name": StringAttr(name),
                        "resource_sha256": StringAttr(descriptor["sha256"]),
                        "resource_layout": StringAttr(descriptor["layout"])})
        block.insert_op_before(resource, contract)
        produced[name] = resource.results[0]
    for operand_index, name in enumerate(INPUTS):
        contract.operands[operand_index] = produced[name]
    for name in LUTS:
        if name not in produced:
            continue
        upload = UnregisteredOp.with_name("mx_gemmini.upload_lut").create(
            operands=[produced[name]],
            attributes={**binding, "lut_target": StringAttr(name.removesuffix("_lut"))})
        block.insert_op_before(upload, contract)
    # A source-specialized function reads these closed resources rather than
    # accepting unused captured tensors from a caller.
    function = contract.parent_op()
    if not isinstance(function, FuncOp):
        raise ValueError("source resource contraction must belong to a function")
    for argument in tuple(block.args):
        if not argument.uses:
            block.erase_arg(argument)
    function.update_function_type()
    return produced
