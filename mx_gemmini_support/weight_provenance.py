"""Conservative source trace for a checkpoint-backed MX matrix operand."""

from __future__ import annotations


def source_parameter_arg(value) -> tuple[int | None, tuple[str, ...], tuple[int, ...] | None]:
    """Follow only a transpose and exact BF16-to-F32 element conversion."""
    from xdsl.ir import BlockArgument

    path = []
    while not isinstance(value, BlockArgument):
        op = value.owner
        name = str(op.name)
        if name == "linalg.transpose":
            if tuple(op.permutation.get_values()) != (1, 0):
                return None, tuple(path + ["dynamic:transpose_permutation"]), None
            value = op.operands[0]
        elif name == "linalg.generic":
            from xdsl.ir.affine import AffineMap
            from xdsl.dialects.linalg.attrs import IteratorTypeAttr

            body = list(op.body.blocks[0].ops)
            shape = tuple(op.inputs[0].type.get_shape()) if len(op.inputs) == 1 else ()
            identity = AffineMap.identity(len(shape))
            if (len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1
                    or len(body) != 2
                    or len(shape) != 2 or tuple(op.results[0].type.get_shape()) != shape
                    or len(op.indexing_maps) != 2
                    or any(mapping.data != identity for mapping in op.indexing_maps)
                    or tuple(op.iterator_types) != (IteratorTypeAttr.parallel(),) * 2
                    or str(body[0].name) != "arith.extf"
                    or str(body[1].name) != "linalg.yield"
                    or str(body[0].operands[0].type) != "bf16"
                    or str(body[0].results[0].type) != "f32"
                    or body[0].operands[0] != op.body.blocks[0].args[0]
                    or body[1].operands[0] != body[0].results[0]):
                return None, tuple(path + ["dynamic:" + name]), None
            value = op.inputs[0]
        else:
            return None, tuple(path + ["dynamic:" + name]), None
        path.append(name)
    return value.index, tuple(path), tuple(value.type.get_shape())
