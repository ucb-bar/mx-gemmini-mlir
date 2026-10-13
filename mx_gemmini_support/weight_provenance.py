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
            value = op.operands[0]
        elif name == "linalg.generic":
            body = list(op.body.blocks[0].ops)
            if (len(op.inputs) != 1 or len(body) != 2
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
