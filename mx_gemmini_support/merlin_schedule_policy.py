"""Bind MX catalog issuer selection to Merlin's graph-level schedule dialect.

The dialect records the selected policy. MX owns the policy names and lowers
the selected policy to a checked ``mk`` kernel; Merlin owns neither ISA nor
target-specific tile choices.
"""

from __future__ import annotations

from merlin.xdsl_dialects import _common
from merlin.xdsl_dialects import schedule as s

_POLICIES = {"native": "mx_native_dram_loop", "movement": "mx_native_transfer_guarded_dram_loop",
             "serial": "mx_serial_commands",
             "queued": "mx_queued_commands",
             "pipelined": "mx_flash_cross_block_pipeline"}


def build_policy_plan(symbol: str, *, schedule: str = "native") -> str:
    """Print one verifiable schedule.plan bound to an exact catalog symbol."""
    from xdsl.dialects.builtin import ModuleOp, StringAttr
    from xdsl.ir import Block, Region

    if not symbol.isidentifier() or schedule not in _POLICIES:
        raise ValueError("MX schedule plan needs a safe symbol and known policy")
    block = Block()
    bind = s.BindOp(result_types=[s.HandleType()],
                    properties={"target": StringAttr("@" + symbol)})
    apply = s.ApplyPolicyOp(
        operands=[[bind.handle]],
        properties={"policy": StringAttr(_POLICIES[schedule])})
    block.add_ops([bind, apply])
    plan = s.PlanOp(
        regions=[Region([block])],
        properties={"sym_name": StringAttr(symbol + "_plan"),
                    "source": s.PolicySourceAttr(s.PolicySource.KERNEL_MINING)})
    module = ModuleOp([plan])
    module.verify()
    text = _common.text(module)
    if selected_policy(text, symbol=symbol) != schedule:
        raise ValueError("MX schedule plan did not round trip")
    return text


def selected_policy(text: str, *, symbol: str) -> str:
    """Read the schedule dialect decision that actual issuer lowering uses."""
    from xdsl.parser import Parser

    if not symbol.isidentifier():
        raise ValueError("MX schedule plan needs a safe symbol")
    module = Parser(_common.make_context(s.get_dialect()), text).parse_module()
    module.verify()
    plans = [op for op in module.body.block.ops if isinstance(op, s.PlanOp)]
    if len(plans) != 1 or plans[0].sym_name.data != symbol + "_plan":
        raise ValueError("MX schedule plan has no exact catalog binding")
    body = list(plans[0].body.block.ops)
    if (len(body) != 2 or not isinstance(body[0], s.BindOp) or
            not isinstance(body[1], s.ApplyPolicyOp) or
            body[0].target.data != "@" + symbol or
            list(body[1].handles) != [body[0].handle]):
        raise ValueError("MX schedule plan lost its exact target or policy handle")
    reverse = {v: k for k, v in _POLICIES.items()}
    policy = body[1].policy.data
    if policy not in reverse:
        raise ValueError("MX schedule plan names an unsupported policy")
    return reverse[policy]
