"""Map one checked physical MX stream onto an issuer transport."""

from __future__ import annotations

from .command_ir import Command, Fence, WaitIdle
from .physical_program import PhysicalProgram


def issuer_commands(program: PhysicalProgram, transport: str) -> tuple[
        Command | Fence | WaitIdle, ...]:
    """Preserve the physical schedule and add Muon gateway completion waits.

    A CPU memory fence alone does not wait for Gemmini. Radiance's
    ``gemmini_fence`` polls the gateway busy register, so each physical
    completion point needs that wait after the CPU fence on Muon.
    """
    if transport not in {"rocket_rocc", "muon_mmio"}:
        raise ValueError("unknown MX issuer transport")
    if not program.steps or not isinstance(program.steps[-1].command, Fence):
        raise ValueError("MX physical program must finish at a completion fence")
    issued: list[Command | Fence | WaitIdle] = []
    for step in program.steps:
        issued.append(step.command)
        if transport == "muon_mmio" and isinstance(step.command, Fence):
            issued.append(WaitIdle())
    return tuple(issued)
