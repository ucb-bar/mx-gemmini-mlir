from __future__ import annotations

import pytest

from mx_gemmini_support.command_ir import Command, Operand, WaitIdle, emit_c


def test_rocket_and_muon_issue_same_command_fields():
    command = Command(27, Operand(buffer="scales", byte_offset=64), Operand(immediate=0x10020))
    rocket = emit_c([command], transport="rocket_rocc", buffers=("scales",))
    muon = emit_c([command], transport="muon_mmio", buffers=("scales",))
    assert ".insn r 0x7b, 3, 27" in rocket
    assert "mx_control_base + 0x10" in muon
    assert "mx_control_base + 0x18" in muon
    assert f"0x{command.instruction_word:08x}" in muon
    assert "(uintptr_t)scales + UINT64_C(64)" in rocket
    assert "(uintptr_t)scales + UINT64_C(64)" in muon


def test_issuer_refuses_unbound_or_truncated_operands():
    with pytest.raises(ValueError, match="seven-bit"):
        Command(128, Operand(immediate=0), Operand(immediate=0))
    with pytest.raises(ValueError, match="64 bits"):
        Operand(immediate=1 << 64)
    command = Command(27, Operand(buffer="scales"), Operand(immediate=0))
    with pytest.raises(ValueError, match="unbound"):
        emit_c([command], transport="rocket_rocc", buffers=())


def test_wait_uses_only_selected_muon_gateway_busy_register():
    source = emit_c([WaitIdle()], transport="muon_mmio", buffers=())
    assert "mx_control_base + 0x20" in source
    with pytest.raises(ValueError, match="no selected busy-register"):
        emit_c([WaitIdle()], transport="rocket_rocc", buffers=())
