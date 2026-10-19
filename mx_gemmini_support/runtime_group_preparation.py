"""Bind generic word-group preparation to the selected MX conversion policy."""

from __future__ import annotations


def emit_runtime_group_preparation(elements: int = 32) -> str:
    from merlin.llvmlower.word_group_preparation import emit_u16_group_preparation

    if type(elements) is not int or elements != 32:
        raise ValueError("MX runtime quantization requires its selected 32-element scale contract")
    return emit_u16_group_preparation(elements, symbol="mx_prepare_group_words",
                                      conversion="mx_round_group_word", magnitude="mx_group_word_magnitude")
