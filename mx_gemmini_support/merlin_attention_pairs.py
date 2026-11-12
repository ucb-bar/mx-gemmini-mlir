"""Compatibility entry point for Merlin's generic attention dataflow analysis."""

from __future__ import annotations


def find_attention_pairs(inventory: dict) -> list[dict]:
    """Delegate source relationships; target legality and precision remain external."""
    from merlin.compiler.frontend.attention_dataflow import find_attention_pairs as analyze

    return analyze(inventory)
