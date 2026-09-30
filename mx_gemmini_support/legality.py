"""Shape rules shared by MX capture and the out-of-tree handoff."""

from __future__ import annotations

from collections.abc import Mapping


def shape_reason(contract: Mapping, fmt: str, m: int, n: int, k: int) -> str | None:
    """Return why a contraction is outside the selected RTL contract, if any."""
    formats = contract.get("formats") or {}
    if fmt not in formats:
        raise ValueError(f"{fmt}: format absent from selected MX contract")
    bounds = formats[fmt]["shape_bounds"]
    for axis, dim in (("M", m), ("N", n), ("K", k)):
        rule = bounds[axis]
        if type(dim) is not int or dim < rule["min"] or dim % rule["multiple_of"]:
            return f"{axis}={dim} outside {fmt} shape bounds"
    return None


def require_shape(contract: Mapping, fmt: str, m: int, n: int, k: int) -> None:
    reason = shape_reason(contract, fmt, m, n, k)
    if reason:
        raise ValueError(reason)
